"""
Slock platform adapter for Hermes gateway.

Connects to a Slock workspace via REST API:
- Inbound: long-poll GET /internal/agent/:id/receive
- Outbound: POST /internal/agent/:id/send
- Uploads: POST /internal/agent/:id/upload
"""

import asyncio
import json
import logging
import os
import random
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import (
    BasePlatformAdapter,
    MessageEvent,
    MessageType,
    SendResult,
    cache_document_from_bytes,
    cache_image_from_bytes,
)

logger = logging.getLogger(__name__)

_RECONNECT_BASE_DELAY = 2.0
_RECONNECT_MAX_DELAY = 60.0
_RECONNECT_JITTER = 0.3
_POLL_TIMEOUT_MS = 30_000
MAX_MESSAGE_LENGTH = 16_000


def check_slock_requirements() -> bool:
    """Check if Slock platform dependencies are available."""
    try:
        import aiohttp  # noqa: F401
    except ImportError:
        logger.warning("Slock: aiohttp not installed. Run: pip install aiohttp")
        return False

    server_url = os.getenv("SLOCK_SERVER_URL", "")
    machine_token = os.getenv("SLOCK_MACHINE_TOKEN", "")
    agent_id = os.getenv("SLOCK_AGENT_ID", "")

    if not server_url:
        logger.warning("Slock: SLOCK_SERVER_URL not set")
        return False
    if not machine_token:
        logger.warning("Slock: SLOCK_MACHINE_TOKEN not set")
        return False
    if not agent_id:
        logger.warning("Slock: SLOCK_AGENT_ID not set")
        return False

    return True


class SlockAdapter(BasePlatformAdapter):
    """Hermes gateway adapter for Slock workspaces."""

    MAX_MESSAGE_LENGTH = MAX_MESSAGE_LENGTH

    def __init__(self, config: PlatformConfig):
        super().__init__(config, Platform.SLOCK)

        extra = config.extra if hasattr(config, "extra") and config.extra else {}

        self._server_url = (
            extra.get("server_url") or os.getenv("SLOCK_SERVER_URL", "")
        ).rstrip("/")
        self._machine_token = (
            extra.get("machine_token")
            or config.token
            or os.getenv("SLOCK_MACHINE_TOKEN", "")
        )
        self._agent_id = extra.get("agent_id") or os.getenv("SLOCK_AGENT_ID", "")

        self._session: Optional["aiohttp.ClientSession"] = None
        self._poll_task: Optional[asyncio.Task] = None
        self._closing = False

        # channel_id (UUID) → target string mapping for sending replies
        self._channel_targets: Dict[str, str] = {}
        # Track seen message seqs for dedup
        self._seen_seqs: set = set()
        self._max_seen = 0

    # ------------------------------------------------------------------
    # HTTP helpers
    # ------------------------------------------------------------------

    def _headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self._machine_token}",
            "Content-Type": "application/json",
        }

    def _api_url(self, path: str) -> str:
        return f"{self._server_url}/internal/agent/{self._agent_id}/{path}"

    async def _api_get(self, path: str, **kwargs) -> Dict[str, Any]:
        import aiohttp

        url = self._api_url(path)
        async with self._session.get(
            url,
            headers=self._headers(),
            timeout=aiohttp.ClientTimeout(total=60),
            **kwargs,
        ) as resp:
            if resp.status >= 400:
                body = await resp.text()
                raise RuntimeError(f"Slock API GET {path} failed ({resp.status}): {body}")
            return await resp.json()

    async def _api_post(self, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        import aiohttp

        url = self._api_url(path)
        async with self._session.post(
            url,
            headers=self._headers(),
            json=payload,
            timeout=aiohttp.ClientTimeout(total=30),
        ) as resp:
            if resp.status >= 400:
                body = await resp.text()
                raise RuntimeError(f"Slock API POST {path} failed ({resp.status}): {body}")
            return await resp.json()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def connect(self) -> bool:
        import aiohttp

        if not self._server_url or not self._machine_token or not self._agent_id:
            logger.error("Slock: missing SLOCK_SERVER_URL, SLOCK_MACHINE_TOKEN, or SLOCK_AGENT_ID")
            return False

        self._closing = False
        self._session = aiohttp.ClientSession()

        # Verify connectivity with a non-blocking receive call
        try:
            url = self._api_url("receive")
            async with self._session.get(
                url,
                headers=self._headers(),
                params={"block": "false"},
                timeout=aiohttp.ClientTimeout(total=10),
            ) as resp:
                if resp.status in (401, 403):
                    body = await resp.text()
                    logger.error("Slock: auth failed (%d): %s", resp.status, body)
                    self._set_fatal_error(
                        "auth_failed",
                        f"Slock authentication failed (HTTP {resp.status})",
                        retryable=False,
                    )
                    await self._notify_fatal_error()
                    await self._session.close()
                    return False
                if resp.status >= 400:
                    body = await resp.text()
                    logger.error("Slock: connectivity check failed (%d): %s", resp.status, body)
                    await self._session.close()
                    return False
                data = await resp.json()
                inbox_count = len(data.get("messages", []))
                logger.info("Slock: connected (agent=%s, %d pending messages)", self._agent_id[:8], inbox_count)
        except Exception as exc:
            logger.error("Slock: failed to connect to %s: %s", self._server_url, exc)
            await self._session.close()
            return False

        self._mark_connected()
        self._poll_task = asyncio.create_task(self._poll_loop())
        return True

    async def disconnect(self) -> None:
        self._closing = True
        if self._poll_task and not self._poll_task.done():
            self._poll_task.cancel()
            try:
                await self._poll_task
            except asyncio.CancelledError:
                pass
        self._poll_task = None

        if self._session and not self._session.closed:
            await self._session.close()
        self._session = None
        self._mark_disconnected()
        logger.info("Slock: disconnected")

    # ------------------------------------------------------------------
    # Long-poll inbound
    # ------------------------------------------------------------------

    async def _poll_loop(self) -> None:
        delay = _RECONNECT_BASE_DELAY
        while not self._closing:
            try:
                await self._poll_once()
                delay = _RECONNECT_BASE_DELAY
            except asyncio.CancelledError:
                return
            except Exception as exc:
                if self._closing:
                    return
                err_str = str(exc).lower()
                if "401" in err_str or "403" in err_str or "unauthorized" in err_str:
                    logger.error("Slock: permanent auth error: %s — stopping poll", exc)
                    self._set_fatal_error("auth_failed", str(exc), retryable=False)
                    await self._notify_fatal_error()
                    return
                logger.warning("Slock: poll error: %s — retrying in %.0fs", exc, delay)
                if self._closing:
                    return
                jitter = delay * _RECONNECT_JITTER * random.random()
                await asyncio.sleep(delay + jitter)
                delay = min(delay * 2, _RECONNECT_MAX_DELAY)

    async def _poll_once(self) -> None:
        import aiohttp

        url = self._api_url("receive")
        params = {"block": "true", "timeout": str(_POLL_TIMEOUT_MS)}

        async with self._session.get(
            url,
            headers=self._headers(),
            params=params,
            timeout=aiohttp.ClientTimeout(total=_POLL_TIMEOUT_MS / 1000 + 10),
        ) as resp:
            if resp.status >= 400:
                body = await resp.text()
                raise RuntimeError(f"receive failed ({resp.status}): {body}")
            data = await resp.json()

        messages = data.get("messages") or []
        for msg in messages:
            await self._handle_inbound(msg)

    async def _handle_inbound(self, msg: Dict[str, Any]) -> None:
        seq = msg.get("seq", 0)
        if seq and seq in self._seen_seqs:
            return
        if seq:
            self._seen_seqs.add(seq)
            if len(self._seen_seqs) > 5000:
                cutoff = self._max_seen - 4000
                self._seen_seqs = {s for s in self._seen_seqs if s > cutoff}
            self._max_seen = max(self._max_seen, seq)

        sender_type = msg.get("sender_type", "user")
        if sender_type == "agent":
            return

        channel_id = msg.get("channel_id", "")
        channel_name = msg.get("channel_name", "")
        channel_type = msg.get("channel_type", "channel")
        sender_id = msg.get("sender_id", "")
        sender_name = msg.get("sender_name", "")
        content = msg.get("content", "")
        message_id = msg.get("message_id", "")
        parent_channel_name = msg.get("parent_channel_name", "")
        parent_channel_id = msg.get("parent_channel_id", "")

        target = self._build_target(channel_type, channel_name, parent_channel_name, message_id, sender_name)
        self._channel_targets[channel_id] = target

        if channel_type == "thread":
            chat_type = "group"
            thread_id = channel_id
            chat_id = channel_id
        elif channel_type == "dm":
            chat_type = "dm"
            thread_id = None
            chat_id = channel_id
        else:
            chat_type = "channel"
            thread_id = None
            chat_id = channel_id

        msg_type = MessageType.TEXT
        if content.startswith("/"):
            msg_type = MessageType.COMMAND

        media_urls: List[str] = []
        media_types: List[str] = []
        attachments = msg.get("attachments") or []
        for att in attachments:
            att_id = att.get("id", "")
            filename = att.get("filename", f"file_{att_id}")
            mime = att.get("mimeType", "application/octet-stream")
            try:
                file_data = await self._download_attachment(att_id)
                if file_data:
                    ext = Path(filename).suffix or ""
                    if mime.startswith("image/"):
                        local_path = cache_image_from_bytes(file_data, ext or ".png")
                    else:
                        local_path = cache_document_from_bytes(file_data, filename)
                    media_urls.append(local_path)
                    media_types.append(mime)
            except Exception as exc:
                logger.warning("Slock: failed to download attachment %s: %s", att_id, exc)

        if media_types and msg_type == MessageType.TEXT:
            if any(m.startswith("image/") for m in media_types):
                msg_type = MessageType.PHOTO
            elif any(m.startswith("audio/") for m in media_types):
                msg_type = MessageType.VOICE
            elif media_types:
                msg_type = MessageType.DOCUMENT

        source = self.build_source(
            chat_id=chat_id,
            chat_name=channel_name,
            chat_type=chat_type,
            user_id=sender_id,
            user_name=sender_name,
            thread_id=thread_id,
        )

        event = MessageEvent(
            text=content,
            message_type=msg_type,
            source=source,
            raw_message=msg,
            message_id=message_id or str(seq),
            media_urls=media_urls if media_urls else None,
            media_types=media_types if media_types else None,
        )

        await self.handle_message(event)

    async def _download_attachment(self, attachment_id: str) -> Optional[bytes]:
        import aiohttp

        url = f"{self._server_url}/internal/agent/{self._agent_id}/attachment/{attachment_id}"
        async with self._session.get(
            url,
            headers={"Authorization": f"Bearer {self._machine_token}"},
            timeout=aiohttp.ClientTimeout(total=30),
        ) as resp:
            if resp.status >= 400:
                logger.warning("Slock: attachment download failed (%d)", resp.status)
                return None
            return await resp.read()

    # ------------------------------------------------------------------
    # Target string helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_target(
        channel_type: str,
        channel_name: str,
        parent_channel_name: str,
        message_id: str,
        sender_name: str,
    ) -> str:
        if channel_type == "thread":
            parent = parent_channel_name or channel_name
            tid = channel_name[7:] if channel_name.startswith("thread-") else channel_name
            return f"#{parent}:{tid}" if tid else f"#{parent}"
        elif channel_type == "dm":
            return f"dm:@{sender_name}" if sender_name else f"dm:{channel_name}"
        else:
            return f"#{channel_name}"

    def _resolve_target(self, chat_id: str, thread_id: Optional[str] = None) -> str:
        if thread_id and thread_id in self._channel_targets:
            return self._channel_targets[thread_id]
        if chat_id in self._channel_targets:
            target = self._channel_targets[chat_id]
            if thread_id:
                base = target.split(":")[0] if ":" in target else target
                return f"{base}:{thread_id[:8]}"
            return target
        if chat_id.startswith("#") or chat_id.startswith("dm:"):
            return chat_id
        return f"#{chat_id}"

    # ------------------------------------------------------------------
    # Send
    # ------------------------------------------------------------------

    async def send(
        self,
        chat_id: str,
        content: str,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        metadata = metadata or {}
        thread_id = metadata.get("thread_id")
        target = self._resolve_target(chat_id, thread_id)

        chunks = self.truncate_message(content, self.MAX_MESSAGE_LENGTH)
        last_msg_id = None

        for chunk in chunks:
            try:
                payload: Dict[str, Any] = {"target": target, "content": chunk}
                resp = await self._api_post("send", payload)
                if resp.get("ok"):
                    last_msg_id = resp.get("messageId")
                else:
                    error = resp.get("error", "unknown error")
                    return SendResult(success=False, error=f"Slock send failed: {error}")
            except Exception as exc:
                return SendResult(success=False, error=str(exc), retryable=True)

        return SendResult(success=True, message_id=last_msg_id)

    async def send_typing(self, chat_id: str, metadata=None) -> None:
        pass

    async def send_image(
        self,
        chat_id: str,
        image_url: str,
        caption: Optional[str] = None,
    ) -> SendResult:
        return await self._send_file_from_url(chat_id, image_url, caption)

    async def send_document(
        self,
        chat_id: str,
        file_path: str,
        caption: Optional[str] = None,
    ) -> SendResult:
        return await self._send_local_file(chat_id, file_path, caption)

    async def send_image_file(
        self,
        chat_id: str,
        file_path: str,
        caption: Optional[str] = None,
    ) -> SendResult:
        return await self._send_local_file(chat_id, file_path, caption)

    # ------------------------------------------------------------------
    # File upload helpers
    # ------------------------------------------------------------------

    async def _upload_file(
        self, chat_id: str, file_data: bytes, filename: str
    ) -> Optional[str]:
        import aiohttp
        from aiohttp import FormData

        target = self._resolve_target(chat_id)

        try:
            channel_resp = await self._api_post("resolve-channel", {"target": target})
            channel_id = channel_resp.get("channelId")
            if not channel_id:
                logger.warning("Slock: could not resolve channel for target %s", target)
                return None
        except Exception as exc:
            logger.warning("Slock: resolve-channel failed: %s", exc)
            return None

        import mimetypes
        content_type = mimetypes.guess_type(filename)[0] or "application/octet-stream"

        url = self._api_url("upload")
        form = FormData()
        form.add_field("file", file_data, filename=filename, content_type=content_type)
        form.add_field("channelId", channel_id)

        async with self._session.post(
            url,
            headers={"Authorization": f"Bearer {self._machine_token}"},
            data=form,
            timeout=aiohttp.ClientTimeout(total=60),
        ) as resp:
            if resp.status >= 400:
                body = await resp.text()
                logger.warning("Slock: upload failed (%d): %s", resp.status, body)
                return None
            data = await resp.json()
            return data.get("id")

    async def _send_file_from_url(
        self, chat_id: str, file_url: str, caption: Optional[str] = None
    ) -> SendResult:
        import aiohttp

        try:
            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=30)
            ) as dl_session:
                async with dl_session.get(file_url) as resp:
                    if resp.status >= 400:
                        return SendResult(
                            success=False,
                            error=f"Failed to download {file_url}: HTTP {resp.status}",
                        )
                    file_data = await resp.read()
                    content_type = resp.content_type or ""
                    # Derive filename from URL
                    url_path = file_url.split("?")[0].split("/")[-1]
                    filename = url_path if "." in url_path else "file"
        except Exception as exc:
            return SendResult(success=False, error=f"Download failed: {exc}")

        att_id = await self._upload_file(chat_id, file_data, filename)
        if not att_id:
            return SendResult(success=False, error="File upload failed")

        target = self._resolve_target(chat_id)
        payload: Dict[str, Any] = {
            "target": target,
            "content": caption or "",
            "attachmentIds": [att_id],
        }
        try:
            resp_data = await self._api_post("send", payload)
            if resp_data.get("ok"):
                return SendResult(success=True, message_id=resp_data.get("messageId"))
            return SendResult(success=False, error=resp_data.get("error", "send failed"))
        except Exception as exc:
            return SendResult(success=False, error=str(exc))

    async def _send_local_file(
        self, chat_id: str, file_path: str, caption: Optional[str] = None
    ) -> SendResult:
        path = Path(file_path)
        if not path.exists():
            return SendResult(success=False, error=f"File not found: {file_path}")

        file_data = path.read_bytes()
        att_id = await self._upload_file(chat_id, file_data, path.name)
        if not att_id:
            return SendResult(success=False, error="File upload failed")

        target = self._resolve_target(chat_id)
        payload: Dict[str, Any] = {
            "target": target,
            "content": caption or "",
            "attachmentIds": [att_id],
        }
        try:
            resp_data = await self._api_post("send", payload)
            if resp_data.get("ok"):
                return SendResult(success=True, message_id=resp_data.get("messageId"))
            return SendResult(success=False, error=resp_data.get("error", "send failed"))
        except Exception as exc:
            return SendResult(success=False, error=str(exc))

    # ------------------------------------------------------------------
    # Chat info
    # ------------------------------------------------------------------

    async def get_chat_info(self, chat_id: str) -> Dict[str, Any]:
        target = self._resolve_target(chat_id)
        chat_type = "dm" if target.startswith("dm:") else "channel"
        name = target.lstrip("#").split(":")[0] if not target.startswith("dm:") else target
        return {"name": name, "type": chat_type, "chat_id": chat_id}
