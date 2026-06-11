"""Raft wake endpoint platform adapter.

This adapter is intentionally narrow: it receives content-free wake hints from
the Raft bridge and injects a synthetic message into Hermes' normal gateway
session pipeline. The bridge remains responsible for Raft message cursors and
body materialization; Hermes tells the agent to run ``raft message check``.
"""

from __future__ import annotations

import asyncio
import hmac
import json
import logging
import time
from typing import Any, Dict, Optional

try:
    from aiohttp import web

    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False
    web = None  # type: ignore[assignment]

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import (
    BasePlatformAdapter,
    MessageEvent,
    MessageType,
    SendResult,
    merge_pending_message_event,
)
from gateway.session import build_session_key

logger = logging.getLogger(__name__)

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8646
DEFAULT_PATH = "/wake"
DEFAULT_RUNTIME_SESSION = "default"
DEFAULT_MAX_BODY_BYTES = 16_384
BRIDGE_TOKEN_HEADER = "x-slock-bridge-token"

_CONTENT_FIELD_NAMES = {
    "body",
    "content",
    "message",
    "messages",
    "preview",
    "snippet",
    "text",
}


def check_slock_requirements() -> bool:
    """Check if Raft wake endpoint dependencies are available."""
    return AIOHTTP_AVAILABLE


def _path_value(value: Any) -> str:
    path = str(value or DEFAULT_PATH).strip() or DEFAULT_PATH
    if not path.startswith("/"):
        path = f"/{path}"
    return path


def _has_content_field(value: Any) -> bool:
    if isinstance(value, dict):
        for key, nested in value.items():
            if str(key).strip().lower() in _CONTENT_FIELD_NAMES:
                return True
            if _has_content_field(nested):
                return True
    elif isinstance(value, list):
        return any(_has_content_field(item) for item in value)
    return False


class SlockAdapter(BasePlatformAdapter):
    """Local HTTP wake endpoint for Raft bridge delivery."""

    def __init__(self, config: PlatformConfig):
        super().__init__(config, Platform.SLOCK)
        extra = config.extra or {}
        self._host: str = str(extra.get("host", DEFAULT_HOST))
        self._port: int = int(extra.get("port", DEFAULT_PORT))
        self._path: str = _path_value(extra.get("path", DEFAULT_PATH))
        self._bridge_token: str = str(extra.get("bridge_token", ""))
        self._runtime_session: str = str(
            extra.get("runtime_session", DEFAULT_RUNTIME_SESSION)
            or DEFAULT_RUNTIME_SESSION
        )
        self._max_body_bytes: int = int(
            extra.get("max_body_bytes", DEFAULT_MAX_BODY_BYTES)
        )
        self._runner = None

    @property
    def runtime_session(self) -> str:
        return self._runtime_session

    async def connect(self) -> bool:
        if not self._bridge_token:
            self._set_fatal_error(
                "missing_bridge_token",
                "RAFT_WAKE_ENDPOINT_TOKEN, SLOCK_WAKE_ENDPOINT_TOKEN, or platforms.slock.extra.bridge_token is required",
                retryable=False,
            )
            return False

        app = web.Application()
        app.router.add_get("/health", self._handle_health)
        app.router.add_post(self._path, self._handle_wake)

        if self._port != 0:
            import socket as _socket

            try:
                with _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM) as sock:
                    sock.settimeout(1)
                    sock.connect(("127.0.0.1", self._port))
                logger.error(
                    "[slock] Port %d already in use. Set SLOCK_WAKE_ENDPOINT_PORT or platforms.slock.extra.port",
                    self._port,
                )
                return False
            except (ConnectionRefusedError, OSError):
                pass

        self._runner = web.AppRunner(app)
        await self._runner.setup()
        site = web.TCPSite(self._runner, self._host, self._port)
        await site.start()
        self._mark_connected()
        logger.info("[slock] Wake endpoint listening on %s:%d%s", self._host, self._port, self._path)
        return True

    async def disconnect(self) -> None:
        if self._runner:
            await self._runner.cleanup()
            self._runner = None
        self._mark_disconnected()
        logger.info("[slock] Disconnected")

    async def send(
        self,
        chat_id: str,
        content: str,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        hint = (
            "Raft wake endpoint is wake-only; adapter send does not deliver to Slock. "
            "Use `raft message send --target \"<target>\"` with the exact target from "
            "`raft message check`, or the legacy `slock message send` alias."
        )
        logger.warning("[slock] %s Dropped adapter response for %s: %s", hint, chat_id, content[:200])
        return SendResult(success=False, error=hint, retryable=False)

    async def get_chat_info(self, chat_id: str) -> Dict[str, Any]:
        return {"name": f"slock/{chat_id}", "type": "slock"}

    async def _handle_health(self, request: "web.Request") -> "web.Response":
        return web.json_response(
            {
                "status": "ok",
                "platform": "slock",
                "runtimeSession": self._runtime_session,
            }
        )

    async def _handle_wake(self, request: "web.Request") -> "web.Response":
        if not self._validate_bridge_token(request.headers.get(BRIDGE_TOKEN_HEADER, "")):
            return web.json_response({"ok": False, "error": "unauthorized"}, status=401)

        content_length = request.content_length or 0
        if content_length > self._max_body_bytes:
            return web.json_response({"ok": False, "error": "payload_too_large"}, status=413)

        try:
            raw_body = await request.read()
        except Exception:
            return web.json_response({"ok": False, "error": "bad_request"}, status=400)

        payload: Dict[str, Any] = {}
        if raw_body.strip():
            try:
                parsed = json.loads(raw_body)
            except json.JSONDecodeError:
                return web.json_response({"ok": False, "error": "invalid_json"}, status=400)
            if not isinstance(parsed, dict):
                return web.json_response({"ok": False, "error": "invalid_payload"}, status=400)
            payload = parsed

        # Do not gate on payload["schema"]: the published bridge still sends the
        # historical Claude-channel schema id, and future ids follow Raft's
        # wake-endpoint compatibility contract rather than this adapter.
        if _has_content_field(payload):
            return web.json_response({"ok": False, "error": "content_not_allowed"}, status=400)

        accepted = await self._accept_wake(payload)
        if not accepted:
            return web.json_response(
                {
                    "ok": False,
                    "error": "not_ready",
                    "runtimeSession": self._runtime_session,
                },
                status=503,
            )

        return web.json_response(
            {
                "ok": True,
                "runtimeSession": self._runtime_session,
            },
            status=202,
        )

    def _validate_bridge_token(self, token: str) -> bool:
        if not self._bridge_token or not token:
            return False
        return hmac.compare_digest(token, self._bridge_token)

    async def _accept_wake(self, payload: Dict[str, Any]) -> bool:
        if not self._message_handler:
            logger.warning("[slock] Wake received before gateway message handler was attached")
            return False

        delivery_id = str(
            payload.get("eventId")
            or payload.get("attemptId")
            or payload.get("messageId")
            or payload.get("delivery_id")
            or payload.get("wake_id")
            or payload.get("id")
            or f"slock-wake-{int(time.time() * 1000)}"
        )
        source = self.build_source(
            chat_id=self._runtime_session,
            chat_name="Raft wake endpoint",
            chat_type="dm",
            user_id="slock-bridge",
            user_name="Raft Bridge",
        )
        event = MessageEvent(
            text=self._wake_prompt(),
            message_type=MessageType.TEXT,
            source=source,
            raw_message=payload,
            message_id=delivery_id,
            internal=True,
        )
        try:
            await self.handle_message(event)
        except Exception:
            logger.exception("[slock] Failed to inject wake event")
            return False
        return True

    async def handle_message(self, event: MessageEvent) -> None:
        """Accept Raft wake hints without interrupting an active Hermes turn."""
        if not self._message_handler:
            return

        session_key = build_session_key(
            event.source,
            group_sessions_per_user=self.config.extra.get("group_sessions_per_user", True),
            thread_sessions_per_user=self.config.extra.get("thread_sessions_per_user", False),
        )

        if session_key in self._active_sessions:
            logger.debug("[slock] Wake queued for busy session %s", session_key)
            merge_pending_message_event(self._pending_messages, session_key, event)
            return

        await super().handle_message(event)

    @staticmethod
    def _wake_prompt() -> str:
        return (
            "Raft wake hint received. New Raft messages may be pending. "
            "Run `raft message check` to inspect and handle them. "
            "When you need to reply, use the exact `target=` shown by `raft message check` "
            "with `raft message send --target \"<target>\"`; for thread targets, keep the "
            "same channel-or-DM suffix. If `raft` is not installed yet, use the legacy "
            "`slock message check` / `slock message send` aliases."
        )
