"""Tests for the Raft channel adapter."""

from unittest.mock import AsyncMock

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import GatewayConfig, Platform, PlatformConfig, _apply_env_overrides
from gateway.platforms.raft import (
    BRIDGE_TOKEN_HEADER,
    DEFAULT_PATH,
    RaftAdapter,
    _has_content_field,
)
from gateway.session import build_session_key

RAFT_CHANNEL_SCHEMA = "raft-channel-wake.v1"
FUTURE_RAFT_CHANNEL_SCHEMA = "raft-channel-wake.v2"


def _make_config(**extra):
    data = {
        "bridge_token": "bridge-secret",
        "runtime_session": "default",
        "port": 0,
    }
    data.update(extra)
    return PlatformConfig(enabled=True, extra=data)


def _make_adapter(**extra):
    return RaftAdapter(_make_config(**extra))


def _create_app(adapter: RaftAdapter) -> web.Application:
    app = web.Application()
    app.router.add_get("/health", adapter._handle_health)
    app.router.add_post(adapter._path, adapter._handle_wake)
    return app


class TestRaftWakePayload:
    def test_detects_content_fields(self):
        assert _has_content_field({"text": "hello"}) is True
        assert _has_content_field({"nested": {"messages": []}}) is True
        assert _has_content_field({"eventId": "evt-1", "messageId": "msg-1"}) is False


class TestRaftWakeHttp:
    @pytest.mark.asyncio
    async def test_send_reports_wake_only_hint(self):
        adapter = _make_adapter()

        result = await adapter.send("default", "hello")

        assert result.success is False
        assert result.retryable is False
        assert "wake-only" in result.error
        assert "raft message send" in result.error

    @pytest.mark.asyncio
    async def test_rejects_missing_bridge_token(self):
        adapter = _make_adapter()
        adapter.handle_message = AsyncMock()

        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as client:
            resp = await client.post(DEFAULT_PATH, json={"eventId": "wake-1"})
            assert resp.status == 401
            body = await resp.json()

        assert body["ok"] is False
        adapter.handle_message.assert_not_called()

    @pytest.mark.asyncio
    async def test_rejects_content_bearing_payload(self):
        adapter = _make_adapter()
        adapter.set_message_handler(AsyncMock())
        adapter.handle_message = AsyncMock()

        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as client:
            resp = await client.post(
                DEFAULT_PATH,
                json={"eventId": "wake-1", "text": "do work"},
                headers={BRIDGE_TOKEN_HEADER: "bridge-secret"},
            )
            assert resp.status == 400
            body = await resp.json()

        assert body == {"ok": False, "error": "content_not_allowed"}
        adapter.handle_message.assert_not_called()

    @pytest.mark.asyncio
    async def test_returns_not_ready_without_gateway_handler(self):
        adapter = _make_adapter()

        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as client:
            resp = await client.post(
                DEFAULT_PATH,
                json={"eventId": "wake-1"},
                headers={BRIDGE_TOKEN_HEADER: "bridge-secret"},
            )
            assert resp.status == 503
            body = await resp.json()

        assert body["ok"] is False
        assert body["runtimeSession"] == "default"

    @pytest.mark.asyncio
    @pytest.mark.parametrize("schema", [RAFT_CHANNEL_SCHEMA, FUTURE_RAFT_CHANNEL_SCHEMA])
    async def test_accepts_content_free_wake_as_internal_event(self, schema):
        adapter = _make_adapter()
        adapter.set_message_handler(AsyncMock())
        adapter.handle_message = AsyncMock()

        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as client:
            resp = await client.post(
                DEFAULT_PATH,
                json={
                    "schema": schema,
                    "attemptId": "attempt-1",
                    "eventId": "wake-1",
                    "messageId": "msg-1",
                    "agentId": "agent-1",
                    "profile": "dev",
                    "coreSessionId": "default",
                    "adapterInstance": "hermes",
                    "occurredAt": "2026-06-11T08:00:00Z",
                },
                headers={BRIDGE_TOKEN_HEADER: "bridge-secret"},
            )
            assert resp.status == 202
            body = await resp.json()

        assert body == {"ok": True, "runtimeSession": "default"}

        adapter.handle_message.assert_awaited_once()
        event = adapter.handle_message.await_args.args[0]
        assert event.internal is True
        assert event.message_id == "wake-1"
        assert event.raw_message["schema"] == schema
        assert event.raw_message["eventId"] == "wake-1"
        assert event.raw_message["attemptId"] == "attempt-1"
        assert event.raw_message["messageId"] == "msg-1"
        assert event.source.platform == Platform.RAFT
        assert event.source.chat_id == "default"
        assert "raft message check" in event.text
        assert "raft message send" in event.text
        assert 'target=' in event.text

    @pytest.mark.asyncio
    async def test_busy_session_queues_without_interrupt(self):
        handler = AsyncMock()
        adapter = _make_adapter()
        adapter.set_message_handler(handler)

        source = adapter.build_source(
            chat_id="default",
            chat_name="Raft channel",
            chat_type="dm",
            user_id="raft-bridge",
            user_name="Raft Bridge",
        )
        session_key = build_session_key(source)
        adapter._active_sessions[session_key] = __import__("asyncio").Event()

        accepted = await adapter._accept_wake({"eventId": "wake-busy"})

        assert accepted is True
        handler.assert_not_called()
        assert session_key in adapter._pending_messages
        pending = adapter._pending_messages[session_key]
        assert pending.message_id == "wake-busy"
        assert "raft message check" in pending.text
        assert "raft message send" in pending.text


class TestRaftConfig:
    def test_env_overrides_enable_raft_platform(self, monkeypatch):
        monkeypatch.setenv("RAFT_CHANNEL_TOKEN", "bridge-secret")
        monkeypatch.setenv("RAFT_CHANNEL_PORT", "8765")
        monkeypatch.setenv("RAFT_CHANNEL_RUNTIME_SESSION", "main")

        config = GatewayConfig()
        _apply_env_overrides(config)

        assert Platform.RAFT in config.platforms
        raft_config = config.platforms[Platform.RAFT]
        assert raft_config.enabled is True
        assert raft_config.extra["bridge_token"] == "bridge-secret"
        assert raft_config.extra["port"] == 8765
        assert raft_config.extra["runtime_session"] == "main"
        assert config.get_connected_platforms() == [Platform.RAFT]

    def test_platform_metadata_and_toolset_are_registered(self):
        from agent.prompt_builder import PLATFORM_HINTS
        from hermes_cli.tools_config import PLATFORMS
        from toolsets import TOOLSETS, validate_toolset

        assert PLATFORMS["raft"]["label"] == "🔔 Raft"
        assert PLATFORMS["raft"]["default_toolset"] == "hermes-raft"
        assert "hermes-raft" in TOOLSETS
        assert "hermes-raft" in TOOLSETS["hermes-gateway"]["includes"]
        assert validate_toolset("hermes-raft")
        assert "raft message send" in TOOLSETS["hermes-raft"]["description"]
        assert "raft message check" in PLATFORM_HINTS["raft"]
        assert "raft message send" in PLATFORM_HINTS["raft"]
