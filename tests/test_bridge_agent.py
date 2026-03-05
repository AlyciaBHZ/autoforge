from __future__ import annotations

import asyncio

import pytest

from autoforge.engine.channels.bridge_agent import AsyncBridgeAgent, BridgeRequest


def test_bridge_normal_response() -> None:
    async def _run() -> None:
        responses: list[str] = []

        async def sender(request: BridgeRequest[str]) -> None:
            await asyncio.sleep(0.01)
            await bridge.receive(request.request_id, "ok")

        async def on_response(response) -> None:
            responses.append(response.payload)

        bridge: AsyncBridgeAgent[str, str] = AsyncBridgeAgent(
            timeout_seconds=0.2,
            on_response=on_response,
        )

        pending = await bridge.send("ping", sender=sender)
        result = await pending.wait()

        assert result.payload == "ok"
        assert responses == ["ok"]

    asyncio.run(_run())


def test_bridge_timeout_then_late_response() -> None:
    async def _run() -> None:
        timeouts: list[str] = []
        late: list[str] = []

        async def sender(request: BridgeRequest[str]) -> None:
            async def _late_reply() -> None:
                await asyncio.sleep(0.06)
                await bridge.receive(request.request_id, "late-ok")

            asyncio.create_task(_late_reply())

        async def on_timeout(event) -> None:
            timeouts.append(event.request_id)

        async def on_late_response(response) -> None:
            late.append(response.payload)

        bridge: AsyncBridgeAgent[str, str] = AsyncBridgeAgent(
            timeout_seconds=0.02,
            on_timeout=on_timeout,
            on_late_response=on_late_response,
        )

        pending = await bridge.send("ping", sender=sender)

        with pytest.raises(asyncio.TimeoutError):
            await pending.wait()

        await asyncio.sleep(0.08)

        assert timeouts == [pending.request_id]
        assert late == ["late-ok"]

    asyncio.run(_run())


def test_bridge_handle_can_be_awaited_later() -> None:
    async def _run() -> None:
        async def sender(request: BridgeRequest[str]) -> None:
            await asyncio.sleep(0.03)
            await bridge.receive(request.request_id, "done")

        bridge: AsyncBridgeAgent[str, str] = AsyncBridgeAgent(timeout_seconds=0.2)

        pending = await bridge.send("job", sender=sender)
        assert not pending.done()

        await asyncio.sleep(0.01)
        result = await pending.wait()
        assert result.payload == "done"

    asyncio.run(_run())
