"""Asynchronous bridge primitives for channel-to-channel messaging.

This models the workflow discussed for Telegram-like remote channels:

1. A sends a message to B and keeps running.
2. If B replies before timeout, trigger the normal callback.
3. If timeout is reached first, trigger timeout callback.
4. If B replies after timeout, still deliver it via a late-response callback.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Awaitable, Callable, Generic, TypeVar
from uuid import uuid4

RequestPayloadT = TypeVar("RequestPayloadT")
ResponsePayloadT = TypeVar("ResponsePayloadT")


@dataclass(frozen=True)
class BridgeRequest(Generic[RequestPayloadT]):
    """Outbound request envelope from local agent A to remote endpoint B."""

    request_id: str
    payload: RequestPayloadT
    created_at: datetime


@dataclass(frozen=True)
class BridgeResponse(Generic[ResponsePayloadT]):
    """Inbound response envelope returning from endpoint B."""

    request_id: str
    payload: ResponsePayloadT
    received_at: datetime


@dataclass(frozen=True)
class BridgeTimeoutEvent:
    """Timeout event emitted when a pending request exceeds its SLA."""

    request_id: str
    timed_out_at: datetime


@dataclass
class PendingBridgeRequest(Generic[ResponsePayloadT]):
    """Handle returned to caller so it can await later (or not await at all)."""

    request_id: str
    future: asyncio.Future[BridgeResponse[ResponsePayloadT]]

    async def wait(self) -> BridgeResponse[ResponsePayloadT]:
        """Wait for a response; may raise ``asyncio.TimeoutError``."""
        return await self.future

    def done(self) -> bool:
        """Whether the request has completed (response or timeout)."""
        return self.future.done()


class AsyncBridgeAgent(Generic[RequestPayloadT, ResponsePayloadT]):
    """Correlation + timeout router for fully async message flows."""

    def __init__(
        self,
        *,
        timeout_seconds: float,
        on_timeout: Callable[[BridgeTimeoutEvent], Awaitable[None]] | None = None,
        on_response: Callable[[BridgeResponse[ResponsePayloadT]], Awaitable[None]] | None = None,
        on_late_response: Callable[[BridgeResponse[ResponsePayloadT]], Awaitable[None]] | None = None,
        max_timed_out_history: int = 1024,
    ) -> None:
        self.timeout_seconds = timeout_seconds
        self.max_timed_out_history = max_timed_out_history
        self._on_timeout = on_timeout
        self._on_response = on_response
        self._on_late_response = on_late_response
        self._pending: dict[str, asyncio.Future[BridgeResponse[ResponsePayloadT]]] = {}
        self._timed_out_ids: dict[str, datetime] = {}
        self._timeout_tasks: dict[str, asyncio.Task[None]] = {}
        self._outbound_tasks: dict[str, asyncio.Task[None]] = {}

    async def send(
        self,
        payload: RequestPayloadT,
        *,
        sender: Callable[[BridgeRequest[RequestPayloadT]], Awaitable[None]],
        request_id: str | None = None,
    ) -> PendingBridgeRequest[ResponsePayloadT]:
        """Send request and return a handle immediately.

        Caller may continue doing other work and await ``handle.wait()`` later.
        """
        rid = request_id or uuid4().hex
        now = datetime.now(timezone.utc)
        fut: asyncio.Future[BridgeResponse[ResponsePayloadT]] = asyncio.get_running_loop().create_future()
        self._pending[rid] = fut

        timeout_task = asyncio.create_task(self._timeout_after(rid))
        self._timeout_tasks[rid] = timeout_task

        outbound_request = BridgeRequest(request_id=rid, payload=payload, created_at=now)
        outbound_task = asyncio.create_task(sender(outbound_request))
        self._outbound_tasks[rid] = outbound_task

        def _complete_outbound(task: asyncio.Task[None], req_id: str = rid) -> None:
            self._outbound_tasks.pop(req_id, None)
            timeout_task = self._timeout_tasks.pop(req_id, None)
            fut = self._pending.pop(req_id, None)

            try:
                task.result()
            except asyncio.CancelledError:
                return
            except Exception as exc:
                if timeout_task is not None:
                    timeout_task.cancel()
                if fut is not None and not fut.done():
                    fut.set_exception(exc)
                return

            if fut is not None:
                self._pending[req_id] = fut
            if timeout_task is not None:
                self._timeout_tasks[req_id] = timeout_task

        outbound_task.add_done_callback(_complete_outbound)
        return PendingBridgeRequest(request_id=rid, future=fut)

    async def receive(self, request_id: str, payload: ResponsePayloadT) -> None:
        """Deliver inbound response from remote endpoint B to local consumers."""
        response = BridgeResponse(
            request_id=request_id,
            payload=payload,
            received_at=datetime.now(timezone.utc),
        )
        if request_id in self._timed_out_ids:
            if self._on_late_response is not None:
                await self._on_late_response(response)
            self._timed_out_ids.pop(request_id, None)
            return

        fut = self._pending.pop(request_id, None)
        timeout_task = self._timeout_tasks.pop(request_id, None)
        self._outbound_tasks.pop(request_id, None)

        if timeout_task is not None:
            timeout_task.cancel()

        if fut is None:
            if self._on_late_response is not None:
                await self._on_late_response(response)
            return

        if not fut.done():
            fut.set_result(response)
        if self._on_response is not None:
            await self._on_response(response)

    async def close(self) -> None:
        """Cancel outstanding work and fail pending futures for shutdown."""
        for task in self._timeout_tasks.values():
            task.cancel()
        for task in self._outbound_tasks.values():
            task.cancel()
        self._timeout_tasks.clear()
        self._outbound_tasks.clear()

        for fut in self._pending.values():
            if not fut.done():
                fut.set_exception(asyncio.CancelledError("bridge closed"))
        self._pending.clear()
        self._timed_out_ids.clear()

    async def _timeout_after(self, request_id: str) -> None:
        try:
            await asyncio.sleep(self.timeout_seconds)
            fut = self._pending.pop(request_id, None)
            self._timeout_tasks.pop(request_id, None)
            if fut is None or fut.done():
                return

            fut.set_exception(asyncio.TimeoutError(f"request timed out: {request_id}"))
            self._timed_out_ids[request_id] = datetime.now(timezone.utc)
            while len(self._timed_out_ids) > self.max_timed_out_history:
                self._timed_out_ids.pop(next(iter(self._timed_out_ids)))

            if self._on_timeout is not None:
                await self._on_timeout(
                    BridgeTimeoutEvent(
                        request_id=request_id,
                        timed_out_at=datetime.now(timezone.utc),
                    )
                )
        except asyncio.CancelledError:
            return
