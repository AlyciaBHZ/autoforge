"""AutoForge Channels — input interfaces and async bridge primitives."""

from .bridge_agent import (
    AsyncBridgeAgent,
    BridgeRequest,
    BridgeResponse,
    BridgeTimeoutEvent,
    PendingBridgeRequest,
)

__all__ = [
    "AsyncBridgeAgent",
    "BridgeRequest",
    "BridgeResponse",
    "BridgeTimeoutEvent",
    "PendingBridgeRequest",
]
