"""Webhook API channel for daemon intake and project management."""

from __future__ import annotations

from dataclasses import dataclass
import hmac
import logging
from pathlib import Path
from typing import Any

from autoforge.engine.config import ForgeConfig
from autoforge.engine.project_registry import ProjectRegistry, ProjectStatus
from autoforge.engine.request_intake import IntakePolicyError, RequestIntakeService

logger = logging.getLogger(__name__)


@dataclass
class RequestContext:
    requester: str
    is_admin: bool = False


def _is_limit_error(msg: str) -> bool:
    lowered = msg.lower()
    return any(
        token in lowered
        for token in ("limit", "quota", "rate", "queue is full", "too many")
    )


async def start_webhook_server(
    config: ForgeConfig,
    registry: ProjectRegistry,
) -> None:
    """Start the FastAPI webhook server."""
    try:
        from fastapi import Depends, FastAPI, HTTPException, Request
        import uvicorn
    except ImportError:
        raise ImportError(
            "Webhook server requires 'fastapi' and 'uvicorn'. "
            "Install them with: pip install autoforge[channels]"
        ) from None

    if config.webhook_require_auth and not config.webhook_secret:
        local_host = config.webhook_host in {"127.0.0.1", "localhost"}
        if not (config.webhook_allow_unauthenticated_local and local_host):
            raise RuntimeError(
                "Webhook auth is required but FORGE_WEBHOOK_SECRET is empty. "
                "Set FORGE_WEBHOOK_SECRET or explicitly allow local unauthenticated mode."
            )

    intake = RequestIntakeService(config, registry)
    app = FastAPI(title="AutoForge API", version="0.2.0")

    async def verify_auth(request: Request) -> None:
        if not config.webhook_require_auth:
            return

        if not config.webhook_secret:
            local_host = config.webhook_host in {"127.0.0.1", "localhost"}
            if config.webhook_allow_unauthenticated_local and local_host:
                return
            raise HTTPException(status_code=503, detail="Webhook authentication misconfigured")

        auth = request.headers.get("Authorization", "").encode()
        expected = f"Bearer {config.webhook_secret}".encode()
        if not hmac.compare_digest(auth, expected):
            raise HTTPException(status_code=401, detail="Unauthorized")

    async def get_request_context(request: Request) -> RequestContext:
        requester_hint = ""
        if config.webhook_trust_requester_header:
            requester_hint = request.headers.get(config.webhook_requester_header, "")
        fallback = request.client.host if request.client else "unknown"
        requester = intake.normalize_requester(
            channel="webhook",
            requester_hint=requester_hint,
            fallback_hint=fallback,
        )

        is_admin = False
        if config.webhook_admin_secret:
            supplied = request.headers.get("X-Autoforge-Admin-Secret", "").encode()
            expected = config.webhook_admin_secret.encode()
            is_admin = hmac.compare_digest(supplied, expected)

        return RequestContext(requester=requester, is_admin=is_admin)

    @app.get("/api/health")
    async def health() -> dict[str, Any]:
        queue_size = await registry.queue_size()
        building = len(await registry.list_by_status(ProjectStatus.BUILDING))
        return {
            "status": "ok",
            "queue_size": queue_size,
            "building": building,
        }

    @app.post("/api/build", dependencies=[Depends(verify_auth)])
    async def build_project(
        request: Request,
        ctx: RequestContext = Depends(get_request_context),
    ) -> dict[str, Any]:
        try:
            body = await request.json()
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid JSON body")

        idempotency_key = request.headers.get(config.webhook_idempotency_header, "")
        description = body.get("description", "")
        budget = body.get("budget", None)

        try:
            result = await intake.enqueue(
                channel="webhook",
                requester_hint=ctx.requester.split(":", 1)[1] if ":" in ctx.requester else ctx.requester,
                fallback_hint=request.client.host if request.client else "unknown",
                description=description,
                budget=budget,
                idempotency_key=idempotency_key,
            )
        except IntakePolicyError as exc:
            status = 429 if _is_limit_error(str(exc)) else 400
            raise HTTPException(status_code=status, detail=str(exc))

        return {
            "id": result.project.id,
            "status": result.project.status.value,
            "queue_position": result.queue_size,
            "budget_usd": result.project.budget_usd,
            "requester": result.project.requested_by,
            "deduplicated": result.deduplicated,
        }

    @app.get("/api/projects", dependencies=[Depends(verify_auth)])
    async def list_projects(
        limit: int = 50,
        all: bool = False,
        ctx: RequestContext = Depends(get_request_context),
    ) -> list[dict[str, Any]]:
        limit = min(max(limit, 1), 500)
        if all and not ctx.is_admin:
            raise HTTPException(status_code=403, detail="Admin scope required")

        if all:
            projects = await registry.list_all(limit=limit)
        else:
            projects = await registry.list_for_requester(ctx.requester, limit=limit)
        return [p.to_dict() for p in projects]

    @app.get("/api/projects/{project_id}", dependencies=[Depends(verify_auth)])
    async def get_project(
        project_id: str,
        ctx: RequestContext = Depends(get_request_context),
    ) -> dict[str, Any]:
        try:
            if ctx.is_admin:
                project = await registry.get(project_id)
            else:
                project = await registry.get_for_requester(project_id, ctx.requester)
        except KeyError:
            raise HTTPException(status_code=404, detail="Project not found")
        return project.to_dict()

    @app.get("/api/projects/{project_id}/deploy", dependencies=[Depends(verify_auth)])
    async def get_deploy_guide(
        project_id: str,
        ctx: RequestContext = Depends(get_request_context),
    ) -> dict[str, str]:
        try:
            if ctx.is_admin:
                project = await registry.get(project_id)
            else:
                project = await registry.get_for_requester(project_id, ctx.requester)
        except KeyError:
            raise HTTPException(status_code=404, detail="Project not found")

        if project.status != ProjectStatus.COMPLETED:
            raise HTTPException(
                status_code=400,
                detail=f"Project not completed (status: {project.status.value})",
            )

        workspace = Path(project.workspace_path).resolve()
        guide_path = (workspace / "DEPLOY_GUIDE.md").resolve()
        if not str(guide_path).startswith(str(workspace)):
            raise HTTPException(status_code=403, detail="Access denied")
        if not guide_path.exists():
            raise HTTPException(status_code=404, detail="Deploy guide not found")

        try:
            return {"guide": guide_path.read_text(encoding="utf-8")}
        except (OSError, UnicodeDecodeError) as exc:
            raise HTTPException(status_code=500, detail=f"Failed to read deploy guide: {exc}")

    @app.delete("/api/projects/{project_id}", dependencies=[Depends(verify_auth)])
    async def cancel_project(
        project_id: str,
        ctx: RequestContext = Depends(get_request_context),
    ) -> dict[str, Any]:
        if ctx.is_admin:
            ok = await registry.cancel(project_id)
        else:
            ok = await registry.cancel_for_requester(project_id, ctx.requester)
        if not ok:
            raise HTTPException(
                status_code=400,
                detail="Only your queued projects can be cancelled",
            )
        return {"id": project_id, "status": "cancelled"}

    server_config = uvicorn.Config(
        app,
        host=config.webhook_host,
        port=config.webhook_port,
        log_level="warning",
    )
    server = uvicorn.Server(server_config)
    await server.serve()
