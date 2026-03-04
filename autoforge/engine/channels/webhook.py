"""Webhook API channel for AutoForge daemon.

FastAPI server exposing a REST API for project management:
  POST   /api/build              — Queue a new project
  GET    /api/projects           — List all projects
  GET    /api/projects/{id}      — Get project details
  GET    /api/projects/{id}/deploy — Get deployment guide
  DELETE /api/projects/{id}      — Cancel a queued project
  GET    /api/health             — Health check
"""

from __future__ import annotations

import hmac
import logging
from pathlib import Path
from typing import Any

from autoforge.engine.config import ForgeConfig
from autoforge.engine.project_registry import ProjectRegistry, ProjectStatus

logger = logging.getLogger(__name__)


async def start_webhook_server(
    config: ForgeConfig,
    registry: ProjectRegistry,
) -> None:
    """Start the FastAPI webhook server."""
    from fastapi import FastAPI, HTTPException, Request, Depends
    from fastapi.responses import JSONResponse
    import uvicorn

    app = FastAPI(title="AutoForge API", version="0.1.0")

    # Auth dependency (constant-time comparison to prevent timing attacks)
    async def verify_auth(request: Request) -> None:
        if not config.webhook_secret:
            return  # No auth configured
        auth = request.headers.get("Authorization", "").encode()
        expected = f"Bearer {config.webhook_secret}".encode()
        if not hmac.compare_digest(auth, expected):
            raise HTTPException(status_code=401, detail="Unauthorized")

    @app.get("/api/health")
    async def health() -> dict[str, Any]:
        queue_size = await registry.queue_size()
        return {
            "status": "ok",
            "queue_size": queue_size,
        }

    @app.post("/api/build", dependencies=[Depends(verify_auth)])
    async def build_project(request: Request) -> dict[str, Any]:
        try:
            body = await request.json()
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid JSON body")

        description = body.get("description", "").strip()
        if not description:
            raise HTTPException(status_code=400, detail="description is required")
        if len(description) > 10000:
            raise HTTPException(status_code=400, detail="description too long (max 10,000 chars)")

        budget = body.get("budget", config.budget_limit_usd)
        try:
            budget = float(budget)
        except (TypeError, ValueError):
            raise HTTPException(status_code=400, detail="budget must be a number")
        if budget <= 0:
            raise HTTPException(status_code=400, detail="budget must be positive")
        if budget > 1000:
            raise HTTPException(status_code=400, detail="budget too high (max $1000)")

        requester = f"webhook:{request.client.host}" if request.client else "webhook:unknown"

        project = await registry.enqueue(
            description=description,
            requested_by=requester,
            budget_usd=budget,
        )
        queue_size = await registry.queue_size()

        return {
            "id": project.id,
            "status": project.status.value,
            "queue_position": queue_size,
            "budget_usd": project.budget_usd,
        }

    @app.get("/api/projects", dependencies=[Depends(verify_auth)])
    async def list_projects(limit: int = 50) -> list[dict[str, Any]]:
        limit = min(max(limit, 1), 500)
        projects = await registry.list_all(limit=limit)
        return [p.to_dict() for p in projects]

    @app.get("/api/projects/{project_id}", dependencies=[Depends(verify_auth)])
    async def get_project(project_id: str) -> dict[str, Any]:
        try:
            project = await registry.get(project_id)
        except KeyError:
            raise HTTPException(status_code=404, detail="Project not found")
        return project.to_dict()

    @app.get("/api/projects/{project_id}/deploy", dependencies=[Depends(verify_auth)])
    async def get_deploy_guide(project_id: str) -> dict[str, str]:
        try:
            project = await registry.get(project_id)
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
        except (OSError, UnicodeDecodeError) as e:
            raise HTTPException(status_code=500, detail=f"Failed to read deploy guide: {e}")

    @app.delete("/api/projects/{project_id}", dependencies=[Depends(verify_auth)])
    async def cancel_project(project_id: str) -> dict[str, Any]:
        ok = await registry.cancel(project_id)
        if not ok:
            raise HTTPException(
                status_code=400,
                detail="Only queued projects can be cancelled",
            )
        return {"id": project_id, "status": "cancelled"}

    # Run the server
    server_config = uvicorn.Config(
        app,
        host=config.webhook_host,
        port=config.webhook_port,
        log_level="warning",
    )
    server = uvicorn.Server(server_config)
    await server.serve()
