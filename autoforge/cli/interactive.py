"""Interactive session mode for AutoForge — session-based workflow.

Flow:
  autoforgeai → first-run setup (API key / GitHub / mode) → session loop
  Each session: describe project → build → show result → next session
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from rich.console import Console

console = Console()

# Operating modes with descriptions
MODES = [
    {
        "name": "Development — generate complete runnable projects",
        "value": "developer",
    },
    {
        "name": "Academic — scientific reasoning, theorem proving, theory evolution",
        "value": "academic",
    },
    {
        "name": "Verification — review & verify existing codebases",
        "value": "verification",
    },
]


def run_interactive() -> dict[str, Any]:
    """Run interactive session mode.

    Returns a dict with keys for the first session:
        action: "generate" | "review" | "import" | "setup"
        mode: "developer" | "academic" | "verification"
        description: str (for generate)
        project_path: str (for review/import)
        budget: float
        max_agents: int
    """
    from InquirerPy import inquirer

    # Step 1: Select operating mode
    mode = inquirer.select(
        message="Select mode:",
        choices=MODES,
        default="developer",
    ).execute()

    result: dict[str, Any] = {"mode": mode}

    # Step 2: Mode-specific flow
    if mode == "developer":
        result.update(_developer_session(inquirer))
    elif mode == "academic":
        result.update(_academic_session(inquirer))
    elif mode == "verification":
        result.update(_verification_session(inquirer))

    return result


def _current_workspace_path() -> str:
    """Use the current working directory as the active project path."""
    return str(Path.cwd().resolve())


def _skip_budget_prompt_for_subscription_mode() -> bool:
    """Return True when selected default models run on OpenAI subscription auth."""
    try:
        from autoforge.cli.setup_wizard import load_global_config
        from autoforge.engine.llm_router import detect_provider
    except Exception:
        return False

    config = load_global_config()
    auth_openai = config.get("auth_openai", {})
    auth_method = str(auth_openai.get("auth_method", "")).strip().lower()
    if auth_method not in ("codex_oauth", "device_code"):
        return False

    model_strong = str(config.get("model_strong", "")).strip()
    model_fast = str(config.get("model_fast", "")).strip()
    if not model_strong or not model_fast:
        return False
    return detect_provider(model_strong) == "openai" and detect_provider(model_fast) == "openai"


def _developer_session(inquirer: Any) -> dict[str, Any]:
    """Developer mode: generate or import a project."""
    action = inquirer.select(
        message="What would you like to do?",
        choices=[
            {"name": "Generate a new project", "value": "generate"},
            {"name": "Import & improve an existing project", "value": "import"},
            {"name": "Configure providers/models", "value": "setup"},
        ],
        default="generate",
    ).execute()

    if action == "setup":
        return {"action": "setup"}

    result: dict[str, Any] = {"action": action}

    if action == "generate":
        description = inquirer.text(
            message="Describe your project:",
            long_instruction="E.g., 'Build a Todo app with user login and task management'",
            validate=lambda x: len(x.strip()) >= 10,
            invalid_message="Please provide at least 10 characters",
        ).execute()
        result["description"] = description

    elif action == "import":
        project_path = _current_workspace_path()
        result["project_path"] = project_path
        console.print(f"[dim]Using current workspace: {project_path}[/dim]")

        enhance = inquirer.confirm(
            message="Add new features to this project?",
            default=False,
        ).execute()

        if enhance:
            enhance_desc = inquirer.text(
                message="Describe the enhancements:",
                long_instruction="E.g., 'Add dark mode toggle and user settings page'",
            ).execute()
            result["enhance_description"] = enhance_desc

    if _skip_budget_prompt_for_subscription_mode():
        console.print("[dim]OpenAI subscription mode detected - skipping USD budget prompt.[/dim]")
    else:
        budget = inquirer.number(
            message="Budget limit (USD):",
            default=10.0,
            float_allowed=True,
            min_allowed=0.5,
            max_allowed=100.0,
        ).execute()
        result["budget"] = float(budget)

    # Max agents
    max_agents = inquirer.number(
        message="Max parallel agents:",
        default=3,
        min_allowed=1,
        max_allowed=8,
    ).execute()
    result["max_agents"] = int(max_agents)

    return result


def _academic_session(inquirer: Any) -> dict[str, Any]:
    """Academic mode: scientific reasoning and theorem proving."""
    action = inquirer.select(
        message="What would you like to do?",
        choices=[
            {"name": "Generate a research project", "value": "generate"},
            {"name": "Analyze an existing codebase", "value": "review"},
            {"name": "Configure providers/models", "value": "setup"},
        ],
        default="generate",
    ).execute()

    if action == "setup":
        return {"action": "setup"}

    result: dict[str, Any] = {"action": action}

    if action == "generate":
        description = inquirer.text(
            message="Describe your research task:",
            long_instruction="E.g., 'Implement a formal proof of the Goldbach conjecture bounds'",
            validate=lambda x: len(x.strip()) >= 10,
            invalid_message="Please provide at least 10 characters",
        ).execute()
        result["description"] = description
    elif action == "review":
        project_path = _current_workspace_path()
        result["project_path"] = project_path
        console.print(f"[dim]Using current workspace: {project_path}[/dim]")

    if _skip_budget_prompt_for_subscription_mode():
        console.print("[dim]OpenAI subscription mode detected - skipping USD budget prompt.[/dim]")
    else:
        budget = inquirer.number(
            message="Budget limit (USD):",
            default=10.0,
            float_allowed=True,
            min_allowed=0.5,
            max_allowed=100.0,
        ).execute()
        result["budget"] = float(budget)
    result["max_agents"] = 3

    return result


def _verification_session(inquirer: Any) -> dict[str, Any]:
    """Verification mode: review and verify existing codebases."""
    action = inquirer.select(
        message="What would you like to do?",
        choices=[
            {"name": "Review current workspace", "value": "review"},
            {"name": "Configure providers/models", "value": "setup"},
        ],
        default="review",
    ).execute()
    if action == "setup":
        return {"action": "setup"}

    project_path = _current_workspace_path()
    console.print(f"[dim]Using current workspace: {project_path}[/dim]")

    result: dict[str, Any] = {
        "action": "review",
        "project_path": project_path,
    }

    if _skip_budget_prompt_for_subscription_mode():
        console.print("[dim]OpenAI subscription mode detected - skipping USD budget prompt.[/dim]")
    else:
        budget = inquirer.number(
            message="Budget limit (USD):",
            default=5.0,
            float_allowed=True,
            min_allowed=0.5,
            max_allowed=50.0,
        ).execute()
        result["budget"] = float(budget)
    result["max_agents"] = 3

    return result
