"""Interactive mode for AutoForge — InquirerPy-based menus."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from rich.console import Console

console = Console()


def run_interactive() -> dict[str, Any]:
    """Run interactive mode and return user choices as a config dict.

    Returns a dict with keys:
        action: "generate" | "review" | "import" | "setup"
        mode: "developer" | "research"
        description: str (for generate)
        project_path: str (for review/import)
        enhance_description: str (for import, optional)
        budget: float
        max_agents: int
        mobile_target: "none" | "ios" | "android" | "both"
    """
    from InquirerPy import inquirer
    from InquirerPy.separator import Separator

    # Step 1: What to do
    action = inquirer.select(
        message="What would you like to do?",
        choices=[
            {"name": "Generate a new project", "value": "generate"},
            {"name": "Review an existing project", "value": "review"},
            {"name": "Import & improve a project", "value": "import"},
            Separator(),
            {"name": "Configure settings", "value": "setup"},
        ],
        default="generate",
    ).execute()

    if action == "setup":
        return {"action": "setup"}

    # Step 2: Operating mode
    mode = inquirer.select(
        message="Operating mode:",
        choices=[
            {"name": "Developer (will modify code)", "value": "developer"},
            {"name": "Research (analysis only, no changes)", "value": "research"},
        ],
        default="developer",
    ).execute()

    result: dict[str, Any] = {"action": action, "mode": mode}

    # Step 3: Action-specific input
    if action == "generate":
        description = inquirer.text(
            message="Describe your project:",
            long_instruction="E.g., 'Build a Todo app with user login and task management'",
            validate=lambda x: len(x.strip()) >= 10,
            invalid_message="Please provide at least 10 characters",
        ).execute()
        result["description"] = description

    elif action == "review":
        project_path = inquirer.filepath(
            message="Path to project:",
            validate=lambda x: Path(x).is_dir(),
            invalid_message="Must be a valid directory",
            only_directories=True,
        ).execute()
        result["project_path"] = project_path

    elif action == "import":
        project_path = inquirer.filepath(
            message="Path to project to import:",
            validate=lambda x: Path(x).is_dir(),
            invalid_message="Must be a valid directory",
            only_directories=True,
        ).execute()
        result["project_path"] = project_path

        enhance = inquirer.confirm(
            message="Do you want to add new features to this project?",
            default=False,
        ).execute()

        if enhance:
            enhance_desc = inquirer.text(
                message="Describe the enhancements:",
                long_instruction="E.g., 'Add dark mode toggle and user settings page'",
            ).execute()
            result["enhance_description"] = enhance_desc

    # Step 4: Budget
    budget = inquirer.number(
        message="Budget limit (USD):",
        default=10.0,
        float_allowed=True,
        min_allowed=0.5,
        max_allowed=100.0,
    ).execute()
    result["budget"] = float(budget)

    # Step 5: Max agents
    max_agents = inquirer.number(
        message="Max parallel agents:",
        default=3,
        min_allowed=1,
        max_allowed=8,
    ).execute()
    result["max_agents"] = int(max_agents)

    # Step 6: Mobile target (for generate/import only)
    if action in ("generate", "import"):
        mobile = inquirer.select(
            message="Generate mobile app?",
            choices=[
                {"name": "No", "value": "none"},
                {"name": "iOS only", "value": "ios"},
                {"name": "Android only", "value": "android"},
                {"name": "Both iOS & Android", "value": "both"},
            ],
            default="none",
        ).execute()
        result["mobile_target"] = mobile
    else:
        result["mobile_target"] = "none"

    return result
