"""AutoForge CLI — interactive multi-mode interface.

The CLI supports a plugin-based command system. Built-in commands are
registered automatically. External commands can be added via
pyproject.toml entry points:

    [project.entry-points."autoforge.commands"]
    my_cmd = "my_plugin.commands:MyCommand"

Each command implements the CLICommand protocol.
"""

from __future__ import annotations

import argparse
from abc import ABC, abstractmethod
from typing import Any


class CLICommand(ABC):
    """Base class for CLI subcommands.

    Subclass this to add new commands to the AutoForge CLI.
    Register via entry points or CommandRegistry.register().
    """

    name: str = ""
    help: str = ""

    @abstractmethod
    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        """Add command-specific arguments to the parser."""

    @abstractmethod
    async def execute(self, args: argparse.Namespace, config: Any) -> int:
        """Execute the command. Return exit code (0 = success)."""


class CommandRegistry:
    """Registry of CLI subcommands."""

    def __init__(self) -> None:
        self._commands: dict[str, CLICommand] = {}

    def register(self, command: CLICommand) -> None:
        self._commands[command.name] = command

    def get(self, name: str) -> CLICommand | None:
        return self._commands.get(name)

    def all(self) -> dict[str, CLICommand]:
        return dict(self._commands)

    def add_subparsers(self, subparsers: Any) -> None:
        """Add all registered commands to an argparse subparsers object."""
        for cmd in self._commands.values():
            parser = subparsers.add_parser(cmd.name, help=cmd.help)
            cmd.add_arguments(parser)

    def discover_plugins(self) -> None:
        """Discover and load CLI command plugins from entry points."""
        try:
            from importlib.metadata import entry_points

            eps = entry_points()
            # Python 3.12+ returns SelectableGroups
            if hasattr(eps, "select"):
                commands = eps.select(group="autoforge.commands")
            else:
                commands = eps.get("autoforge.commands", [])

            for ep in commands:
                try:
                    cmd_class = ep.load()
                    cmd = cmd_class()
                    if isinstance(cmd, CLICommand) and cmd.name:
                        self.register(cmd)
                except Exception:
                    pass
        except ImportError:
            pass


__all__ = ["CLICommand", "CommandRegistry"]
