#!/usr/bin/env python3
"""AutoForge — AI-powered multi-agent development platform.

Preferred usage (install first: pip install autoforge):
    autoforge                                    # Interactive mode
    autoforge generate "Build a Todo app"        # Generate new project
    autoforge review ./my-project                # Review existing project
    autoforge import ./my-project                # Import & improve
    autoforge setup                              # Configure settings

Legacy usage (still supported):
    python forge.py "Build a Todo app"
    python forge.py --resume
    python forge.py --status
"""

from autoforge.cli.app import main

if __name__ == "__main__":
    main()
