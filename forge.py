#!/usr/bin/env python3
"""AutoForge — AI-powered multi-agent development platform.

Usage:
    python forge.py                                    # Interactive mode
    python forge.py generate "Build a Todo app"        # Generate new project
    python forge.py review ./my-project                # Review existing project
    python forge.py import ./my-project                # Import & improve
    python forge.py setup                              # Configure settings
    python forge.py status                             # Show projects
    python forge.py resume                             # Resume interrupted run

Legacy usage (still supported):
    python forge.py "Build a Todo app with user login"
    python forge.py --resume
    python forge.py --status
"""

from cli.app import main

if __name__ == "__main__":
    main()
