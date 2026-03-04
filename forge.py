#!/usr/bin/env python3
"""AutoForge — AI-powered multi-agent development platform.

Preferred usage (install first: pip install forgeai):
    forgeai                                      # Interactive session
    forgeai generate "Build a Todo app"          # Generate new project
    forgeai review ./my-project                  # Review existing project
    forgeai import ./my-project                  # Import & improve
    forgeai setup                                # Configure settings

Legacy usage (still supported):
    python forge.py "Build a Todo app"
    python forge.py --resume
    python forge.py --status

Daemon mode:
    forgeai daemon start                         # Start 24/7 daemon
    forgeai daemon stop                          # Stop daemon
    forgeai daemon status                        # Check daemon status

Queue management:
    forgeai queue "project description"          # Add to queue
    forgeai projects                             # List all projects
    forgeai deploy <project_id>                  # Show deploy guide
"""

from autoforge.cli.app import main

if __name__ == "__main__":
    main()
