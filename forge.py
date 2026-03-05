#!/usr/bin/env python3
"""AutoForge — AI-powered multi-agent development platform.

Preferred usage (install first: pip install autoforgeai):
    autoforgeai                                  # Interactive session
    autoforgeai generate "Build a Todo app"      # Generate new project
    autoforgeai review ./my-project              # Review existing project
    autoforgeai import ./my-project              # Import & improve
    autoforgeai setup                            # Configure settings

Legacy usage (still supported):
    python forge.py "Build a Todo app"
    python forge.py --resume
    python forge.py --status

Daemon mode:
    autoforgeai daemon start                     # Start 24/7 daemon
    autoforgeai daemon stop                      # Stop daemon
    autoforgeai daemon status                    # Check daemon status

Queue management:
    autoforgeai queue "project description"      # Add to queue
    autoforgeai projects                         # List all projects
    autoforgeai deploy <project_id>              # Show deploy guide

Paper reproduction:
    autoforgeai paper infer "research goal"      # Infer likely ICLR papers
    autoforgeai paper benchmark                  # Benchmark inference quality
    autoforgeai paper reproduce "goal"           # Build reproduction brief/prompt
    autoforgeai paper reproduce "goal" --strict-contract  # Enforce artifact/schema contract
"""

from autoforge.cli.app import main

if __name__ == "__main__":
    main()
