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

Daemon mode:
    python forge.py daemon start       # Start 24/7 daemon
    python forge.py daemon stop        # Stop daemon
    python forge.py daemon status      # Check daemon status
    python forge.py daemon install     # Install as system service

Queue management:
    python forge.py queue "project description"   # Add to queue
    python forge.py projects                      # List all projects
    python forge.py deploy <project_id>           # Show deploy guide

Paper reproduction:
    python forge.py paper infer "research goal"   # Infer likely ICLR papers
    python forge.py paper benchmark               # Benchmark inference quality
    python forge.py paper reproduce "goal"        # Build reproduction brief/prompt
"""

from autoforge.cli.app import main

if __name__ == "__main__":
    main()
