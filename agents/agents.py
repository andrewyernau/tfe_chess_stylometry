#!/usr/bin/env python3
"""Agent command wrapper - Simplified command interface.

Usage:
    python agents.py /init
    python agents.py /agent <agent_name>
    python agents.py /list
"""

import sys
from pathlib import Path

# Add current directory to path to import agent_cli
sys.path.insert(0, str(Path(__file__).parent))

from agent_cli import AgentCLI


def main():
    """Main entry point for simplified agent commands."""
    
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python agents.py /init              - Initialize repository context")
        print("  python agents.py /agent <name>      - Execute agent task")
        print("  python agents.py /list              - List available agents")
        sys.exit(1)
    
    command = sys.argv[1].lower()
    cli = AgentCLI()
    
    try:
        if command == '/init':
            cli.cmd_init()
            
        elif command == '/agent':
            if len(sys.argv) < 3:
                print("[AGENT_CLI] Error: Agent name required")
                print("Usage: python agents.py /agent <agent_name>")
                sys.exit(1)
            agent_name = sys.argv[2]
            cli.cmd_agent(agent_name)
            
        elif command == '/list':
            cli.cmd_list()
            
        else:
            print(f"[AGENT_CLI] Unknown command: {command}")
            print("Available commands: /init, /agent, /list")
            sys.exit(1)
            
    except Exception as e:
        print(f"[AGENT_CLI] Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
