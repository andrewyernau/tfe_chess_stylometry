#!/usr/bin/env python3
"""Agent CLI - Command line interface for managing repository agents.

This module provides commands to initialize repository context and execute
agent-specific tasks based on agent definitions found in .md files.
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import re


@dataclass
class AgentConfig:
    """Agent configuration parsed from markdown file.
    
    Attributes
    ----------
    name : str
        Agent name identifier.
    description : str
        Agent description.
    model : str
        Model to use (e.g., 'opus', 'sonnet').
    tools : List[str]
        List of tools available to the agent.
    content : str
        Full content of the agent definition.
    file_path : str
        Path to the agent definition file.
    """
    name: str
    description: str
    model: str
    tools: List[str]
    content: str
    file_path: str


class AgentCLI:
    """Command line interface for repository agents."""
    
    def __init__(self, repo_root: str = None):
        """Initialize AgentCLI.
        
        Parameters
        ----------
        repo_root : str, optional
            Repository root path. If None, uses current directory.
        """
        self.repo_root = Path(repo_root) if repo_root else Path.cwd()
        self.context_dir = self.repo_root / "context"
        self.context_file = self.context_dir / "repo_context.json"
        
    def parse_agent_file(self, file_path: Path) -> Optional[AgentConfig]:
        """Parse agent configuration from markdown file.
        
        Parameters
        ----------
        file_path : Path
            Path to agent markdown file.
            
        Returns
        -------
        Optional[AgentConfig]
            Parsed agent configuration or None if parsing fails.
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Extract frontmatter between --- delimiters (must be at start of file)
            # Only match if --- is at the very beginning (after optional whitespace)
            frontmatter_match = re.match(r'^\s*---\s*\n(.*?)\n---', content, re.MULTILINE | re.DOTALL)
            
            if not frontmatter_match:
                print(f"[AGENT_CLI] Warning: No frontmatter found in {file_path}")
                return None
            
            frontmatter = frontmatter_match.group(1)
            
            # Parse frontmatter fields
            name_match = re.search(r'name:\s*(.+)', frontmatter)
            desc_match = re.search(r'description:\s*(.+)', frontmatter)
            model_match = re.search(r'model:\s*(.+)', frontmatter)
            tools_match = re.search(r'tools:\s*(.+)', frontmatter)
            
            if not all([name_match, desc_match, model_match]):
                print(f"[AGENT_CLI] Warning: Missing required fields in {file_path}")
                return None
            
            # Parse tools (comma-separated list)
            tools = []
            if tools_match:
                tools = [t.strip() for t in tools_match.group(1).split(',')]
            
            return AgentConfig(
                name=name_match.group(1).strip(),
                description=desc_match.group(1).strip(),
                model=model_match.group(1).strip(),
                tools=tools,
                content=content,
                file_path=str(file_path)
            )
            
        except Exception as e:
            print(f"[AGENT_CLI] Error parsing {file_path}: {e}")
            return None
    
    def discover_agents(self) -> Dict[str, AgentConfig]:
        """Discover all agent definitions in repository root.
        
        Returns
        -------
        Dict[str, AgentConfig]
            Dictionary mapping agent names to their configurations.
        """
        agents = {}
        
        # Search for agent files in docs/agents/ directory
        agents_dir = self.repo_root / "docs" / "agents"
        if agents_dir.exists():
            for md_file in agents_dir.glob("*.md"):
                if md_file.name.upper() in ["README.MD"]:
                    continue
                
                agent_config = self.parse_agent_file(md_file)
                if agent_config:
                    # Use filename without extension as key (e.g., ARCHITECT.md -> architect)
                    agent_key = md_file.stem.lower()
                    agents[agent_key] = agent_config
        
        # Also check root for backward compatibility
        for md_file in self.repo_root.glob("*.md"):
            if md_file.name.upper() in ["README.MD", "QUICKSTART.MD", "AGENTS.MD"]:
                continue
            
            agent_config = self.parse_agent_file(md_file)
            if agent_config:
                agent_key = md_file.stem.lower()
                if agent_key not in agents:  # Don't override docs/agents/ files
                    agents[agent_key] = agent_config
                
        return agents
    
    def generate_repo_context(self) -> Dict[str, Any]:
        """Generate repository context information.
        
        Returns
        -------
        Dict[str, Any]
            Repository context dictionary.
        """
        context = {
            "repo_root": str(self.repo_root),
            "structure": {},
            "agents": {},
            "metadata": {
                "python_files": [],
                "notebooks": [],
                "key_directories": []
            }
        }
        
        # Discover agents
        agents = self.discover_agents()
        for agent_key, agent_config in agents.items():
            context["agents"][agent_key] = {
                "name": agent_config.name,
                "description": agent_config.description,
                "model": agent_config.model,
                "tools": agent_config.tools,
                "file": agent_config.file_path
            }
        
        # Scan repository structure (excluding venv, .git, etc.)
        exclude_dirs = {'.git', 'venv', '__pycache__', '.ipynb_checkpoints', 'node_modules'}
        
        for root, dirs, files in os.walk(self.repo_root):
            # Filter out excluded directories
            dirs[:] = [d for d in dirs if d not in exclude_dirs]
            
            rel_path = os.path.relpath(root, self.repo_root)
            if rel_path == '.':
                rel_path = 'root'
            
            # Track key directories
            if rel_path not in ['root'] and not rel_path.startswith('.'):
                context["metadata"]["key_directories"].append(rel_path)
            
            # Track Python files and notebooks
            for file in files:
                file_path = os.path.join(rel_path, file)
                if file.endswith('.py'):
                    context["metadata"]["python_files"].append(file_path)
                elif file.endswith('.ipynb'):
                    context["metadata"]["notebooks"].append(file_path)
        
        # Build structure tree
        context["structure"] = self._build_tree_structure()
        
        return context
    
    def _build_tree_structure(self) -> Dict[str, Any]:
        """Build directory tree structure.
        
        Returns
        -------
        Dict[str, Any]
            Tree structure of the repository.
        """
        tree = {}
        exclude_dirs = {'.git', 'venv', '__pycache__', '.ipynb_checkpoints', 'node_modules'}
        
        for root, dirs, files in os.walk(self.repo_root):
            dirs[:] = [d for d in dirs if d not in exclude_dirs]
            
            rel_path = os.path.relpath(root, self.repo_root)
            if rel_path == '.':
                current = tree
            else:
                parts = rel_path.split(os.sep)
                current = tree
                for part in parts:
                    if part not in current:
                        current[part] = {}
                    current = current[part]
            
            # Add files to current level
            if files:
                current['_files'] = [f for f in files if not f.startswith('.')]
        
        return tree
    
    def cmd_init(self) -> None:
        """Initialize repository context.
        
        Creates context directory and generates repo_context.json file
        with repository structure and agent information.
        """
        print("[AGENT_CLI] Initializing repository context...")
        
        # Create context directory if it doesn't exist
        self.context_dir.mkdir(exist_ok=True)
        
        # Generate context
        context = self.generate_repo_context()
        
        # Save to JSON file
        with open(self.context_file, 'w', encoding='utf-8') as f:
            json.dump(context, f, indent=2, ensure_ascii=False)
        
        print(f"[AGENT_CLI] ✓ Context initialized at: {self.context_file}")
        print(f"[AGENT_CLI] ✓ Discovered {len(context['agents'])} agent(s):")
        
        for agent_key, agent_info in context['agents'].items():
            print(f"  - {agent_key}: {agent_info['description']}")
        
        print(f"[AGENT_CLI] ✓ Tracked {len(context['metadata']['python_files'])} Python file(s)")
        print(f"[AGENT_CLI] ✓ Tracked {len(context['metadata']['notebooks'])} notebook(s)")
        
    def cmd_agent(self, agent_name: str) -> None:
        """Execute agent-specific task.
        
        Parameters
        ----------
        agent_name : str
            Name of the agent to execute.
            
        Raises
        ------
        ValueError
            If agent is not found or context is not initialized.
        """
        # Check if context exists
        if not self.context_file.exists():
            raise ValueError(
                "[AGENT_CLI] Context not initialized. Run '/init' first."
            )
        
        # Load context
        with open(self.context_file, 'r', encoding='utf-8') as f:
            context = json.load(f)
        
        # Find agent
        agent_key = agent_name.lower()
        if agent_key not in context['agents']:
            available = ", ".join(context['agents'].keys())
            raise ValueError(
                f"[AGENT_CLI] Agent '{agent_name}' not found. Available agents: {available}"
            )
        
        agent_info = context['agents'][agent_key]
        
        print(f"[AGENT_CLI] Executing agent: {agent_info['name']}")
        print(f"[AGENT_CLI] Description: {agent_info['description']}")
        print(f"[AGENT_CLI] Model: {agent_info['model']}")
        print(f"[AGENT_CLI] Tools: {', '.join(agent_info['tools'])}")
        print()
        
        # Load full agent configuration
        agent_config = self.parse_agent_file(Path(agent_info['file']))
        
        if agent_config:
            print("[AGENT_CLI] Agent Configuration:")
            print("=" * 60)
            print(agent_config.content)
            print("=" * 60)
            print()
            print("[AGENT_CLI] Agent ready to execute tasks based on the above configuration.")
            print("[AGENT_CLI] Context available at:", self.context_file)
    
    def cmd_list(self) -> None:
        """List all available agents."""
        agents = self.discover_agents()
        
        if not agents:
            print("[AGENT_CLI] No agents found in repository.")
            return
        
        print(f"[AGENT_CLI] Available agents ({len(agents)}):")
        print()
        
        for agent_key, agent_config in agents.items():
            print(f"  {agent_key}:")
            print(f"    Name: {agent_config.name}")
            print(f"    Description: {agent_config.description}")
            print(f"    Model: {agent_config.model}")
            print(f"    Tools: {', '.join(agent_config.tools)}")
            print(f"    File: {agent_config.file_path}")
            print()


def main():
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(
        description="Agent CLI - Manage repository agents",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # /init command
    subparsers.add_parser('init', help='Initialize repository context')
    
    # /agent command
    agent_parser = subparsers.add_parser('agent', help='Execute agent task')
    agent_parser.add_argument('name', help='Agent name')
    
    # /list command
    subparsers.add_parser('list', help='List all available agents')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    try:
        cli = AgentCLI()
        
        if args.command == 'init':
            cli.cmd_init()
        elif args.command == 'agent':
            cli.cmd_agent(args.name)
        elif args.command == 'list':
            cli.cmd_list()
            
    except Exception as e:
        print(f"[AGENT_CLI] Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
