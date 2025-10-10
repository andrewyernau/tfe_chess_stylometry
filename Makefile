.PHONY: init agent list help

# Python interpreter
PYTHON := python3

# Agent CLI script
AGENT_CLI := agents.py

help:
	@echo "Agent CLI Commands"
	@echo "=================="
	@echo ""
	@echo "make init              - Initialize repository context"
	@echo "make list              - List all available agents"
	@echo "make agent NAME=<name> - Execute specific agent"
	@echo ""
	@echo "Examples:"
	@echo "  make init"
	@echo "  make list"
	@echo "  make agent NAME=architect"
	@echo ""

init:
	@$(PYTHON) $(AGENT_CLI) /init

list:
	@$(PYTHON) $(AGENT_CLI) /list

agent:
ifndef NAME
	@echo "Error: Agent name required"
	@echo "Usage: make agent NAME=<agent_name>"
	@exit 1
endif
	@$(PYTHON) $(AGENT_CLI) /agent $(NAME)
