.PHONY: init agent list help jupyter clean test

# Python interpreter
PYTHON := python3

# Agent CLI script
AGENT_CLI := agents.py

# Directories
LABS := labs
VENV := venv

help:
	@echo "Chess Stylometry - Comandos Disponibles"
	@echo "========================================"
	@echo ""
	@echo "Sistema de Agentes:"
	@echo "  make init              - Inicializar contexto del repositorio"
	@echo "  make list              - Listar agentes disponibles"
	@echo "  make agent NAME=<name> - Ejecutar agente específico"
	@echo ""
	@echo "Desarrollo:"
	@echo "  make jupyter           - Iniciar Jupyter Notebook"
	@echo "  make clean             - Limpiar archivos temporales"
	@echo "  make test              - Ejecutar tests (cuando existan)"
	@echo ""
	@echo "Ejemplos:"
	@echo "  make init"
	@echo "  make agent NAME=architect"
	@echo "  make jupyter"
	@echo ""

init:
	@$(PYTHON) $(AGENT_CLI) /init

list:
	@$(PYTHON) $(AGENT_CLI) /list

agent:
ifndef NAME
	@echo "Error: Se requiere nombre de agente"
	@echo "Uso: make agent NAME=<nombre_agente>"
	@exit 1
endif
	@$(PYTHON) $(AGENT_CLI) /agent $(NAME)

jupyter:
	@echo "Iniciando Jupyter Notebook..."
	@./start_jupyter.sh

clean:
	@echo "Limpiando archivos temporales..."
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name ".ipynb_checkpoints" -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@find . -type f -name "*.pyo" -delete 2>/dev/null || true
	@find . -type f -name ".DS_Store" -delete 2>/dev/null || true
	@echo "✓ Limpieza completada"

test:
	@echo "Ejecutando tests..."
	@$(PYTHON) -m pytest $(LABS)/tests/ -v 2>/dev/null || echo "No hay tests implementados todavía"
