# Agent System - Quick Start Guide

Sistema de comandos para gestionar y ejecutar agentes en el repositorio.

## 🚀 Uso Rápido

### Con Makefile (Recomendado)

```bash
# Ver ayuda
make help

# Inicializar contexto del repositorio
make init

# Listar agentes disponibles
make list

# Ejecutar un agente específico
make agent NAME=architect
```

### Con Python directamente

```bash
# Inicializar contexto
python3 agents.py /init

# Listar agentes
python3 agents.py /list

# Ejecutar agente
python3 agents.py /agent architect
```

## 📋 Comandos Disponibles

| Comando | Descripción | Ejemplo |
|---------|-------------|---------|
| `make init` | Inicializa el contexto del repositorio | `make init` |
| `make list` | Lista todos los agentes disponibles | `make list` |
| `make agent NAME=<name>` | Ejecuta un agente específico | `make agent NAME=architect` |

## 🤖 Agentes Actuales

### architect-agent
- **Descripción**: Responsable de arquitectura del sistema y estructura del código
- **Modelo**: Opus
- **Herramientas**: code-search, repo-analyzer, Mermaid
- **Archivo**: `ARCHITECT.md`
- **Uso**: `make agent NAME=architect`

## 📁 Archivos del Sistema

```
jupyter/
├── agent_cli.py        # Sistema principal de CLI para agentes
├── agents.py           # Wrapper simplificado para comandos
├── Makefile           # Comandos make para facilitar uso
├── AGENT_CLI.md       # Documentación detallada del CLI
├── AGENTS.md          # Reglas de desarrollo de agentes
├── ARCHITECT.md       # Definición del agente architect
└── context/
    └── repo_context.json  # Contexto generado del repositorio
```

## 🆕 Crear un Nuevo Agente

1. **Crear archivo de definición** en la raíz del repositorio (ej: `DATA-ANALYST.md`):

```markdown
---
name: data-analyst-agent
description: Analiza datasets y genera reportes estadísticos
model: sonnet
tools: pandas, numpy, matplotlib
---

Este agente se encarga de analizar datos de partidas de ajedrez
y generar visualizaciones y reportes estadísticos.
```

2. **Actualizar el contexto**:
```bash
make init
```

3. **Verificar que el agente fue registrado**:
```bash
make list
```

4. **Ejecutar el nuevo agente**:
```bash
make agent NAME=data-analyst
```

## 📊 Archivo de Contexto

El comando `make init` genera `context/repo_context.json` con:

- ✅ Estructura completa del repositorio
- ✅ Lista de agentes disponibles con sus configuraciones
- ✅ Archivos Python y notebooks detectados
- ✅ Directorios clave del proyecto
- ✅ Metadatos del repositorio

Los agentes pueden usar este contexto para entender la estructura del proyecto antes de ejecutar tareas.

## 🔧 Reglas de Desarrollo

Todos los agentes deben seguir las reglas definidas en `AGENTS.md`:

- ✅ snake_case para nombres de variables y funciones
- ✅ SCREAMING_SNAKE_CASE para constantes
- ✅ Docstrings (NumPy/Google style) en todas las funciones
- ✅ Type annotations explícitas
- ✅ Unit tests obligatorios
- ✅ Errores con prefijo `[AGENT_NAME]`

## 📖 Documentación Completa

Para información detallada sobre:
- Formato de archivos de agente
- Estructura del archivo de contexto
- Creación de alias
- Ejemplos avanzados

Ver: **[AGENT_CLI.md](AGENT_CLI.md)**

## ⚡ Ejemplos de Uso

### Inicializar el repositorio
```bash
$ make init
[AGENT_CLI] Initializing repository context...
[AGENT_CLI] ✓ Context initialized at: context/repo_context.json
[AGENT_CLI] ✓ Discovered 1 agent(s):
  - architect: Responsible for system architecture and codebase structure
[AGENT_CLI] ✓ Tracked 2 Python file(s)
[AGENT_CLI] ✓ Tracked 1 notebook(s)
```

### Listar agentes
```bash
$ make list
[AGENT_CLI] Available agents (1):

  architect:
    Name: architect-agent
    Description: Responsible for system architecture and codebase structure
    Model: opus
    Tools: code-search, repo-analyzer, Mermaid
    File: /path/to/ARCHITECT.md
```

### Ejecutar agente
```bash
$ make agent NAME=architect
[AGENT_CLI] Executing agent: architect-agent
[AGENT_CLI] Description: Responsible for system architecture and codebase structure
[AGENT_CLI] Model: opus
[AGENT_CLI] Tools: code-search, repo-analyzer, Mermaid
...
[AGENT_CLI] Agent ready to execute tasks based on the above configuration.
```
