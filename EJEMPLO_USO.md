# Ejemplo de Uso del Sistema de Agentes

Este documento muestra ejemplos prácticos de uso del sistema de agentes.

## Ejemplo 1: Inicializar el repositorio

```bash
$ make init
```

**Salida:**
```
[AGENT_CLI] Initializing repository context...
[AGENT_CLI] Warning: No frontmatter found in /home/andrewyernau/dev/jupyter/AGENT_CLI.md
[AGENT_CLI] ✓ Context initialized at: /home/andrewyernau/dev/jupyter/context/repo_context.json
[AGENT_CLI] ✓ Discovered 1 agent(s):
  - architect: Responsible for system architecture and codebase structure
[AGENT_CLI] ✓ Tracked 2 Python file(s)
[AGENT_CLI] ✓ Tracked 1 notebook(s)
```

## Ejemplo 2: Listar agentes disponibles

```bash
$ make list
```

**Salida:**
```
[AGENT_CLI] Available agents (1):

  architect:
    Name: architect-agent
    Description: Responsible for system architecture and codebase structure
    Model: opus
    Tools: code-search, repo-analyzer, Mermaid
    File: /home/andrewyernau/dev/jupyter/ARCHITECT.md
```

## Ejemplo 3: Ejecutar el agente architect

```bash
$ make agent NAME=architect
```

**Salida:**
```
[AGENT_CLI] Executing agent: architect-agent
[AGENT_CLI] Description: Responsible for system architecture and codebase structure
[AGENT_CLI] Model: opus
[AGENT_CLI] Tools: code-search, repo-analyzer, Mermaid

[AGENT_CLI] Agent Configuration:
============================================================
---
name: architect-agent
description: Responsible for system architecture and codebase structure
model: opus
tools: code-search, repo-analyzer, Mermaid
---

Capable of analyzing a repository and generating an architecture document 
(architecture.md) with a Mermaid diagram. For this complex task use Opus.
============================================================

[AGENT_CLI] Agent ready to execute tasks based on the above configuration.
[AGENT_CLI] Context available at: /home/andrewyernau/dev/jupyter/context/repo_context.json
```

El agente **architect** ahora tiene acceso a:
- Configuración completa del repositorio
- Estructura de directorios
- Lista de archivos Python y notebooks
- Metadatos del proyecto

## Ejemplo 4: Ver el contexto generado

```bash
$ cat context/repo_context.json | python3 -m json.tool | head -40
```

**Salida parcial:**
```json
{
    "repo_root": "/home/andrewyernau/dev/jupyter",
    "structure": {
        "_files": [
            "ARCHITECT.md",
            "AGENTS.md",
            "agent_cli.py",
            "agents.py",
            "README.md"
        ],
        "labs": {
            "src": {},
            "dataset": {
                "_files": ["chessgame0001.pgn"],
                "generated": {},
                "splits": {}
            },
            "notebooks": {
                "_files": ["chess-0000-preview.ipynb"]
            },
            "utils": {},
            "output": {}
        }
    },
    "agents": {
        "architect": {
            "name": "architect-agent",
            "description": "Responsible for system architecture and codebase structure",
            "model": "opus",
            "tools": ["code-search", "repo-analyzer", "Mermaid"],
            "file": "/home/andrewyernau/dev/jupyter/ARCHITECT.md"
        }
    }
}
```

## Ejemplo 5: Intentar ejecutar un agente inexistente

```bash
$ make agent NAME=nonexistent
```

**Salida:**
```
[AGENT_CLI] Error: [AGENT_CLI] Agent 'nonexistent' not found. Available agents: architect
make: *** [Makefile:28: agent] Error 1
```

## Ejemplo 6: Crear un nuevo agente (Data Analyst)

### Paso 1: Crear archivo de definición

```bash
$ cat > DATA-ANALYST.md << 'AGENT'
---
name: data-analyst-agent
description: Analyzes chess game datasets and generates statistical reports
model: sonnet
tools: pandas, numpy, matplotlib, seaborn
---

# Data Analyst Agent

Este agente está especializado en:

1. **Análisis de datasets PGN**: Parsea y analiza archivos de partidas
2. **Generación de estadísticas**: Calcula métricas de jugadores y partidas
3. **Visualizaciones**: Crea gráficos de distribuciones y tendencias
4. **Reportes**: Genera informes en formato Markdown con insights

## Tareas que puede realizar:

- Análisis exploratorio de datos (EDA) de partidas
- Distribución de ELO de jugadores
- Análisis de aperturas más comunes
- Tendencias temporales en el juego
- Correlaciones entre variables del juego

## Modelo y herramientas:

- **Modelo**: Sonnet (balance entre velocidad y capacidad)
- **Herramientas**: pandas, numpy, matplotlib, seaborn
AGENT
```

### Paso 2: Actualizar contexto

```bash
$ make init
[AGENT_CLI] Initializing repository context...
[AGENT_CLI] ✓ Context initialized at: /home/andrewyernau/dev/jupyter/context/repo_context.json
[AGENT_CLI] ✓ Discovered 2 agent(s):
  - architect: Responsible for system architecture and codebase structure
  - data-analyst: Analyzes chess game datasets and generates statistical reports
[AGENT_CLI] ✓ Tracked 2 Python file(s)
[AGENT_CLI] ✓ Tracked 1 notebook(s)
```

### Paso 3: Verificar que el agente fue registrado

```bash
$ make list
[AGENT_CLI] Available agents (2):

  architect:
    Name: architect-agent
    Description: Responsible for system architecture and codebase structure
    Model: opus
    Tools: code-search, repo-analyzer, Mermaid
    File: /home/andrewyernau/dev/jupyter/ARCHITECT.md

  data-analyst:
    Name: data-analyst-agent
    Description: Analyzes chess game datasets and generates statistical reports
    Model: sonnet
    Tools: pandas, numpy, matplotlib, seaborn
    File: /home/andrewyernau/dev/jupyter/DATA-ANALYST.md
```

### Paso 4: Ejecutar el nuevo agente

```bash
$ make agent NAME=data-analyst
[AGENT_CLI] Executing agent: data-analyst-agent
[AGENT_CLI] Description: Analyzes chess game datasets and generates statistical reports
[AGENT_CLI] Model: sonnet
[AGENT_CLI] Tools: pandas, numpy, matplotlib, seaborn
...
[AGENT_CLI] Agent ready to execute tasks based on the above configuration.
```

## Resumen

El sistema de agentes permite:

✅ **Auto-descubrimiento**: Los agentes se detectan automáticamente  
✅ **Configuración simple**: Solo crear un archivo .md con frontmatter  
✅ **Contexto compartido**: Todos los agentes acceden al repo_context.json  
✅ **Fácil extensión**: Agregar nuevos agentes es trivial  
✅ **Comandos simples**: make init, make list, make agent NAME=...  

