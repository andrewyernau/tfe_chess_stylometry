# Agent CLI Commands

Sistema de comandos para gestionar agentes en el repositorio.

## Comandos disponibles

### `/init` - Inicializar contexto del repositorio

Escanea el repositorio y crea un archivo de contexto con:
- Estructura del repositorio
- Agentes disponibles
- Archivos Python y notebooks
- Metadatos del proyecto

```bash
python3 agents.py /init
```

**Output:**
```
[AGENT_CLI] Initializing repository context...
[AGENT_CLI] ✓ Context initialized at: context/repo_context.json
[AGENT_CLI] ✓ Discovered 1 agent(s):
  - architect: Responsible for system architecture and codebase structure
[AGENT_CLI] ✓ Tracked 2 Python file(s)
[AGENT_CLI] ✓ Tracked 1 notebook(s)
```

### `/agent <nombre>` - Ejecutar agente específico

Carga y ejecuta un agente específico basado en su definición en archivos `.md`.

```bash
python3 agents.py /agent architect
```

**Output:**
```
[AGENT_CLI] Executing agent: architect-agent
[AGENT_CLI] Description: Responsible for system architecture and codebase structure
[AGENT_CLI] Model: opus
[AGENT_CLI] Tools: code-search, repo-analyzer, Mermaid

[AGENT_CLI] Agent Configuration:
============================================================
[Muestra la configuración completa del agente]
============================================================

[AGENT_CLI] Agent ready to execute tasks based on the above configuration.
[AGENT_CLI] Context available at: context/repo_context.json
```

### `/list` - Listar agentes disponibles

Muestra todos los agentes descubiertos en el repositorio.

```bash
python3 agents.py /list
```

**Output:**
```
[AGENT_CLI] Available agents (1):

  architect:
    Name: architect-agent
    Description: Responsible for system architecture and codebase structure
    Model: opus
    Tools: code-search, repo-analyzer, Mermaid
    File: /home/user/repo/ARCHITECT.md
```

## Estructura de archivos de agente

Los agentes se definen en archivos `.md` en la raíz del repositorio con el siguiente formato:

```
---
name: agent-name
description: Agent description
model: opus|sonnet|haiku
tools: tool1, tool2, tool3
---

Additional agent documentation and instructions here.
```

### Ejemplo: ARCHITECT.md

```markdown
---
name: architect-agent
description: Responsible for system architecture and codebase structure
model: opus
tools: code-search, repo-analyzer, Mermaid
---

Capable of analyzing a repository and generating an architecture document (architecture.md) 
with a Mermaid diagram. For this complex task use Opus.
```

## Archivo de contexto

El comando `/init` genera `context/repo_context.json` con:

```json
{
  "repo_root": "/path/to/repo",
  "structure": {
    "labs": {
      "notebooks": {...},
      "dataset": {...}
    }
  },
  "agents": {
    "architect": {
      "name": "architect-agent",
      "description": "...",
      "model": "opus",
      "tools": ["code-search", "repo-analyzer", "Mermaid"],
      "file": "/path/to/ARCHITECT.md"
    }
  },
  "metadata": {
    "python_files": [...],
    "notebooks": [...],
    "key_directories": [...]
  }
}
```

## Alias convenientes (opcional)

Puedes crear alias en tu `.bashrc` o `.zshrc`:

```bash
alias agent-init='python3 ~/dev/jupyter/agents.py /init'
alias agent-run='python3 ~/dev/jupyter/agents.py /agent'
alias agent-list='python3 ~/dev/jupyter/agents.py /list'
```

Uso:
```bash
agent-init
agent-run architect
agent-list
```

## Agregar nuevos agentes

1. Crea un archivo `.md` en la raíz del repositorio (ej: `DATA-ANALYST.md`)
2. Usa el formato de frontmatter especificado arriba
3. Ejecuta `/init` para actualizar el contexto
4. Ejecuta `/agent data-analyst` para usar el nuevo agente

## Notas

- Los nombres de agentes se normalizan a minúsculas
- Se excluyen automáticamente: `README.md` y `AGENTS.md`
- El contexto se guarda en `context/repo_context.json`
- Los agentes pueden acceder al contexto para entender la estructura del repositorio
