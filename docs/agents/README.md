# Sistema de Agentes

Documentación del sistema automatizado de agentes del proyecto.

---

## 📖 Archivos

- **[AGENT_CLI.md](AGENT_CLI.md)**: Documentación técnica completa del CLI
- **[AGENTS.md](AGENTS.md)**: Reglas de desarrollo para agentes
- **[ARCHITECT.md](ARCHITECT.md)**: Definición del agente architect

---

## 🚀 Uso Rápido

```bash
# Inicializar contexto
make init

# Listar agentes disponibles
make list

# Ejecutar agente
make agent NAME=architect
```

---

## AI Agentes Disponibles

### architect
- **Propósito**: Análisis de arquitectura y generación de documentación
- **Modelo**: Opus
- **Herramientas**: code-search, repo-analyzer, Mermaid

---

## 🆕 Crear Nuevo Agente

1. Crear archivo `docs/agents/MI-AGENTE.md`:

```markdown
---
name: mi-agente
description: Descripción del agente
model: sonnet
tools: herramienta1, herramienta2
---

Instrucciones y contexto del agente aquí.
```

2. Reinicializar contexto:

```bash
make init
```

3. Ejecutar:

```bash
make agent NAME=mi-agente
```

---

Ver [AGENT_CLI.md](AGENT_CLI.md) para documentación completa.
