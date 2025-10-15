# Estructura del Repositorio

Mapa visual de la organización del proyecto Chess Stylometry.

---

## �� Árbol de Directorios

```
jupyter/                         # 🏠 Raíz del proyecto
│
├── 📁 labs/                     # 🔬 Código principal
│   ├── dataset/                 # Datos de partidas
│   │   ├── chessgame0001.pgn   # PGN de ejemplo
│   │   ├── generated/          # Imágenes generadas
│   │   └── splits/             # Train/Val/Test
│   ├── notebooks/              # Jupyter notebooks
│   │   ├── chess_cnn_visual_temporal.ipynb
│   │   └── chess-0000-preview-checkpoint.ipynb
│   ├── src/                    # Scripts de producción
│   ├── utils/                  # Utilidades helper
│   └── output/                 # Resultados y gráficos
│       ├── chess_visual_temporal_*.png
│       └── cnn_friendly_*.png
│
├── 📁 docs/                     # 📚 Documentación
│   ├── agents/                 # Sistema de agentes
│   │   ├── README.md          # Índice de agentes
│   │   ├── AGENT_CLI.md       # Doc técnica CLI
│   │   ├── AGENTS.md          # Reglas de desarrollo
│   │   └── ARCHITECT.md       # Agente architect
│   ├── architecture.md         # Arquitectura del sistema
│   ├── chess_cnn_visual_temporal.md
│   ├── siamese_neural_networks.md
│   ├── websites.md
│   ├── chess_rating_estimation.pdf
│   └── detection_stylometry.pdf
│
├── 📁 context/                  # Contexto generado
│   └── repo_context.json       # Metadata del repo
│
├── 🐍 agent_cli.py             # Sistema CLI de agentes
├── 🐍 agents.py                # Wrapper de agentes
│
├── 📄 README.md                # Documentación principal
├── 📄 QUICKSTART.md            # Guía de inicio rápido
├── 📄 CONTRIBUTING.md          # Guía de contribución
├── 📄 ESTRUCTURA.md            # Este archivo
│
├── 📄 requirements.txt         # Dependencias Python
├── 📄 Makefile                 # Comandos automatizados
├── 📄 start_jupyter.sh         # Script de inicio
│
├── 🔧 .gitignore               # Archivos ignorados
└── 🔧 .gitattributes           # Atributos Git
```

---

## 🎯 Archivos Clave

### Documentación Principal

| Archivo | Propósito |
|---------|-----------|
| `README.md` | Visión general del proyecto |
| `QUICKSTART.md` | Guía de inicio rápido |
| `CONTRIBUTING.md` | Convenciones de código |
| `docs/architecture.md` | Arquitectura detallada |

### Sistema de Agentes

| Archivo | Propósito |
|---------|-----------|
| `agent_cli.py` | Core del sistema CLI |
| `agents.py` | Wrapper simplificado |
| `Makefile` | Comandos make |
| `docs/agents/` | Definiciones de agentes |

### Código del Proyecto

| Directorio | Propósito |
|------------|-----------|
| `labs/src/` | Scripts de producción |
| `labs/utils/` | Utilidades helper |
| `labs/notebooks/` | Experimentación |
| `labs/dataset/` | Datos y resultados |

---

## 🚀 Comandos Rápidos

```bash
# Setup inicial
make init                    # Inicializar contexto

# Desarrollo
make jupyter                 # Iniciar Jupyter
make clean                   # Limpiar temporales
make test                    # Ejecutar tests

# Agentes
make list                    # Ver agentes
make agent NAME=architect    # Ejecutar agente
```

---

## 📝 Flujo de Trabajo

```
1. Clonar repo
   ↓
2. Crear venv y instalar deps (requirements.txt)
   ↓
3. Explorar notebooks (labs/notebooks/)
   ↓
4. Implementar código (labs/src/)
   ↓
5. Probar (labs/tests/)
   ↓
6. Documentar (docs/)
```

---

## 🔒 Archivos Ignorados (.gitignore)

- `jupyter.log` ⚠️ **CRÍTICO: contiene tokens**
- `venv/`, `__pycache__/`
- `.ipynb_checkpoints/`
- `labs/models/*.pth` (checkpoints grandes)
- `labs/dataset/generated/*.png` (imágenes generadas)

---

## 📦 Dependencias (requirements.txt)

```
python-chess    # Parser PGN
numpy, pandas   # Datos
matplotlib      # Visualización
jupyter         # Notebooks
pytest          # Testing
```

---

**Última actualización**: Octubre 2025
