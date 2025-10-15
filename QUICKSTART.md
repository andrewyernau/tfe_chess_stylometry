# Guía de Inicio Rápido

Guía rápida para comenzar a trabajar en el proyecto Chess Stylometry.

---

## 🚀 Instalación

### 1. Clonar y configurar entorno

```bash
# Clonar repositorio
git clone <repository-url>
cd jupyter

# Crear entorno virtual
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Instalar dependencias
pip install -r requirements.txt
```

### 2. Verificar instalación

```bash
# Comprobar que python-chess está instalado
python -c "import chess; print(chess.__version__)"

# Comprobar Jupyter
jupyter --version
```

---

## 📓 Trabajar con Jupyter

### Opción 1: Script de inicio rápido

```bash
./start_jupyter.sh
```

Este script inicia Jupyter y guarda los logs en `jupyter.log` (que no se subirá a Git).

### Opción 2: Inicio manual

```bash
jupyter notebook
```

### Notebooks disponibles

- `labs/notebooks/chess_cnn_visual_temporal.ipynb`: Experimentos con CNN y codificaciones visuales
- `labs/notebooks/chess-0000-preview-checkpoint.ipynb`: Preview inicial

---

## 🤖 Sistema de Agentes

El repositorio incluye un sistema automatizado de agentes para tareas comunes.

### Comandos básicos

```bash
# Inicializar contexto del repositorio
make init

# Ver agentes disponibles
make list

# Ejecutar un agente
make agent NAME=architect
```

### Agentes disponibles

Ver `docs/agents/` para documentación detallada de cada agente.

---

## 📁 Estructura de trabajo

### Datos

```bash
labs/dataset/
├── chessgame0001.pgn    # Archivos PGN de partidas
├── generated/           # Imágenes generadas (codificaciones)
└── splits/              # Train/Val/Test splits
```

### Código

```bash
labs/
├── src/                 # Scripts de producción
├── utils/               # Utilidades helper
├── notebooks/           # Jupyter notebooks
└── output/              # Resultados y visualizaciones
```

---

## 📖 Documentación

- **README.md**: Visión general del proyecto
- **docs/architecture.md**: Arquitectura detallada del sistema
- **docs/agents/**: Sistema de agentes
- **docs/*.pdf**: Papers de referencia

---

## 🔧 Comandos útiles

### Make commands

```bash
make help               # Ver ayuda
make init               # Inicializar contexto de agentes
make list               # Listar agentes
make agent NAME=X       # Ejecutar agente X
```

### Python

```bash
# Ejecutar scripts (cuando existan)
python labs/src/parser.py --input labs/dataset/chessgame0001.pgn

# Tests (cuando existan)
pytest labs/tests/
```

---

## ⚠️ Notas importantes

- El archivo `jupyter.log` contiene tokens sensibles y **NO debe subirse a Git**
- Los checkpoints de Jupyter (`.ipynb_checkpoints/`) están ignorados en Git
- El entorno virtual `venv/` no se sube al repositorio

---

## 🆘 Solución de problemas

### Jupyter no inicia

```bash
# Reinstalar Jupyter
pip install --upgrade jupyter notebook

# Verificar que el puerto no esté ocupado
jupyter notebook list
```

### Problemas con dependencias

```bash
# Actualizar pip
pip install --upgrade pip

# Reinstalar dependencias
pip install -r requirements.txt --force-reinstall
```

---

## 📚 Próximos pasos

1. Explorar los notebooks en `labs/notebooks/`
2. Revisar la documentación en `docs/`
3. Familiarizarse con el sistema de agentes
4. Comenzar a implementar parsers y encoders

---

**Para más información**, consulta [README.md](README.md) o la documentación en `docs/`.
