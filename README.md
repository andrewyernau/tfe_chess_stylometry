# Chess Stylometry - TFE

**Trabajo de Fin de Estudios - Ingeniería de Telecomunicación**  
**Universidad Politécnica de Cartagena**

Identificación de jugadores de ajedrez mediante análisis estilométrico usando Redes Neuronales Convolucionales (CNN) y representaciones visuales de partidas.

---

## Descripción

Este proyecto investiga **stylometry** (estilometría) aplicada al ajedrez: identificar jugadores por su estilo de juego característico. A diferencia de métodos tradicionales basados en features manuales, utilizamos **representaciones visuales** de partidas como entrada para CNNs.

### Enfoque

Basado en papers de investigación (ver `docs/`), exploramos múltiples codificaciones visuales:

- **Mapas de calor**: Frecuencia de movimientos y zonas de presión
- **Trayectorias temporales**: Flujo de piezas durante la partida  
- **Campos vectoriales**: Direcciones y magnitudes de amenazas
- **Estados de tablero**: Representaciones posicionales secuenciales

## Objetivos

- [x] Diseñar pipeline de procesamiento PGN
- [x] Implementar múltiples codificaciones visuales
- [x] Entrenar y evaluar modelos CNN (Siamese Network)
- [x] **NUEVO v2.0:** Implementar CNN dual-channel con mapas de calor temporales y verificar su efectividad
- [ ] Comparar diferentes representaciones
- [ ] Analizar interpretabilidad de patrones aprendidos

---

## 📁 Estructura del Repositorio

```
jupyter/
├── labs/                      # 🔬 Código principal del proyecto
│   ├── dataset/               # Datos PGN y generados
│   │   ├── *.pgn             # Archivos PGN originales
│   │   ├── generated/        # Imágenes generadas
│   │   └── splits/           # Train/Val/Test
│   ├── notebooks/            # Jupyter notebooks
│   ├── src/                  # Scripts de producción
│   ├── utils/                # Utilidades
│   └── output/               # Resultados y visualizaciones
│
├── docs/                      # 📚 Documentación y papers
│   ├── *.pdf                 # Papers de referencia
│   ├── *.md                  # Documentación técnica
│   └── agents/               # Docs del sistema de agentes
│
├── context/                   # Contexto generado automáticamente
├── agent_cli.py              # Sistema CLI de agentes
├── agents.py                 # Wrapper de agentes
├── Makefile                  # Comandos automáticos
├── requirements.txt          # Dependencias Python
└── start_jupyter.sh          # Script de inicio rápido
```

---

---

## ✨ NUEVO: Estilometría Conductual v2.0

**Sistema completo de CNN dual-channel para análisis estilométrico temporal de ajedrez**

### Características Principales
- 🔍 **Extracción eficiente** de partidas desde archivos PGN masivos (188GB)
- 🎨 **Mapas de calor** de tiempos de decisión por movimiento
- 🧠 **CNN dual-channel** con ResNet50 pre-entrenado
- ⚡ **Pipeline automatizado** de 3 fases
- 📊 **Validación automática** con métricas de aprendizaje

### Inicio Rápido v2.0

```bash
cd labs

# Procesar datos y generar imágenes
python pipeline_stylometry.py \
  --pgn-file dataset/generated/lichess_db.pgn \
  --players Magnus Hikaru Fabiano \
  --max-games 50 --timeout 600

# Entrenar CNN dual-channel
jupyter notebook notebooks/0002_dual_channel_cnn.ipynb
```

📖 **Documentación completa:** [`labs/QUICK_START.md`](labs/QUICK_START.md)

---

## 🚀 Inicio Rápido (Sistema Base)

### 1. Instalación

```bash
# Clonar repositorio
git clone <repository-url>
cd jupyter

# Crear y activar entorno virtual
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Instalar dependencias
pip install -r requirements.txt
```

### 2. Usar Jupyter

```bash
# Iniciar Jupyter Notebook
./start_jupyter.sh

# O manualmente
jupyter notebook
```

### 3. Sistema de Agentes

```bash
# Inicializar contexto del repositorio
make init

# Listar agentes disponibles  
make list

# Ejecutar agente específico
make agent NAME=architect
```

Ver [QUICKSTART.md](QUICKSTART.md) para más detalles.

---

## 📊 Tecnologías

- **Python 3.12**: Lenguaje principal
- **python-chess**: Análisis de partidas PGN
- **NumPy/Pandas**: Procesamiento de datos
- **Matplotlib/Seaborn**: Visualización
- **PyTorch**: Deep Learning (a implementar)
- **Jupyter**: Notebooks interactivos

---

## 📖 Documentación

- **[QUICKSTART.md](QUICKSTART.md)**: Guía de inicio rápido
- **[docs/architecture.md](docs/architecture.md)**: Arquitectura detallada del sistema  
- **[docs/agents/](docs/agents/)**: Sistema de agentes automatizados
- **[docs/*.pdf](docs/)**: Papers de referencia

---

## 🤖 Sistema de Agentes

Este proyecto incluye un sistema automatizado de agentes para tareas repetitivas:

```bash
make init              # Inicializar contexto
make list              # Ver agentes disponibles
make agent NAME=X      # Ejecutar agente
```

Ver documentación completa en [docs/agents/](docs/agents/).

---

## 📝 Estado del Proyecto

**Fase actual**: Desarrollo de pipeline de datos

- [x] Configuración del repositorio
- [x] Sistema de agentes funcional
- [x] Documentación base
- [ ] Pipeline de procesamiento PGN
- [ ] Generadores de codificaciones visuales
- [ ] Modelos CNN

---

## 👤 Autor

**André Yermak Naumenko**  
Universidad Politécnica de Cartagena  
Grado en Ingeniería Telemática  
Año 2025

---

**Última actualización**: Octubre 2025

