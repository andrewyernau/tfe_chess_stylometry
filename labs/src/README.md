# Source Directory (src/)

Este directorio contiene todos los módulos Python utilizados por el notebook `002_blocks_siamese_cnn.ipynb`.

## Módulos principales:

### Pipeline y procesamiento
- **`pipeline_stylometry_blocks.py`** - Pipeline principal de estilometría con bloques
  - Función: `discover_available_events()` - Descubre eventos en archivos PGN
  - Clase: `ChessStylometryPipelineV002` - Pipeline completo de extracción y generación

- **`extract_player_games_by_event_parallel.py`** - Extracción paralela de partidas por jugador
  - Extrae partidas de jugadores específicos de archivos PGN masivos
  - Procesamiento paralelo para optimizar rendimiento

- **`event_discovery_parallel.py`** - Descubrimiento paralelo de eventos en PGN
  - Escanea archivos PGN para identificar eventos disponibles
  - Estadísticas de partidas por evento

### Generación de imágenes
- **`generate_images_blocks.py`** - Generación de imágenes por bloques
  - Genera imágenes de tableros de ajedrez para bloques temporales específicos
  - Función: `generate_images_blocks()` - Procesamiento principal

- **`generate_decision_heatmaps.py`** - Generación de mapas de calor
  - Crea heatmaps de decisiones de jugadores
  - Clase: `GrayscaleHeatmapGenerator` - Generador de mapas de calor en escala de grises

- **`parse_games_to_images.py`** - Parser de partidas a imágenes
  - Función: `overlay_temporal_sequence()` - Crea secuencias temporales de tableros

### Modelo y entrenamiento
- **`multi_channel_model.py`** - Construcción del modelo CNN multicanal
  - Función: `build_multi_channel_model()` - Construye modelo Siamese CNN
  - Soporta múltiples bloques temporales como canales de entrada

- **`multi_channel_generator.py`** - Generadores de datos para entrenamiento
  - Función: `create_train_val_generators()` - Crea generadores de train/validation
  - Carga eficiente de imágenes en batches

## Dependencias entre módulos:

```
002_blocks_siamese_cnn.ipynb
├── pipeline_stylometry_blocks
│   ├── extract_player_games_by_event_parallel
│   ├── event_discovery_parallel
│   └── generate_images_blocks
│       ├── parse_games_to_images
│       └── generate_decision_heatmaps
├── multi_channel_generator
└── multi_channel_model
```

## Uso desde el notebook:

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path('../src').resolve()))

from pipeline_stylometry_blocks import discover_available_events, ChessStylometryPipelineV002
from multi_channel_generator import create_train_val_generators
from multi_channel_model import build_multi_channel_model
```

## Notas:
- Todos los módulos están diseñados para trabajar con la estructura de directorios de `labs/`
- Los imports internos entre módulos son relativos y funcionan automáticamente
- El directorio `src/` debe estar en el `sys.path` para importar los módulos
