# Chess Stylometry – Documentación Exhaustiva para Estudio

**Proyecto**: Trabajo de Fin de Estudios – Ingeniería Telemática (UPCT)  
**Autor**: André Yermak Naumenko  
**Fecha**: Febrero 2026

---

## Índice

1. [Visión General del Proyecto](#1-visión-general-del-proyecto)
2. [Arquitectura del Pipeline Completo](#2-arquitectura-del-pipeline-completo)
3. [Fase 1: Descubrimiento de Eventos (event_discovery_parallel.py)](#3-fase-1-descubrimiento-de-eventos)
4. [Fase 2: Extracción de Jugadores y Partidas (extract_player_games_by_event_parallel.py)](#4-fase-2-extracción-de-jugadores-y-partidas)
5. [Fase 3: Generación de Heatmaps de Decisión (generate_decision_heatmaps.py)](#5-fase-3-generación-de-heatmaps-de-decisión)
6. [Fase 4: Generación de Imágenes por Bloques Temporales (generate_images_blocks.py)](#6-fase-4-generación-de-imágenes-por-bloques-temporales)
7. [Fase 5: Orquestación del Pipeline (pipeline_stylometry_blocks.py)](#7-fase-5-orquestación-del-pipeline)
8. [Fase 6: Exportación de Embeddings (embedding_exporter.py)](#8-fase-6-exportación-de-embeddings)
9. [Notebook 101: Red Siamesa con Validación por ELO](#9-notebook-101-red-siamesa-con-validación-por-elo)
   - 9.1 [Configuración y Selección de Jugadores](#91-configuración-y-selección-de-jugadores)
   - 9.2 [Descubrimiento de Muestras (GameSample)](#92-descubrimiento-de-muestras)
   - 9.3 [Split Train/Validation](#93-split-trainvalidation)
   - 9.4 [Pipeline de Datos: Tripletes y tf.data](#94-pipeline-de-datos-tripletes-y-tfdata)
   - 9.5 [Arquitectura del Modelo de Embeddings](#95-arquitectura-del-modelo-de-embeddings)
   - 9.6 [Red Siamesa y Triplet Loss](#96-red-siamesa-y-triplet-loss)
   - 9.7 [Entrenamiento](#97-entrenamiento)
   - 9.8 [Evaluación de Distancias](#98-evaluación-de-distancias)
   - 9.9 [Verificación de Embeddings](#99-verificación-de-embeddings)
   - 9.10 [Exportación de Centroides](#910-exportación-de-centroides)
10. [Conceptos Clave para el Examen](#10-conceptos-clave-para-el-examen)

---

## 1. Visión General del Proyecto

### ¿Qué es la Estilometría en Ajedrez?

La **estilometría** es la disciplina que identifica autoría mediante análisis de estilo. Aplicada al ajedrez, busca **identificar quién jugó una partida** analizando patrones de juego característicos (posiciones favoritas, tiempos de reflexión, zonas del tablero preferidas).

### Enfoque del Proyecto

En lugar de usar features manuales (como apertura favorita, tendencia de sacrificio, etc.), este proyecto utiliza **representaciones visuales** de las partidas como entrada para redes neuronales:

```
Partida PGN → Imágenes visuales (tablero + heatmap) → CNN/Red Siamesa → Embedding 256D → Identificación
```

### ¿Por qué imágenes?

1. **Board images**: Capturan la posición acumulada del tablero (qué posiciones se alcanzan)
2. **Heatmaps de decisión**: Capturan el patrón temporal de reflexión (dónde piensa más el jugador)

Combinando ambas modalidades, se codifica tanto **QUÉ posiciones se alcanzan** como **DÓNDE se concentra la atención/tiempo del jugador**.

### Flujo General

```
┌─────────────────────────────────────────────────────────────────────┐
│                      PIPELINE COMPLETO                               │
│                                                                      │
│  PGN masivo (Lichess)                                                │
│       │                                                              │
│       ▼                                                              │
│  ┌─────────────────────┐                                             │
│  │ event_discovery_     │  → Descubre tipos de evento disponibles     │
│  │ parallel.py          │    (Blitz, Bullet, Classical, etc.)         │
│  └─────────┬───────────┘                                             │
│            ▼                                                         │
│  ┌─────────────────────┐                                             │
│  │ extract_player_     │  → Filtra y extrae PGNs individuales        │
│  │ games_by_event_     │    por jugador para un evento específico     │
│  │ parallel.py         │                                             │
│  └─────────┬───────────┘                                             │
│            ▼                                                         │
│  ┌─────────────────────┐   ┌─────────────────────┐                   │
│  │ generate_images_    │   │ generate_decision_   │                   │
│  │ blocks.py           │   │ heatmaps.py          │                   │
│  │ (Board images)      │   │ (Heatmap images)     │                   │
│  └─────────┬───────────┘   └─────────┬───────────┘                   │
│            ▼                          ▼                               │
│  ┌──────────────────────────────────────────┐                        │
│  │ board_images/player/game_XXXX_block_YY.png │                      │
│  │ heatmap_images/player/game_XXXX_block_YY.png │                    │
│  └─────────────────────┬────────────────────┘                        │
│                        ▼                                             │
│  ┌─────────────────────────────────────┐                             │
│  │ Notebook 101: Red Siamesa           │                             │
│  │  · Carga imágenes (board + heat)    │                             │
│  │  · Genera embeddings 256D           │                             │
│  │  · Entrena con Triplet Loss         │                             │
│  │  · Evalúa distancias intra/inter    │                             │
│  └─────────────────────┬───────────────┘                             │
│                        ▼                                             │
│  ┌─────────────────────────────────────┐                             │
│  │ Embeddings + Centroides por jugador │                             │
│  │  · embedding: vector 256D por game  │                             │
│  │  · centroide: media de embeddings   │                             │
│  └─────────────────────────────────────┘                             │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 2. Arquitectura del Pipeline Completo

### Estructura de Directorios

```
labs/
├── src/                                  # Scripts del pipeline
│   ├── event_discovery_parallel.py       # Descubrimiento de eventos en PGN
│   ├── extract_player_games_by_event_parallel.py  # Extracción de partidas
│   ├── generate_decision_heatmaps.py     # Generación de heatmaps
│   ├── generate_images_blocks.py         # Generación de imágenes de tablero
│   ├── pipeline_stylometry_blocks.py     # Orquestador principal
│   ├── embedding_exporter.py             # Exportador de embeddings (PyTorch/ResNet18)
│   ├── multi_channel_generator.py        # Generador Keras para entrenamiento
│   ├── multi_channel_model.py            # Modelo multi-canal clasificador
│   └── parse_games_to_images.py          # Versión legacy (imagen única)
├── notebooks/
│   ├── 000_siamese_nn.ipynb              # Idea inicial de red siamesa
│   ├── 001_single_channel_cnn.ipynb      # CNN canal único
│   ├── 002_blocks_siamese_cnn.ipynb      # CNN con bloques
│   ├── 003_mass_dataset_processing.ipynb # Procesamiento masivo
│   ├── 100_chess_siamese.ipynb           # Prototipo principal (100 jugadores)
│   ├── 101_chess_siamese_validation.ipynb # Validación con 10 jugadores (NUEVO)
│   └── output/events/Rated_Blitz_game/   # Dataset generado
│       ├── player_pgns/                  # ~1500 PGNs individuales
│       ├── board_images/                 # ~1482 jugadores con imágenes
│       └── heatmap_images/               # Heatmaps de decisión
└── dataset/
    ├── testpgns/                         # 10 PGNs de prueba (clásicos)
    └── embedding/                        # Embeddings pre-computados (1238 jugadores)
```

### Nomenclatura de Archivos de Imagen

```
game_0001_block_01.png   → Partida 1, Bloque temporal 1 (jugadas 15-30)
game_0001_block_02.png   → Partida 1, Bloque temporal 2 (jugadas 30-45)
game_0001_block_03.png   → Partida 1, Bloque temporal 3 (jugadas 45-60)
```

Cada partida genera **3 bloques** × **2 tipos** (board + heat) = **6 imágenes** por partida.

---

## 3. Fase 1: Descubrimiento de Eventos

**Script**: `labs/src/event_discovery_parallel.py`

### ¿Qué hace?

Escanea archivos PGN masivos (30+ GB comprimidos, 200+ GB descomprimidos) de Lichess para descubrir qué tipos de eventos contienen (Blitz, Bullet, Classical, etc.).

### ¿Cómo lo hace?

```python
class ParallelEventDiscovery:
    def __init__(self, pgn_path: Path, num_workers=None, max_workers=None):
        # Auto-calcula workers según RAM disponible (~1GB por worker)
    
    def discover_events(self, max_games=None, verbose=True) -> Dict:
        # Retorna: {evento: {games, unique_players, white_wins, black_wins, draws}}
```

**Algoritmo en 3 fases:**

1. **Chunk Boundaries**: Divide el archivo PGN en chunks alineados a partidas completas.
   Busca marcadores `[Event` en los bytes del archivo para encontrar fronteras seguras.

2. **Procesamiento paralelo por chunks**: Cada worker lee su chunk con un `LimitedReader`
   (buffer de 1MB) para mantener RAM baja (~100-200MB/worker).
   ```python
   def process_chunk(chunk_info, pgn_path):
       # Streaming: lee game por game con chess.pgn.read_game()
       # Extrae headers: Event, White, Black, Result
       # Acumula estadísticas en defaultdict
   ```

3. **Fusión de resultados**: Combina estadísticas de todos los chunks con GC periódico.

### Entrada/Salida

- **Entrada**: Ruta al PGN masivo (e.g., `lichess_db_2024-01.pgn`)
- **Salida**: Diccionario con estadísticas por evento:
  ```python
  {
      "Rated Blitz game": {"games": 5000000, "unique_players": 200000, ...},
      "Rated Bullet game": {"games": 3000000, ...},
      ...
  }
  ```

### Punto clave para examen

La paralelización requiere **alinear chunks a fronteras de partida**. No se puede cortar el archivo PGN en posiciones arbitrarias porque una partida podría quedar partida entre dos chunks.

---

## 4. Fase 2: Extracción de Jugadores y Partidas

**Script**: `labs/src/extract_player_games_by_event_parallel.py`

### ¿Qué hace?

Dado un tipo de evento (e.g., "Rated Blitz game"), extrae las partidas de los N jugadores más activos, generando un PGN individual por jugador.

### ¿Cómo lo hace?

```python
class ParallelPlayerExtractorByEvent:
    def __init__(self, pgn_path, event_type, num_players=20, 
                 games_per_player=30, min_games_threshold=0.7, num_workers=None):
```

**Algoritmo en 3 fases:**

1. **Descubrimiento de jugadores** (paralelo):
   ```python
   discover_players_in_chunk(chunk_info, pgn_path, event_type):
       # Filtra partidas por Event header == event_type
       # Cuenta frecuencia de aparición de cada jugador
       # Retorna: Counter({player: num_appearances})
   ```
   
2. **Extracción de partidas** (paralelo):
   ```python
   extract_games_from_chunk(chunk_info, pgn_path, event_type, selected_players, games_per_player):
       # Filtra por evento Y por jugador seleccionado
       # Extrae objetos chess.pgn.Game completos
       # Respeta límite games_per_player
   ```

3. **Filtrado por umbral**: Solo jugadores con `total_games >= 0.7 * games_per_player`

### Estructura de Datos

```python
@dataclass
class PlayerGameStats:
    white_games: List[chess.pgn.Game]   # Partidas como blancas
    black_games: List[chess.pgn.Game]   # Partidas como negras
    
    @property
    def total_games(self) -> int:
        return len(self.white_games) + len(self.black_games)
```

### Salida

```
events/Rated_Blitz_game/player_pgns/
├── Ashot2001.pgn       # Todas las partidas de Ashot2001
├── DeathIsReal.pgn     # Todas las partidas de DeathIsReal
├── ...
└── Oganian_Miran.pgn
```

---

## 5. Fase 3: Generación de Heatmaps de Decisión

**Script**: `labs/src/generate_decision_heatmaps.py`

### ¿Qué hace?

Convierte los **tiempos de reflexión** de cada jugada en una imagen de calor (heatmap) en escala de grises sobre el tablero 8×8, escalada a 192×192 píxeles.

### Concepto Clave

Los archivos PGN de Lichess contienen anotaciones de reloj en cada jugada:
```
1. e4 { [%clk 2:59] } 1... e5 { [%clk 2:58] }
2. Nf3 { [%clk 2:55] } 2... Nc6 { [%clk 2:54] }
```

El **tiempo de decisión** se calcula como:
```
decision_time = reloj_anterior - reloj_actual - incremento
```

### Algoritmo detallado

```python
class GrayscaleHeatmapGenerator:
    def generate_heatmap(game, start_move=1, end_move=None, percentile_clip=95.0):
        
        # 1. EXTRAER tiempos de reloj de las anotaciones PGN
        for move in game.mainline():
            clock = parse_clk(move.comment)  # "[%clk H:MM:SS]" → segundos
            decision_time = prev_clock - clock - increment
        
        # 2. CALCULAR porcentaje de decisión
        decision_pct = (decision_time / tiempo_base) * 100
        
        # 3. MAPEAR a escala de grises [30, 255]
        #    - 0 (negro) = sin datos
        #    - 30-255 = decisión rápida → lenta
        #    - Percentil 95 = clipping (evita outliers)
        
        grid = np.zeros((8, 8))  # Tablero 8×8
        for move_decision in decisions:
            square = move.from_square  # Casilla de ORIGEN del movimiento
            row, col = divmod(square, 8)
            grid[7-row][col] += decision_pct  # Acumula tiempo
        
        # 4. NORMALIZAR con clipping al percentil 95
        p95 = np.percentile(grid[grid > 0], 95)
        normalized = np.clip(grid / p95, 0, 1) * 225 + 30
        
        # 5. ESCALAR a 192×192 con INTER_NEAREST
        heatmap_192 = cv2.resize(grid_8x8, (192, 192), interpolation=INTER_NEAREST)
        
        # 6. DIBUJAR líneas de cuadrícula
        draw_grid_lines(heatmap_192, line_color=(80, 80, 80))
```

### Interpretación Visual

```
Negro (0)      = Sin datos / casilla no usada
Gris oscuro    = Decisiones rápidas (el jugador mueve sin pensar mucho)
Gris claro     = Decisiones moderadas
Blanco (255)   = Decisiones muy lentas (el jugador piensa mucho aquí)
```

### Punto clave para examen

El heatmap codifica **estilo de pensamiento**: un jugador que siempre piensa mucho en el centro tendrá las casillas centrales claras. Otro que piensa más en los flancos tendrá patrones laterales. Esta es la "huella digital temporal" del jugador.

---

## 6. Fase 4: Generación de Imágenes por Bloques Temporales

**Script**: `labs/src/generate_images_blocks.py`

### ¿Qué hace?

Genera imágenes de tablero (board images) con **superposición temporal** (temporal overlay): las posiciones antiguas se ven transparentes y las recientes opacas. Divide cada partida en **3 bloques temporales**.

### Configuración de Bloques

```python
BLOCK_CONFIG = [
    (15, 30, "block_01"),   # Semi-movimientos 15-30 (apertura tardía/inicio medio juego)
    (30, 45, "block_02"),   # Semi-movimientos 30-45 (medio juego)
    (45, 60, "block_03"),   # Semi-movimientos 45-60 (medio juego tardío/final)
]
```

> **Semi-movimiento (halfmove)**: Un turno de un solo color. Movimiento 15 de blancas = halfmove 29.

### Algoritmo de Temporal Overlay

```python
def overlay_temporal_sequence(board_sequence, compression_factor=1):
    """
    Superpone secuencia de tableros con transparencia gradual.
    
    Para N posiciones de tablero:
      - Posición 1 (más antigua): alpha ≈ 30% (casi transparente)
      - Posición N (más reciente): alpha = 100% (totalmente opaco)
      
    Resultado: Una sola imagen que muestra la "trayectoria" del juego.
    Las piezas que se mueven dejan un "rastro" visual.
    """
    N = len(board_sequence)
    canvas = np.zeros_like(board_sequence[0], dtype=np.float32)
    
    for i, board_img in enumerate(board_sequence):
        alpha = 0.3 + 0.7 * (i / (N - 1))  # 0.3 → 1.0 linealmente
        canvas = canvas * (1 - alpha) + board_img * alpha
    
    return canvas.astype(np.uint8)
```

### Generación por partida

```python
def _generate_board_block(game, output_dir, game_num, block_label, start, end, compression):
    # 1. Reproducir partida hasta move[start]
    # 2. Extraer secuencia de tableros [start..end]
    # 3. Renderizar cada tablero como SVG → PNG (cairosvg)
    # 4. Aplicar overlay temporal
    # 5. Guardar como game_XXXX_block_YY.png
```

### Punto clave para examen

Cada bloque temporal captura una **fase del juego**:
- **Bloque 1**: Transición apertura→medio juego (estructura de peones, desarrollo)
- **Bloque 2**: Medio juego puro (táctica, estrategia)
- **Bloque 3**: Final del medio juego/final (simplificación, técnica)

La superposición temporal permite que la CNN "vea" el flujo de la partida en una sola imagen, en lugar de una secuencia de tableros estáticos.

---

## 7. Fase 5: Orquestación del Pipeline

**Script**: `labs/src/pipeline_stylometry_blocks.py`

### ¿Qué hace?

Orquesta todo el pipeline desde el PGN masivo hasta las imágenes generadas. Es el **punto de entrada principal**.

```python
class ChessStylometryPipelineV002:
    def __init__(self, massive_pgn, output_base, event_type, 
                 num_players=20, games_per_player=30,
                 move_start=15, move_end=30, num_blocks=3):
```

### Fases de Ejecución

```
┌─────────────────────────────────────────────────┐
│ FASE 0: Detección de datos existentes           │
│   · ¿Existen ya player_pgns/ para este evento?  │
│   · Si sí: reutilizar PGNs, limpiar imágenes    │
│   · Si no: proceder a Fase 1                     │
└─────────────────────┬───────────────────────────┘
                      ▼
┌─────────────────────────────────────────────────┐
│ FASE 1: Extracción de jugadores                  │
│   · ParallelPlayerExtractorByEvent()             │
│   · Guarda PGNs en events/{evento}/player_pgns/  │
└─────────────────────┬───────────────────────────┘
                      ▼
┌─────────────────────────────────────────────────┐
│ FASE 2: Generación de imágenes (paralelizado)    │
│   · Para cada jugador (en paralelo):             │
│     - generate_images_blocks() → board_images/   │
│     - Heatmaps integrados en el mismo proceso    │
│   · 6 imágenes/partida (3 board + 3 heatmap)    │
└─────────────────────────────────────────────────┘
```

### Cálculo Automático de Bloques

```python
# move_start=15, move_end=30, num_blocks=3
start_halfmove = (15 * 2) - 1 = 29
end_halfmove = 30 * 2 = 60
total_halfmoves = 60 - 29 + 1 = 32
halfmoves_per_block = 32 // 3 ≈ 10

# Resultado:
block_01: halfmoves 29-39  (moves ~15-20)
block_02: halfmoves 39-49  (moves ~20-25)
block_03: halfmoves 49-59  (moves ~25-30)
```

### Uso desde línea de comandos

```bash
python pipeline_stylometry_blocks.py \
  --pgn-file dataset/generated/lichess_db.pgn \
  --output-dir output/ \
  --event-type "Rated Blitz game" \
  --num-players 100 \
  --games-per-player 40 \
  --move-start 15 \
  --move-end 30 \
  --num-blocks 3
```

---

## 8. Fase 6: Exportación de Embeddings

**Script**: `labs/src/embedding_exporter.py`

### ¿Qué hace?

Convierte las imágenes generadas (board + heatmap) en vectores numéricos (embeddings) usando ResNet18 pre-entrenada, y calcula un **centroide** por jugador.

> **NOTA**: Este script usa PyTorch/ResNet18 (embeddings de 1024D). El notebook 101 usa TensorFlow/ResNet50 (embeddings de 256D). Son dos aproximaciones distintas.

### Arquitectura ResNet18

```python
class ResNetFeatureExtractor:
    def __init__(self, image_size=(224,224), device="cpu", weights="imagenet"):
        # ResNet18 pre-entrenada en ImageNet
        # Se elimina la capa de clasificación (fc → Identity)
        # Normalización: ImageNet (mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
```

### Algoritmo de Embedding por Partida

```
Para cada partida con N bloques:

  Bloque 1:  board_img → ResNet18 → board_vec (512D)
             heat_img  → ResNet18 → heat_vec  (512D)
             block_1_vec = concat(board_vec, heat_vec)  → 1024D

  Bloque 2:  board_img → ResNet18 → board_vec (512D)
             heat_img  → ResNet18 → heat_vec  (512D)  
             block_2_vec = concat(board_vec, heat_vec)  → 1024D

  Bloque 3:  (ídem)    → block_3_vec (1024D)

  game_embedding = mean(block_1_vec, block_2_vec, block_3_vec)  → 1024D
  game_embedding = L2_normalize(game_embedding)                  → 1024D (||v||=1)
```

### Cálculo de Centroides

```python
# Para cada jugador con M partidas:
stacked = np.stack([emb_game_1, emb_game_2, ..., emb_game_M])  # (M, 1024)
centroid = stacked.mean(axis=0)                                  # (1024,)
np.save(f"{player}/centroid.npy", centroid)
```

El **centroide** es el "estilo promedio" del jugador: la media de todos sus embeddings de partida.

### Estructura de Salida

```
embedding/
├── manifest.json
│   {
│     "event": "Rated_Blitz_game",
│     "players": 1238,
│     "vectors": {
│       "PlayerName": {"samples": 15, "embedding_dim": 1024, "centroid_path": "..."}
│     }
│   }
└── PlayerName/
    ├── centroid.npy          # Vector promedio (1024D)
    ├── game_0001.npy         # Embedding partida 1
    ├── game_0002.npy         # Embedding partida 2
    └── ...
```

---

## 9. Notebook 101: Red Siamesa con Validación por ELO

### 9.1 Configuración y Selección de Jugadores

El notebook 101 es una versión reducida para validar si la red siamesa puede distinguir jugadores con **gran diferencia de ELO** usando solo 10 jugadores.

```python
SELECTED_PLAYERS = [
    "PegasYucu",         # ELO ~720  (Muy bajo)
    "vladimir_NN",       # ELO ~878  (Bajo)
    "LeeNewport31",      # ELO ~1192 (Bajo-Medio)
    "Aziz303",           # ELO ~1387 (Medio-Bajo)
    "NeverSettleCastle",  # ELO ~1595 (Medio)
    "DeathIsReal",       # ELO ~1703 (Medio-Alto)
    "AleksoSanchees",    # ELO ~1936 (Alto)
    "ivan_tsventukh",    # ELO ~2084 (Alto)
    "chess64chess",      # ELO ~2261 (Muy Alto)
    "Oganian_Miran",     # ELO ~2765 (Elite)
]
```

**Rango total**: 720 → 2765 = **2045 puntos de diferencia**

```python
CONFIG = {
    "event_name": "Rated_Blitz_game",
    "num_blocks": 3,                    # 3 fases temporales por partida
    "image_size": (192, 192),           # Resolución de imágenes
    "players_limit": 10,                # Solo 10 jugadores
    "min_samples_per_player": 4,        # Mínimo 4 partidas por jugador
    "max_games_per_player": 40,         # Máximo 40 partidas
    "validation_fraction": 0.2,         # 20% para validación
    "triplets_per_epoch": 2048,         # Tripletes por época (reducido)
    "val_triplets": 512,                # Tripletes de validación (reducido)
    "batch_size": 16,                   # Tamaño de batch
    "epochs": 50,                       # Más épocas (dataset más pequeño)
    "margin": 0.4,                      # Margen de triplet loss
    "learning_rate": 1e-4,              # Learning rate (Adam)
    "players_whitelist": SELECTED_PLAYERS,
    "embedding_cache_dir": "embedding_validation",
    "seed": 314159,
}
```

### 9.2 Descubrimiento de Muestras

```python
@dataclass(frozen=True)
class GameSample:
    player: str              # Nombre del jugador (e.g., "DeathIsReal")
    game_id: str             # ID de partida (e.g., "0001")
    board_paths: Tuple[str, ...]  # Rutas a 3 imágenes de tablero (una por bloque)
    heat_paths: Tuple[str, ...]   # Rutas a 3 heatmaps (una por bloque)
    event: str               # Tipo de evento
```

La función `discover_game_samples()` recorre la estructura de directorios:

```python
def discover_game_samples(board_dir, heat_dir, block_labels, 
                          players_limit=None, whitelist=None, 
                          min_samples=4, max_games_per_player=None):
    """
    Para cada jugador en board_dir/:
      1. Si hay whitelist → filtrar solo jugadores de la lista
      2. Buscar game IDs via patrón: game_*_block_01.png
      3. Para cada game_id, verificar que EXISTEN los 3 bloques
         tanto en board_images/ como en heatmap_images/
      4. Solo aceptar jugadores con >= min_samples partidas completas
    
    Retorna: {player: [GameSample, ...], ...}, DataFrame con estadísticas
    """
```

**Validación de integridad**: Una partida solo se incluye si tiene las 6 imágenes:
```
board_images/Player/game_0001_block_01.png  ✓
board_images/Player/game_0001_block_02.png  ✓
board_images/Player/game_0001_block_03.png  ✓
heatmap_images/Player/game_0001_block_01.png ✓
heatmap_images/Player/game_0001_block_02.png ✓
heatmap_images/Player/game_0001_block_03.png ✓
```

### 9.3 Split Train/Validation

```python
def split_samples_by_player(samples_by_player, val_fraction, seed):
    """
    Para cada jugador:
      1. Shuffle sus partidas con semilla fija
      2. Reservar val_fraction (20%) para validación
      3. Asegurar que quedan >= 2 partidas para entrenamiento
         (necesario para crear pares positivos en tripletes)
    
    Retorna: train_dict, val_dict, train_flat, val_flat
    """
```

**Restricción crítica**: Cada jugador debe tener **al menos 2 partidas** en train para poder formar tripletes (anchor + positive del mismo jugador).

### 9.4 Pipeline de Datos: Tripletes y tf.data

#### ¿Qué es un triplete?

```
(Anchor, Positive, Negative)
   │          │          │
   ▼          ▼          ▼
 Partida A   Partida B   Partida C
 Jugador X   Jugador X   Jugador Y ≠ X
```

- **Anchor**: Partida de referencia de un jugador
- **Positive**: Otra partida del MISMO jugador (debería estar "cerca" en el espacio de embeddings)
- **Negative**: Partida de un jugador DISTINTO (debería estar "lejos")

#### Generación de Tripletes

```python
class TripletFactory:
    def build(self, num_triplets, seed):
        """
        Para crear cada triplete:
          1. Escoger jugador ancla aleatorio
          2. Escoger 2 partidas aleatorias de ese jugador (anchor, positive)
          3. Escoger jugador negativo aleatorio (≠ ancla)
          4. Escoger 1 partida aleatoria del negativo
        
        Máximo intentos: 20 × num_triplets (por si hay fallos)
        """
```

#### Serialización para tf.data

```python
def serialize_triplets(triplets):
    """Convierte lista de tripletes a diccionario de arrays numpy:
    {
        "anchor_board": array de rutas board (N, 3),  # N tripletes, 3 bloques
        "anchor_heat": array de rutas heat (N, 3),
        "positive_board": ...,
        "positive_heat": ...,
        "negative_board": ...,
        "negative_heat": ...,
    }
    """
```

#### Carga de Imágenes (load_sample)

```python
def _decode_board(path):
    """PNG → float32 [0,1], 3 canales (RGB), resize a 192×192"""
    image = tf.io.read_file(path)
    image = tf.io.decode_png(image, channels=3)
    image = tf.image.resize(image, (192, 192))
    image = tf.image.convert_image_dtype(image, tf.float32)
    return image  # Shape: (192, 192, 3)

def _decode_heat(path):
    """PNG → float32 [0,1], 1 canal (grayscale), resize a 192×192"""
    image = tf.io.read_file(path)
    image = tf.io.decode_png(image, channels=1)
    image = tf.image.resize(image, (192, 192))
    image = tf.image.convert_image_dtype(image, tf.float32)
    return image  # Shape: (192, 192, 1)

def load_sample(board_paths, heat_paths):
    """
    Carga los 3 bloques de board y heat para una partida.
    
    board_paths: tensor (3,) con rutas a 3 PNGs de tablero
    heat_paths: tensor (3,) con rutas a 3 PNGs de heatmap
    
    Retorna:
      board: (3, 192, 192, 3)  → 3 bloques × RGB
      heat:  (3, 192, 192, 1)  → 3 bloques × Grayscale
    """
    board = tf.map_fn(_decode_board, board_paths, ...)  # Paralelo, 3 imágenes
    heat = tf.map_fn(_decode_heat, heat_paths, ...)     # Paralelo, 3 imágenes
    return board, heat
```

#### Pipeline tf.data

```python
def build_tf_dataset(triplet_dict, batch_size, shuffle, description):
    ds = tf.data.Dataset.from_tensor_slices(triplet_dict)
    if shuffle:
        ds = ds.shuffle(buffer_size=len(data), seed=SEED)
    
    def mapper(sample):
        # Cargar las 6 imágenes del triplete
        anchor_board, anchor_heat = load_sample(sample["anchor_board"], sample["anchor_heat"])
        positive_board, positive_heat = load_sample(...)
        negative_board, negative_heat = load_sample(...)
        
        features = (anchor_board, anchor_heat,
                    positive_board, positive_heat,
                    negative_board, negative_heat)
        dummy_label = tf.zeros((1,))  # No se usa (la loss es custom)
        return features, dummy_label
    
    ds = ds.map(mapper, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(AUTOTUNE)
    return ds
```

### 9.5 Arquitectura del Modelo de Embeddings

Este es el **corazón del sistema**: convierte una partida (6 imágenes) en un vector de 256 dimensiones.

```
┌───────────────────────────────────────────────────────────────────────┐
│                    EMBEDDING MODEL (256D output)                      │
│                                                                       │
│  Inputs:                                                              │
│    board_sequence: (batch, 3, 192, 192, 3)  → 3 bloques RGB          │
│    heat_sequence:  (batch, 3, 192, 192, 1)  → 3 bloques Grayscale    │
│                                                                       │
│  ┌─────────────────────────────┐  ┌─────────────────────────────┐     │
│  │ Board Encoder (por bloque)  │  │ Heat Encoder (por bloque)   │     │
│  │                             │  │                             │     │
│  │ ResNet50 (ImageNet, frozen) │  │ Conv2D(32, 3, relu)        │     │
│  │ GlobalAveragePooling2D      │  │ MaxPooling2D(2)            │     │
│  │ Dense(512, gelu)            │  │ Conv2D(64, 3, relu)        │     │
│  │ Dropout(0.2)                │  │ GlobalAveragePooling2D     │     │
│  │                             │  │ Dense(256, gelu)           │     │
│  │ Output: (batch, 3, 512)     │  │ Output: (batch, 3, 256)   │     │
│  └──────────────┬──────────────┘  └──────────────┬──────────────┘     │
│                 │                                 │                    │
│                 └────────────┬────────────────────┘                    │
│                              ▼                                        │
│                 ┌──────────────────────────┐                          │
│                 │ Concatenate              │                          │
│                 │ (batch, 3, 512+256=768)  │                          │
│                 └────────────┬─────────────┘                          │
│                              ▼                                        │
│                 ┌──────────────────────────┐                          │
│                 │ LayerNormalization       │                          │
│                 │ TimeDistributed          │                          │
│                 │   Dense(256, gelu)       │                          │
│                 │ (batch, 3, 256)          │                          │
│                 └────────────┬─────────────┘                          │
│                              ▼                                        │
│                 ┌──────────────────────────┐                          │
│                 │ Masking(mask_value=0.0)  │  ← Ignora bloques vacíos│
│                 └────────────┬─────────────┘                          │
│                              ▼                                        │
│                 ┌──────────────────────────┐                          │
│                 │ Bidirectional GRU(256)   │  ← Captura dependencias  │
│                 │   return_sequences=True  │    temporales entre      │
│                 │ (batch, 3, 512)          │    bloques                │
│                 └────────────┬─────────────┘                          │
│                              ▼                                        │
│                 ┌──────────────────────────┐                          │
│                 │ GlobalAveragePooling1D   │  ← Pool temporal         │
│                 │ (batch, 512)             │                          │
│                 └────────────┬─────────────┘                          │
│                              ▼                                        │
│                 ┌──────────────────────────┐                          │
│                 │ Dense(512, gelu)         │                          │
│                 │ Dropout(0.3)             │                          │
│                 │ Dense(256, gelu)         │                          │
│                 └────────────┬─────────────┘                          │
│                              ▼                                        │
│                 ┌──────────────────────────┐                          │
│                 │ L2 Normalization         │  ← ||embedding|| = 1     │
│                 │ (batch, 256)             │                          │
│                 └──────────────────────────┘                          │
│                                                                       │
│  Output: embedding vector (batch, 256) con norma L2 = 1              │
└───────────────────────────────────────────────────────────────────────┘
```

#### ¿Por qué esta arquitectura?

| Componente | Justificación |
|------------|---------------|
| **ResNet50 frozen** | Transfer learning: aprovecha features visuales aprendidas en ImageNet sin overfitting |
| **Heat Encoder ligero** | Los heatmaps son simples (grayscale, 8×8 upscaled), no necesitan ResNet50 |
| **Concatenación board+heat** | Fusión multimodal: combina QUÉ posiciones + CÓMO piensa |
| **Bidirectional GRU** | Captura relaciones temporales entre bloques (apertura↔medio juego↔final) |
| **L2 Normalization** | Embeddings en la hiperesfera unitaria → distancias comparables |

#### Parámetros del Modelo

```
Total:          26,054,144 (99.39 MB)
Entrenables:     2,466,432 (9.41 MB)    ← Solo estas capas se actualizan
No entrenables: 23,587,712 (89.98 MB)   ← ResNet50 frozen
```

### 9.6 Red Siamesa y Triplet Loss

#### Estructura de la Red Siamesa

```python
# Tres ramas que COMPARTEN el mismo embedding_model
anchor_embedding   = embedding_model([anchor_board, anchor_heat])     # (batch, 256)
positive_embedding = embedding_model([positive_board, positive_heat]) # (batch, 256)
negative_embedding = embedding_model([negative_board, negative_heat]) # (batch, 256)
```

Las tres ramas usan **exactamente el mismo modelo** (pesos compartidos). Esto es lo que hace que sea "siamesa": la misma red procesa anchor, positive y negative.

#### DistanceLayer

```python
class DistanceLayer(layers.Layer):
    def call(self, anchor, positive, negative):
        # Distancia anchor-positive (debería ser PEQUEÑA)
        ap_distance = tf.reduce_sum(tf.square(anchor - positive), axis=1)
        
        # Distancia anchor-negative (debería ser GRANDE)
        an_distance = tf.reduce_sum(tf.square(anchor - negative), axis=1)
        
        return ap_distance, an_distance
```

#### Triplet Loss

```
L = max(d(anchor, positive) - d(anchor, negative) + margin, 0)
```

```python
class SiameseModel(Model):
    def _compute_loss(self, data):
        siamese_inputs, _ = data
        ap_distance, an_distance = self.siamese_network(siamese_inputs, training=True)
        loss = tf.maximum(ap_distance - an_distance + self.margin, 0.0)
        return tf.reduce_mean(loss)
```

**Interpretación**:
- Si `d(A,P) < d(A,N) - margin` → Loss = 0 (triplete ya satisfecho)
- Si `d(A,P) > d(A,N) - margin` → Loss > 0 (necesita acercar A↔P y/o alejar A↔N)

**Margin = 0.4**: Fuerza una separación mínima de 0.4 entre distancia positiva y negativa.

#### Custom Training Loop

```python
def train_step(self, data):
    with tf.GradientTape() as tape:
        loss = self._compute_loss(data)
    
    # Solo actualiza pesos entrenables (NO ResNet50 frozen)
    gradients = tape.gradient(loss, self.siamese_network.trainable_weights)
    self.optimizer.apply_gradients(zip(gradients, self.siamese_network.trainable_weights))
    
    self.loss_tracker.update_state(loss)
    return {"loss": self.loss_tracker.result()}
```

### 9.7 Entrenamiento

```python
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)

callbacks = [
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,       # Reduce LR a la mitad si no mejora
        patience=20,       # Espera 20 épocas sin mejora
        min_lr=1e-6,       # LR mínimo
    ),
]

history = siamese_model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=50,             # 50 épocas (dataset reducido)
    callbacks=callbacks,
)
```

**Parámetros de entrenamiento (notebook 101)**:
- **Batch size**: 16 tripletes por batch
- **Tripletes/época**: 2048 train + 512 val
- **Batches/época**: ~128 train + ~32 val
- **Optimizer**: Adam (lr=1e-4)

### 9.8 Evaluación de Distancias

Tras el entrenamiento, se evalúa la **separabilidad** de los embeddings:

```python
# Para cada par de muestras de validación:
for i, j in all_pairs(val_embeddings):
    dist = ||emb_i - emb_j||₂  (distancia euclidiana)
    
    if player_i == player_j:
        positive_distances.append(dist)   # Intra-player
    else:
        negative_distances.append(dist)   # Inter-player
```

**Resultado esperado (si la red funciona)**:
```
intra-player (positivas): distancias pequeñas (mismo estilo)
inter-player (negativas): distancias grandes (estilos distintos)

Histograma:
  ████████ intra-player (izquierda)
              ████████████ inter-player (derecha)
            ↑
        Separación = red funciona
```

### 9.9 Verificación de Embeddings

La celda de verificación (nueva en notebook 101) realiza:

1. **Selección aleatoria**: Elige jugador y partida al azar
2. **Visualización**: Muestra las 6 imágenes originales (3 board + 3 heat)
3. **Generación de embedding**: Pasa las imágenes por el modelo → vector 256D
4. **Consistencia**: Verifica que la misma entrada produce el mismo embedding (determinismo)
5. **Visualización del vector**: Barplot del embedding 256D
6. **Comparación entre jugadores**: Distancias L2 entre todos los pares de jugadores con sus ELOs

> **NOTA IMPORTANTE**: No es posible "reconstruir" la imagen desde el embedding porque el modelo es solo un **encoder** (no un autoencoder). Lo que se verifica es que el embedding es **determinista** (mismo input → mismo output) y **discriminativo** (jugadores distintos → embeddings distintos).

### 9.10 Exportación de Centroides

```python
def export_embedding_cache(samples, model, output_dir):
    """
    Para cada muestra de entrenamiento:
      1. Generar embedding con el modelo
      2. Guardar como {player}/{game_id}.npy
    
    Para cada jugador:
      3. Centroide = media de todos sus embeddings
      4. Guardar como {player}/centroid.npy
    
    5. Generar manifest.json con metadatos
    """
```

El **centroide** es el embedding "prototipo" de cada jugador. Para identificar una nueva partida:

```
nueva_partida → embedding_model → embedding_nuevo (256D)

Para cada jugador:
    dist = ||embedding_nuevo - centroide_jugador||₂

jugador_identificado = argmin(dist)
```

---

## 10. Conceptos Clave para el Examen

### ¿Qué es una Red Siamesa?

Red neuronal con **ramas que comparten pesos**. No clasifica directamente, sino que aprende una **función de distancia** entre pares de entradas.

### ¿Qué es Triplet Loss?

Función de pérdida que opera sobre tripletes (Anchor, Positive, Negative):
```
L = max(d(A,P) - d(A,N) + margin, 0)
```
Objetivo: `d(A,P) + margin < d(A,N)` (positivos más cerca que negativos, con margen de seguridad).

### ¿Qué es un Embedding?

Representación vectorial compacta de una entrada compleja. En este proyecto: 6 imágenes → 1 vector de 256 dimensiones.

### ¿Qué es un Centroide?

Media de todos los embeddings de un jugador. Representa su "estilo promedio" en el espacio latente.

### ¿Qué es Transfer Learning?

Usar una red pre-entrenada (ResNet50 en ImageNet) como extractor de features, sin re-entrenar sus capas. Solo se entrenan las capas superiores.

### ¿Por qué Board + Heatmap?

| Modalidad | Captura | Ejemplo |
|-----------|---------|---------|
| **Board Image** | Posiciones alcanzadas, estructura, patrones tácticos | "Este jugador siempre tiene peones avanzados" |
| **Heatmap** | Patrón temporal de reflexión, zonas de concentración | "Este jugador piensa más en el flanco de rey" |

La **fusión** de ambas crea un "fingerprint" estilométrico más rico.

### ¿Por qué Bloques Temporales?

Las partidas de ajedrez tienen fases con estilos diferentes. Al dividir en 3 bloques:
- **Bloque 1**: Apertura/Inicio → Estilo de desarrollo
- **Bloque 2**: Medio juego → Estilo táctico/estratégico  
- **Bloque 3**: Final → Estilo de simplificación

La GRU Bidireccional captura las **transiciones** entre estas fases.

### ¿Por qué L2 Normalization?

Proyecta todos los embeddings a la **hiperesfera unitaria** (||v|| = 1). Esto:
- Hace las distancias comparables entre distintas escalas
- Evita que vectores de mayor magnitud dominen
- Simplifica el cálculo de similitud (distancia euclidiana ∝ coseno)

### Flujo de Identificación (Inferencia)

```
1. Nueva partida desconocida
      ↓
2. Generar 6 imágenes (3 board + 3 heat)
      ↓
3. Pasar por embedding_model → vector 256D
      ↓
4. Calcular distancia L2 a cada centroide de jugador conocido
      ↓
5. El jugador con menor distancia = identificación
```

### Métricas de Éxito

- **Intra-player distance**: Distancia media entre embeddings del mismo jugador → debería ser **baja**
- **Inter-player distance**: Distancia media entre embeddings de distintos jugadores → debería ser **alta**
- **Ratio inter/intra**: Cuanto mayor, mejor separación
- **Loss convergencia**: La triplet loss debería decrecer progresivamente

---

*Documento generado como guía de estudio para el TFE de Chess Stylometry.*
