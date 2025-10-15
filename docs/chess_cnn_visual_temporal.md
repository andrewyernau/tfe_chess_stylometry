# Codificación Visual Temporal para Chess CNN v3.0

## Resumen ejecutivo

Esta implementación prioriza la **interpretabilidad visual** para redes neuronales convolucionales, permitiendo que la CNN reconozca piezas de ajedrez visualmente mientras codifica información temporal mediante intensidad de color y metadata de estado del juego.

**Versión**: 3.0  
**Fecha**: Septiembre 2025 
**Implementación**: `chess_cnn_visual_temporal.ipynb`

---

## 1. Filosofía y motivación

### 1.1 Problema con enfoques abstractos

Las versiones anteriores (v1.0, v2.0) usaban codificación abstracta:
- Piezas representadas como valores numéricos
- Alta compresión sacrificaba reconocimiento visual
- La CNN debía "aprender" qué es cada número

**Limitación**: Una CNN está optimizada para procesar **imágenes visuales**, no abstracciones numéricas.

### 1.2 Solución: Codificación visual

Esta versión usa tableros **renderizados visualmente**:
- Piezas dibujadas como en un tablero real
- La CNN "ve" las piezas directamente
- Compresión controlada para mantener reconocimiento
- Información temporal codificada en intensidad de brillo

### 1.3 Ventajas clave

✅ **Visual**: Piezas reconocibles por la CNN  
✅ **Transfer learning**: Compatible con CNNs preentrenadas  
✅ **Temporal**: Intensidad indica antigüedad del movimiento  
✅ **Metadata**: Estado del juego integrado en la imagen  
✅ **Flexible**: Funciona en color o escala de grises  

---

## 2. Arquitectura del sistema

### 2.1 Pipeline completo

```
┌─────────────────────────────────────────────────────────────┐
│                    PIPELINE COMPLETO                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  PGN File                                                   │
│     ↓                                                       │
│  Parse movimientos 15→23 (9 movimientos)                   │
│     ↓                                                       │
│  Para cada movimiento:                                      │
│  ┌────────────────────────────────────┐                    │
│  │  1. Generar SVG sin coordenadas    │                    │
│  │  2. Convertir SVG → PNG (RGB)      │                    │
│  │  3. Extraer metadata del estado    │                    │
│  └────────────────────────────────────┘                    │
│     ↓                                                       │
│  INVERTIR orden (reciente → antiguo)                        │
│     ↓                                                       │
│  Para cada posición:                                        │
│  ┌────────────────────────────────────┐                    │
│  │  1. Comprimir (÷ factor)           │                    │
│  │  2. Aplicar intensidad temporal    │                    │
│  │  3. Superponer con alpha blending  │                    │
│  └────────────────────────────────────┘                    │
│     ↓                                                       │
│  Agregar banda de metadata (40px)                          │
│     ↓                                                       │
│  Imagen final RGB (H × W × 3)                              │
│     ↓                                                       │
│  [Opcional] Convertir a escala de grises                   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 Orden temporal INVERSO

**Decisión clave**: Los movimientos se procesan del más reciente al más antiguo.

```
Array de entrada:
  [0]: Movimiento 23 (MÁS RECIENTE)  → Intensidad 100%
  [1]: Movimiento 22                 → Intensidad ~87%
  [2]: Movimiento 21                 → Intensidad ~75%
  ...
  [7]: Movimiento 16                 → Intensidad ~43%
  [8]: Movimiento 15 (MÁS ANTIGUO)   → Intensidad 30%
```

**Razón**: Lo más importante (estado actual) tiene máxima visibilidad.

---

## 3. Compresión visual controlada

### 3.1 Concepto

La compresión **reduce el tamaño** pero **preserva la reconocibilidad** de las piezas.

```
Tamaño comprimido = Tamaño original ÷ Factor
```

### 3.2 Tabla de compresión

| Factor | Tamaño original | Resultado | Reconocimiento | Uso |
|--------|----------------|-----------|----------------|-----|
| **1**  | 400×400        | 400×400   | Excelente      | Máxima calidad |
| **2**  | 400×400        | 200×200   | Muy bueno      | **Recomendado** |
| **4**  | 400×400        | 100×100   | Aceptable      | Balance |
| **8**  | 400×400        | 50×50     | Limitado       | Mínimo |

### 3.3 Ejemplos visuales

#### Factor 1 (sin compresión)
```
Cada pieza: ~50×50 pixels
Detalles: Corona del rey, forma del caballo visibles
Peso: ~470 KB (PNG)
```

#### Factor 2 (recomendado)
```
Cada pieza: ~25×25 pixels
Detalles: Silueta clara, tipo reconocible
Peso: ~120 KB (PNG)
```

#### Factor 4
```
Cada pieza: ~12×12 pixels
Detalles: Silueta reconocible, algunos detalles perdidos
Peso: ~30 KB (PNG)
```

#### Factor 8 (límite)
```
Cada pieza: ~6×6 pixels
Detalles: Apenas reconocible, solo para contexto general
Peso: ~8 KB (PNG)
```

### 3.4 Método de compresión

Se usa **interpolación de área** (`cv2.INTER_AREA`):

```python
compressed = cv2.resize(image, (new_width, new_height), 
                        interpolation=cv2.INTER_AREA)
```

**Ventajas**:
- Promedia regiones (anti-aliasing natural)
- Preserva mejor las formas de las piezas
- Óptima para reducción de tamaño

---

## 4. Codificación temporal por intensidad

### 4.1 Concepto

La **intensidad de brillo** indica cuán reciente o antiguo es un movimiento.

```
Intensidad = MAX - (índice_temporal / (N-1)) × (MAX - MIN)
```

Donde:
- `MAX = 1.0` (100% brillo) para movimiento reciente
- `MIN = 0.3` (30% brillo) para movimiento antiguo
- `N` = número total de movimientos

### 4.2 Ejemplo con 9 movimientos (15→23)

| Índice | Movimiento | Intensidad | Brillo | Visual |
|--------|------------|------------|--------|--------|
| 0      | 23         | 1.00       | 100%   | Brillante ████████████ |
| 1      | 22         | 0.91       | 91%    | Muy claro ███████████░ |
| 2      | 21         | 0.83       | 83%    | Claro     ██████████░░ |
| 3      | 20         | 0.74       | 74%    | Medio     █████████░░░ |
| 4      | 19         | 0.66       | 66%    | Medio     ████████░░░░ |
| 5      | 18         | 0.57       | 57%    | Oscuro    ███████░░░░░ |
| 6      | 17         | 0.49       | 49%    | Oscuro    ██████░░░░░░ |
| 7      | 16         | 0.40       | 40%    | Muy oscuro█████░░░░░░░ |
| 8      | 15         | 0.30       | 30%    | Mínimo    ████░░░░░░░░ |

### 4.3 Aplicación

```python
def apply_temporal_intensity(image, temporal_index, total_moves):
    intensity = 1.0 - (temporal_index / (total_moves - 1)) * 0.7
    
    # Multiplicar cada pixel por intensidad
    result = (image.astype(np.float32) * intensity).astype(np.uint8)
    
    return result
```

### 4.4 Superposición (alpha blending)

Las imágenes se combinan usando **alpha blending ponderado**:

```python
result = image_reciente
for img_antigua in resto_imagenes:
    weight = calcular_peso(antigüedad)
    result = cv2.addWeighted(result, alpha, img_antigua, 1-alpha, 0)
```

**Efecto**: Movimientos recientes "predominan" visualmente sobre antiguos.

---

## 5. Banda de metadata

### 5.1 Concepto

Una banda vertical de **40 pixels de ancho** codifica el estado del juego visualmente.

```
┌────────────────┬────┐
│                │ M  │
│                │ E  │
│   Tablero      │ T  │
│   400×400      │ A  │
│                │ D  │
│                │ A  │
│                │ T  │
│                │ A  │
└────────────────┴────┘
```

### 5.2 Estructura (8 secciones verticales)

Cada sección ocupa 1/8 de la altura total:

```
┌──────────────────────────────────┐
│  1. Enroque ♔ lado ♔ (blancas)   │ ← Blanco si disponible
├──────────────────────────────────┤
│  2. Enroque ♔ lado ♕ (blancas)   │ ← Blanco si disponible
├──────────────────────────────────┤
│  3. Enroque ♚ lado ♔ (negras)    │ ← Negro si disponible
├──────────────────────────────────┤
│  4. Enroque ♚ lado ♕ (negras)    │ ← Negro si disponible
├──────────────────────────────────┤
│  5. Rey blanco en jaque          │ ← Rojo si en jaque
├──────────────────────────────────┤
│  6. Rey negro en jaque           │ ← Rojo si en jaque
├──────────────────────────────────┤
│  7. Turno actual                 │ ← Blanco/Negro según turno
├──────────────────────────────────┤
│  8. Gradiente temporal           │ ← Brillante→oscuro
└──────────────────────────────────┘
```

### 5.3 Código de colores

| Estado | Color | RGB | Significado |
|--------|-------|-----|-------------|
| Enroque disponible (blancas) | Blanco | (255, 255, 255) | Puede enrocar |
| Enroque disponible (negras) | Negro | (0, 0, 0) | Puede enrocar |
| No disponible | Gris | (128, 128, 128) | No puede enrocar |
| Jaque | Rojo | (255, 0, 0) | Rey en jaque |
| Turno blancas | Blanco | (255, 255, 255) | Mueven blancas |
| Turno negras | Negro | (0, 0, 0) | Mueven negras |

### 5.4 Ejemplo de lectura

```
Si la banda muestra:
  Sección 1: Blanco  → Blancas pueden enrocar corto
  Sección 2: Gris    → Blancas NO pueden enrocar largo
  Sección 3: Negro   → Negras pueden enrocar corto
  Sección 4: Negro   → Negras pueden enrocar largo
  Sección 5: Gris    → Rey blanco NO en jaque
  Sección 6: Rojo    → Rey negro EN JAQUE
  Sección 7: Blanco  → Turno de blancas
  Sección 8: Gradiente → Visualización temporal
```

### 5.5 Uso en CNN

La CNN puede aprender a:
1. **Ignorar** la banda durante procesamiento del tablero
2. **Concatenar** features de ambas partes
3. **Usar** la metadata como entrada auxiliar

Ejemplo con Keras:

```python
# Opción 1: Separar entrada
board_input = Input(shape=(200, 200, 3))
metadata_input = Input(shape=(200, 40, 3))

board_features = CNN_branch(board_input)
metadata_features = Dense_branch(GlobalAveragePooling2D()(metadata_input))

combined = Concatenate()([board_features, metadata_features])
output = Dense(1)(combined)
```

---

## 6. Funciones principales

### 6.1 `board_to_png_array()`

Convierte tablero chess.Board a imagen PNG RGB.

**Entrada**:
- `board`: Objeto chess.Board
- `size`: Tamaño en pixels (default: 400)

**Salida**:
- Array numpy (size, size, 3), dtype=uint8

**Proceso**:
1. Genera SVG sin coordenadas usando `chess.svg.board()`
2. Convierte SVG → PNG con `cairosvg`
3. Carga PNG como array numpy RGB

**Dependencia**: Requiere `cairosvg` instalado.

---

### 6.2 `extract_game_state_metadata()`

Extrae metadata de estado del juego.

**Entrada**:
- `board`: Objeto chess.Board

**Salida**:
- Diccionario con flags booleanos:
  ```python
  {
      'white_kingside_castle': bool,
      'white_queenside_castle': bool,
      'black_kingside_castle': bool,
      'black_queenside_castle': bool,
      'white_in_check': bool,
      'black_in_check': bool,
      'white_to_move': bool
  }
  ```

**Uso**: Información para la banda de metadata.

---

### 6.3 `extract_board_sequence()`

Extrae secuencia de tableros desde PGN en orden inverso.

**Entrada**:
- `pgn_text`: String con PGN
- `start_move`: Movimiento inicial (más antiguo)
- `end_move`: Movimiento final (más reciente)
- `board_size`: Tamaño de renderizado

**Salida**:
- Lista de diccionarios **en orden inverso** (reciente → antiguo):
  ```python
  [
      {
          'board_image': np.ndarray (400, 400, 3),
          'move_number': int,
          'metadata': dict,
          'temporal_index': int  # 0 = más reciente
      },
      ...
  ]
  ```

**Proceso**:
1. Parse PGN
2. Avanza hasta `end_move`
3. Extrae posiciones de `start_move` a `end_move`
4. **INVIERTE** el orden
5. Agrega `temporal_index`

---

### 6.4 `compress_board_image()`

Comprime imagen dividiendo por factor.

**Entrada**:
- `image`: Array numpy (H, W, 3)
- `compression_factor`: Divisor del tamaño

**Salida**:
- Array comprimido (H/factor, W/factor, 3)

**Validación**: Error si resultado < 8×8 pixels.

---

### 6.5 `apply_temporal_intensity()`

Aplica modulación de brillo según posición temporal.

**Entrada**:
- `image`: Array numpy (H, W, 3), uint8
- `temporal_index`: Índice temporal (0 = reciente)
- `total_moves`: Número total de movimientos
- `min_intensity`: Intensidad mínima (default: 0.3)
- `max_intensity`: Intensidad máxima (default: 1.0)

**Salida**:
- Array con intensidad modulada, uint8

**Fórmula**:
```
intensity = max - (index / (total-1)) × (max - min)
result = image × intensity
```

---

### 6.6 `create_metadata_band()`

Genera banda visual con metadata.

**Entrada**:
- `metadata_list`: Lista de diccionarios de metadata
- `height`: Altura de la banda
- `width`: Ancho de la banda (default: 40)

**Salida**:
- Array numpy (height, width, 3), uint8

**Estructura**: 8 secciones verticales codificando estado del juego.

---

### 6.7 `overlay_temporal_sequence()`

Superpone secuencia temporal con alpha blending.

**Entrada**:
- `board_sequence`: Lista de tableros (orden inverso)
- `compression_factor`: Factor de compresión
- `min_intensity`, `max_intensity`: Rango de intensidad
- `add_metadata_band`: Agregar banda de metadata (bool)
- `metadata_band_width`: Ancho de la banda

**Salida**:
- Array numpy con imagen final
  - Con metadata: (H, W+band_width, 3)
  - Sin metadata: (H, W, 3)

**Proceso**:
1. Procesar primera imagen (más reciente)
2. Para cada imagen subsecuente:
   - Comprimir
   - Aplicar intensidad temporal
   - Superponer con alpha blending ponderado
3. Agregar banda de metadata si se solicita

---

### 6.8 `convert_to_grayscale_temporal()`

Convierte a escala de grises preservando metadata en color.

**Entrada**:
- `image_rgb`: Array RGB (H, W, 3)
- `preserve_metadata`: Mantener banda en color (bool)
- `metadata_band_width`: Ancho de la banda

**Salida**:
- Array numpy (H, W, 3) - mantiene 3 canales

**Proceso**:
1. Separar tablero y metadata
2. Convertir tablero a grayscale
3. Reconvertir grayscale a RGB (valores iguales en R=G=B)
4. Concatenar con metadata en color

---

## 7. Aplicaciones en CNN

### 7.1 Evaluación de posición

**Objetivo**: Predecir ventaja en centipeones.

**Configuración**:
```python
START_MOVE = 10
END_MOVE = 18  # 9 movimientos
COMPRESSION_FACTOR = 2
```

**Arquitectura**:
```python
model = Sequential([
    Conv2D(64, 3, activation='relu', input_shape=(200, 240, 3)),
    MaxPooling2D(2),
    Conv2D(128, 3, activation='relu'),
    MaxPooling2D(2),
    Conv2D(256, 3, activation='relu'),
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='tanh')  # Salida: -1 a +1
])
```

**Dataset**: 1M partidas con evaluación Stockfish.

**Resultado esperado**: MAE < 0.5 peones.

---

### 7.2 Predicción de movimiento

**Objetivo**: Predecir próximo movimiento (clasificación 64×64).

**Configuración**:
```python
START_MOVE = variable
END_MOVE = START_MOVE + 4  # 5 movimientos de contexto
COMPRESSION_FACTOR = 1  # Sin compresión (máxima precisión)
```

**Arquitectura**: U-Net modificada

**Salida**: Mapa de calor 8×8 con probabilidades de destino.

**Dataset**: 5M movimientos de partidas maestras.

**Resultado esperado**: Top-1 accuracy ~50%, Top-3 ~80%.

---

### 7.3 Detección de tácticas

**Objetivo**: Clasificar si hay táctica (mate, tenedor, clavada, etc.)

**Configuración**:
```python
START_MOVE = variable  # Centrado en táctica
END_MOVE = START_MOVE + 3  # 4 movimientos
COMPRESSION_FACTOR = 2
```

**Arquitectura**: EfficientNetB0 con transfer learning

**Dataset**: 200k posiciones (100k tácticas + 100k normales)

**Resultado esperado**: Accuracy > 95%.

---

### 7.4 Transfer learning

Esta representación es compatible con **CNNs preentrenadas** en ImageNet:

```python
from tensorflow.keras.applications import ResNet50

# Cargar ResNet50 preentrenada
base_model = ResNet50(weights='imagenet', include_top=False, 
                      input_shape=(200, 240, 3))

# Congelar capas base
base_model.trainable = False

# Agregar capas custom
model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='tanh')  # Evaluación
])
```

**Ventaja**: La red ya sabe extraer features visuales generales.

---

## 8. Comparativa con versiones anteriores

| Característica | v1.0 | v2.0 | v3.0 (actual) |
|----------------|------|------|---------------|
| **Representación** | Arrays numéricos | Codificación RGBA | Imágenes visuales |
| **Piezas reconocibles** | No | No | **Sí** |
| **Compresión típica** | 8×8 pixels | 100×100 pixels | 200×200 pixels |
| **Reducción datos** | 99% | 93% | 75% |
| **Orden temporal** | Creciente | Acumulativo | **Inverso** |
| **Metadata** | No | En canales G/A | **Banda visual** |
| **Transfer learning** | No compatible | Parcial | **Completamente** |
| **Interpretabilidad** | Baja | Media | **Alta** |
| **Mejor para** | Embedding | Clustering | **Predicción CNN** |

### Cuándo usar cada versión

**v1.0**: No recomendada (obsoleta)

**v2.0**: 
- Datasets muy grandes (>10M ejemplos)
- Clustering y búsqueda por similitud
- Cuando memoria es crítica

**v3.0** (esta versión):
- **Entrenamiento de CNNs** (recomendado)
- Transfer learning
- Cuando importa interpretabilidad
- Evaluación y predicción

---

## 9. Mejores prácticas

### 9.1 Selección de hiperparámetros

| Parámetro | Recomendado | Rango | Impacto |
|-----------|-------------|-------|---------|
| `COMPRESSION_FACTOR` | 2 | 1-4 | Visual: menor = mejor reconocimiento |
| `NUM_MOVES` | 8-9 | 4-16 | Temporal: más = más contexto |
| `MIN_INTENSITY` | 0.3 | 0.1-0.5 | Visual: menor = más contraste |
| `BOARD_SIZE_PIXELS` | 400 | 200-800 | Calidad: mayor = más detalle |

### 9.2 Preprocesamiento

#### Normalización

```python
# Normalizar a [0, 1]
image_normalized = image.astype(np.float32) / 255.0

# O estandarización
mean = np.array([0.485, 0.456, 0.406])  # ImageNet means
std = np.array([0.229, 0.224, 0.225])   # ImageNet stds
image_standardized = (image_normalized - mean) / std
```

#### Aumentación

Permitidas:
- ✅ Flip horizontal (simetría del tablero)
- ✅ Rotación 180° (invertir perspectiva)
- ✅ Ajuste de brillo (±20%)
- ✅ Ajuste de contraste (±20%)

No recomendadas:
- ❌ Rotación arbitraria
- ❌ Crop aleatorio (pierde casillas)
- ❌ Distorsión geométrica

### 9.3 Batch processing

```python
def process_batch_pgns(pgn_files, start_move, end_move, 
                       compression_factor):
    images = []
    labels = []
    
    for pgn_file in pgn_files:
        pgn_text = pgn_file.read_text()
        
        try:
            sequence = extract_board_sequence(pgn_text, start_move, 
                                              end_move)
            image = overlay_temporal_sequence(sequence, compression_factor)
            
            images.append(image)
            labels.append(extract_label(pgn_file))
            
        except Exception as e:
            print(f"Error en {pgn_file}: {e}")
            continue
    
    return np.array(images), np.array(labels)
```

### 9.4 Validación

```python
# Verificar distribución de intensidades
plt.hist(image.flatten(), bins=50)
plt.title('Distribución de intensidades')
plt.show()

# Verificar metadata
metadata_band = image[:, -40:, :]
print(f"Metadata única: {np.unique(metadata_band)}")

# Verificar rango
assert image.min() >= 0 and image.max() <= 255
```

---

## 10. Limitaciones y consideraciones

### 10.1 Dependencia de cairosvg

**Problema**: Requiere `cairosvg` para conversión SVG→PNG.

**Solución**:
```bash
# Linux
pip install cairosvg

# Puede requerir dependencias del sistema
sudo apt-get install libcairo2-dev
```

**Alternativa**: Implementar rendering simple (ya incluido como fallback).

### 10.2 Tamaño de memoria

Con factor de compresión 2 y 9 movimientos:

```
Tamaño por imagen: ~120 KB (PNG comprimido)
Dataset 100k:      ~12 GB
Dataset 1M:        ~120 GB
```

**Mitigación**:
- Usar factor 4 (reduce a ~30 KB/imagen)
- Comprimir PNG al guardar
- Usar formato HDF5 para almacenamiento

### 10.3 Costo computacional de rendering

Generar SVG→PNG es **lento** (~50-100ms por tablero).

**Solución**:
```python
# Cachear posiciones comunes
position_cache = {}

def get_board_image_cached(fen):
    if fen not in position_cache:
        board = chess.Board(fen)
        position_cache[fen] = board_to_png_array(board)
    return position_cache[fen]
```

### 10.4 Superposición puede oscurecer detalles

Con muchos movimientos, piezas antiguas quedan muy oscuras.

**Mitigación**:
- Limitar a 8-12 movimientos
- Ajustar `min_intensity` (ej: 0.4 en vez de 0.3)
- Usar pesos adaptativos en el blending

---

## 11. Extensiones futuras

### 11.1 Metadata extendida

Agregar más información a la banda:
- Material count (diferencia de material)
- Control del centro
- Movilidad de piezas
- Eval del motor (opcional)

### 11.2 Múltiples perspectivas

Generar variantes:
- Desde perspectiva de blancas
- Desde perspectiva de negras
- Vista 3D proyectada

### 11.3 Codificación de variantes

Incluir líneas alternativas sugeridas por el motor:

```
Imagen principal: Línea jugada
Canal alpha:      Mejor línea según engine
```

### 11.4 Atención temporal

Usar mecanismo de atención para ponderar movimientos:

```python
# Aprender pesos temporales
temporal_weights = AttentionLayer()(metadata_embedding)
weighted_images = images * temporal_weights
```

---

## 12. Conclusiones

### 12.1 Resumen

La codificación visual temporal v3.0 ofrece:

✅ **Piezas visualmente reconocibles** para CNNs  
✅ **Información temporal** codificada en intensidad  
✅ **Metadata del juego** integrada en banda visual  
✅ **Compatible** con transfer learning  
✅ **Flexible** en compresión (factor 1-8)  
✅ **Interpretable** por humanos y máquinas  

### 12.2 Resultados esperados

Con configuración recomendada (factor 2, 9 movimientos):
- Tamaño: 200×240×3 (~120 KB)
- Piezas: Claramente reconocibles
- Contexto temporal: 9 movimientos
- Metadata: 7 flags de estado + gradiente

### 12.3 Cuándo usar

**Ideal para**:
- Entrenamiento de CNNs desde cero
- Transfer learning con ResNet/EfficientNet
- Tareas de predicción y clasificación
- Cuando importa interpretabilidad

**No ideal para**:
- Datasets masivos (>5M ejemplos) → usar v2.0
- Búsqueda por similitud → usar v2.0 con embeddings
- Recursos muy limitados → usar v2.0 con factor 8

---

## 13. Referencias

### 13.1 Implementación

- **Notebook**: `labs/notebooks/chess_cnn_visual_temporal.ipynb`
- **Documentación**: Este archivo
- **Tests**: Incluidos en el notebook

### 13.2 Dependencias

```bash
pip install chess numpy opencv-python matplotlib pillow cairosvg
```

### 13.3 Datasets recomendados

- **Lichess Database**: https://database.lichess.org/
- **CCRL**: http://www.computerchess.org.uk/ccrl/
- **Kaggle Chess**: https://www.kaggle.com/datasets?search=chess

---

**Versión**: 3.0  
**Última actualización**: Diciembre 2024  
**Autor**: Chess CNN Team  
**Licencia**: Sigue pautas de AGENTS.md
