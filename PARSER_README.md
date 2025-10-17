# Parser de Partidas de Ajedrez a Imágenes

Script para convertir partidas de ajedrez desde archivos PGN a imágenes con codificación visual temporal, basado en el notebook `chess_cnn_visual_temporal.ipynb`.

**⚠️ IMPORTANTE**: El script debe ejecutarse desde el directorio `labs/`.

## Características

- **Codificación temporal visual**: Las jugadas más antiguas se muestran más oscuras, las recientes más brillantes
- **Sin banda de metadatos**: Solo el tablero de ajedrez puro
- **Procesamiento en lote**: Procesa múltiples archivos PGN automáticamente
- **Compresión configurable**: Controla el tamaño de salida con factores de 1x a 8x
- **Rango de movimientos flexible**: Especifica qué parte de la partida visualizar

## Instalación

El script requiere las siguientes dependencias (ya instaladas en el venv):

```bash
pip install python-chess opencv-python numpy cairosvg
```

## Uso desde línea de comandos

### Sintaxis básica

**IMPORTANTE**: Ejecutar desde el directorio `labs/`

```bash
cd labs/
python parse_games_to_images.py \
    --start-move 15 \
    --end-move 23 \
    --compression-factor 2
```

### Parámetros

| Parámetro | Requerido | Default | Descripción |
|-----------|-----------|---------|-------------|
| `--start-move` | ✓ | - | Movimiento inicial (más antiguo) |
| `--end-move` | ✓ | - | Movimiento final (más reciente) |
| `--compression-factor` | ✗ | 2 | Factor de compresión (1, 2, 4, 8) |
| `--pgn-dir` | ✗ | `dataset/testpgns` | Directorio con archivos .pgn |
| `--output-dir` | ✗ | `output/parsed_games` | Directorio de salida |

### Ejemplos de uso

**Ejemplo 1: Configuración básica (movimientos 15-23, compresión 2x)**
```bash
cd labs/
python parse_games_to_images.py --start-move 15 --end-move 23
```

**Ejemplo 2: Máxima calidad, sin compresión**
```bash
cd labs/
python parse_games_to_images.py \
    --start-move 10 \
    --end-move 20 \
    --compression-factor 1 \
    --output-dir output/high_quality
```

**Ejemplo 3: Alta compresión para eficiencia**
```bash
cd labs/
python parse_games_to_images.py \
    --start-move 5 \
    --end-move 15 \
    --compression-factor 4 \
    --output-dir output/compressed
```

**Ejemplo 4: Rango amplio de movimientos**
```bash
cd labs/
python parse_games_to_images.py \
    --start-move 1 \
    --end-move 30 \
    --compression-factor 2 \
    --output-dir output/full_game
```

## Uso desde Python

Puedes importar y usar el script desde otro archivo Python (debe estar en el directorio `labs/` o importar correctamente):

```python
from pathlib import Path
from parse_games_to_images import main as parse_games

parse_games(
    pgn_dir=Path("dataset/testpgns"),
    output_dir=Path("output/my_images"),
    start_move=15,
    end_move=23,
    compression_factor=2
)
```

Ver `labs/ejemplo_uso_parser.py` para más ejemplos.

## Estructura de archivos PGN

El script espera archivos `.pgn` en el directorio especificado:

```
dataset/testpgns/
├── Antunesl.pgn      (13 partidas)
├── Howell.pgn        (13 partidas)
├── Izsak.pgn         (10 partidas)
├── Kantsler.pgn      (11 partidas)
├── Magem.pgn         (13 partidas)
├── Moroz.pgn         (12 partidas)
├── Narciso.pgn       (9 partidas)
├── Olafsson.pgn      (13 partidas)
├── Saldano.pgn       (12 partidas)
└── Zhigalko.pgn      (9 partidas)
```

Total: ~90-114 partidas (algunas pueden tener menos movimientos que el rango solicitado)

## Formato de salida

Las imágenes se generan con el siguiente formato de nombre:

```
{nombre_archivo}_game{numero}.png
```

Ejemplos:
- `Antunesl_game01.png`
- `Howell_game05.png`
- `Zhigalko_game09.png`

## Tamaños de imagen según factor de compresión

| Factor | Tamaño | Reducción | Memoria | Piezas reconocibles |
|--------|--------|-----------|---------|---------------------|
| 1x | 400x400 | 0% | ~468 KB | ✓ SÍ |
| 2x | 200x200 | 75% | ~117 KB | ✓ SÍ |
| 4x | 100x100 | 93.8% | ~29 KB | ⚠ PARCIAL |
| 8x | 50x50 | 98.4% | ~7 KB | ✗ NO |

**Recomendación**: Usa factor 2x para balance entre calidad y eficiencia.

## Codificación temporal

El script aplica un gradiente de intensidad a las jugadas:

- **Jugada más reciente** (posición final): Intensidad 100% (brillante)
- **Jugada más antigua** (posición inicial): Intensidad 30% (oscuro)
- **Jugadas intermedias**: Gradiente suave entre 30% y 100%

Esto permite que una CNN identifique visualmente la antigüedad de cada jugada.

## Manejo de errores

El script continúa procesando aunque algunas partidas fallen:

- ✓ Partidas procesadas exitosamente se marcan con ✓
- ✗ Partidas con errores se marcan con ✗ y se reportan
- Al final muestra un resumen con el total procesado

Errores comunes:
- Partida tiene menos movimientos que el rango solicitado
- PGN malformado
- Problemas de codificación UTF-8

## Ejemplo de salida

```
======================================================================
PROCESANDO PARTIDAS DE AJEDREZ
======================================================================
Directorio PGN: dataset/testpgns
Directorio salida: output/parsed_games
Rango de movimientos: 15-23
Factor de compresión: 2x
Archivos PGN encontrados: 10
======================================================================

📁 Procesando: Antunesl.pgn
   ------------------------------------------------------------------
✓ Antunesl_game01.png (200x200)
✓ Antunesl_game02.png (200x200)
...
   ------------------------------------------------------------------
   Partidas procesadas: 13

======================================================================
RESUMEN FINAL
======================================================================
Total de archivos PGN: 10
Total de partidas procesadas: 114
Imágenes generadas en: output/parsed_games
======================================================================
```

## Diferencias con el notebook original

Este script difiere del `chess_cnn_visual_temporal.ipynb` en:

1. **Sin banda de metadatos**: Solo tablero de ajedrez, sin columna lateral
2. **Procesamiento en lote**: Procesa múltiples archivos/partidas automáticamente
3. **CLI amigable**: Parámetros por línea de comandos o importación Python
4. **Manejo robusto de errores**: Continúa procesando aunque fallen algunas partidas
5. **Salida organizada**: Nombres de archivo descriptivos con jugador y número

## Notas técnicas

- Cada tablero se renderiza a 400x400 píxeles base antes de compresión
- SVG → PNG usando cairosvg para máxima calidad visual
- Transparencia implementada mediante multiplicación de intensidad
- Acumulación usando `np.maximum()` para evitar sobreescritura
- Formato de salida: PNG RGB (3 canales, uint8)
