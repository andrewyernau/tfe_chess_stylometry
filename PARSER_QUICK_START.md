# Quick Start - Parser de Partidas

## Ubicación del script

El script está en `labs/parse_games_to_images.py`. Ejecuta todos los comandos desde el directorio `labs/`.

## Uso más simple

```bash
# Ir al directorio labs
cd labs/

# Activar entorno virtual
source ../venv/bin/activate

# Ejecutar con parámetros mínimos
python parse_games_to_images.py --start-move 15 --end-move 23
```

Esto procesará todos los PGN en `dataset/testpgns/` y generará ~114 imágenes en `output/parsed_games/`.

## Desde otro script Python

```python
from pathlib import Path
from parse_games_to_images import main as parse_games

# Llamada simple
parse_games(
    pgn_dir=Path("dataset/testpgns"),
    output_dir=Path("mi_carpeta_salida"),
    start_move=15,      # Posición inicial
    end_move=23,        # Posición final
    compression_factor=2  # 2x compresión (200x200 px)
)
```

## Parámetros importantes

- `start_move` y `end_move`: Rango de movimientos a visualizar (REQUERIDOS)
- `compression_factor`: 1 (máxima calidad), 2 (recomendado), 4 u 8 (menor calidad)

## Resultado

Genera una imagen PNG por cada partida en cada archivo .pgn:
- Nombre: `{jugador}_game{numero}.png`
- Tamaño: Depende del factor de compresión
- Efecto: Jugadas antiguas oscuras, recientes brillantes
