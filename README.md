# Chess Stylometry - TFE

**Trabajo de Fin de Estudios · Ingeniería de Telecomunicación (UPCT)**

Identificación de jugadores de ajedrez mediante estilometría visual y aprendizaje métrico.

---

## Estado actual (Notebook 104)

El flujo principal está en **`labs/notebooks/104_chess_siamese_robust.ipynb`**.

Este notebook corrige los problemas observados en versiones previas:

- señal centrada en el **jugador objetivo** (no mezcla indiscriminada de jugadas del rival),
- normalización de perspectiva por color (alineación white/black),
- preprocesado correcto para ResNet50,
- entrenamiento métrico estable con **batch-hard triplet + CE auxiliar**,
- validación hold-out estricta con métricas interpretables (Top-1/Top-3 por centroide, kNN, margen intra/inter).

---

## Estructura relevante

- `labs/notebooks/104_chess_siamese_robust.ipynb`: pipeline completo train/val/hold-out + evaluación.
- `labs/notebooks/103_chess_siamese_copy.ipynb`: versión anterior (referencia histórica).
- `labs/src/`: utilidades de extracción/procesado de datos e imágenes.
- `main.py`: CLI para exportación de embeddings (`generate-embeddings`).

---

## Ejecución recomendada

1. Montar dataset comprimido en `/pgn_data` con `index.db` y `players/*.pgn.zst`.
2. Abrir Jupyter y ejecutar `104_chess_siamese_robust.ipynb` por celdas.
3. Revisar métricas hold-out finales y matriz de confusión antes de comparar benchmarks.

> El entrenamiento está configurado para corridas largas (hasta 2000 épocas) con early stopping tardío, pensado para análisis serio de convergencia.

---

## Exportación rápida de embeddings (CLI)

Cuando ya existen PNGs de tablero + heatmap:

```bash
python main.py generate-embeddings \
  --event-dir labs/events/<event_name> \
  --device cpu \
  --weights imagenet
```

Esto exporta vectores `.npy` por partida, centroides por jugador y `manifest.json`.

---

## Autor

**André Yermak Naumenko**  
Universidad Politécnica de Cartagena  
Año 2026
