# Chess Stylometry - TFE

**Trabajo de Fin de Estudios · Ingeniería de Telecomunicación (UPCT)**

Identificación de jugadores de ajedrez mediante estilometría visual y aprendizaje métrico.

---

## Estado actual (Notebook 104)

El flujo principal está en **`labs/notebooks/`**.

---

## Estructura relevante

- `labs/src/`: utilidades de extracción/procesado de datos e imágenes.
- `main.py`: CLI para exportación de embeddings (`generate-embeddings`).

---

## Ejecución recomendada

1. Montar dataset comprimido en `/pgn_data` con `index.db` y `players/*.pgn.zst`.
2. Abrir Jupyter y ejecutar `120*.ipynb` o `130*.ipynb` para el entrenamiento.

> El entrenamiento está configurado para corridas largas (hasta 2000 épocas) con early stopping tardío, pensado para análisis serio de convergencia.

### Persistencia y reanudación de entrenamiento (Notebook 110)

En `labs/notebooks/110_chess_siamese.ipynb` se guarda estado de entrenamiento en:

- `labs/notebooks/output/events/<event_name>/training_state/history.csv`
- `labs/notebooks/output/events/<event_name>/training_state/checkpoints/metric_trainer_last.weights.h5`
- `labs/notebooks/output/events/<event_name>/training_state/checkpoints/metric_trainer_best.weights.h5`
- `labs/notebooks/output/events/<event_name>/training_state/backup/`

Si se cierra navegador, kernel o contenedor, al volver a ejecutar la celda de entrenamiento el notebook intenta reanudar automáticamente desde `backup/` (y, si no existe backup, desde `metric_trainer_last.weights.h5`).

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
