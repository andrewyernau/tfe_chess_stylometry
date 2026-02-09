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

---

## Inicio Rápido

```bash
cd labs

# Procesar datos y generar imágenes
python pipeline_stylometry.py \
  --pgn-file dataset/generated/lichess_db.pgn \
  --max-games 50 --timeout 600

# Entrenar CNN
jupyter notebook
```
### Exporting embeddings

Once the board and heatmap PNGs exist under `labs/events/<event_name>`, run:

```bash
python main.py generate-embeddings \
  --event-dir labs/events/My_Event \
  --device cpu \
  --weights imagenet
```

This command stores one `.npy` vector per game, computes per-player centroids, and writes a manifest alongside the generated embeddings.

---

## Autor

**André Yermak Naumenko**  
Universidad Politécnica de Cartagena  
Grado en Ingeniería Telemática  
Año 2026

---

**Última actualización**: Febrero 2026

