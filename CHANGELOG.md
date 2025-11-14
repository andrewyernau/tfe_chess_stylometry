# Changelog - Estilometría Conductual de Ajedrez

## [2.0.0] - 2025-10-25

### ✨ Nuevas Características

#### Sistema de Extracción de Partidas
- **`extract_player_games.py`**: Extractor eficiente para archivos PGN masivos (188GB+)
  - Búsqueda por jugador con timeout configurable
  - Buffer I/O optimizado (64KB)
  - Procesamiento streaming sin cargar archivo completo
  - Umbrales mínimo/máximo de partidas por jugador
  - Progreso en tiempo real cada 10,000 partidas

#### Mapas de Calor de Decisión
- **`generate_decision_heatmaps.py`**: Visualización de tiempos de decisión
  - Gradiente de colores: frío (rápido) → cálido (lento)
  - Grid 8x8 para blancas y negras separadamente
  - Cálculo temporal: |T(n+1) - T(n)| - incremento
  - Tiempos relativos (%) o absolutos (segundos)
  - Percentile clipping para normalización

#### Pipeline Integrado
- **`pipeline_stylometry.py`**: Automatización completa de 3 fases
  1. Extracción de partidas por jugador
  2. Generación de imágenes de tablero
  3. Generación de mapas de calor
  - Organización automática de datos para CNN

#### CNN Dual-Channel
- **`notebooks/0002_dual_channel_cnn.ipynb`**: Red Siamese de dos canales
  - Canal 1: Imágenes de tablero (ResNet50)
  - Canal 2: Mapas de calor de decisión (ResNet50)
  - Arquitectura: Embeddings concatenados → combiner → L2 norm
  - Triplet Loss para aprendizaje métrico
  - Validación automática con curvas y métricas

#### Testing y Documentación
- **`test_stylometry.py`**: Suite de tests automatizados (4 tests, 100% pass)
- **`README_STYLOMETRY.md`**: Documentación completa del sistema (8KB)
- **`IMPLEMENTATION_SUMMARY.md`**: Detalles técnicos de implementación (8KB)
- **`QUICK_START.md`**: Guía de inicio rápido en 3 pasos
- **`example_usage.sh`**: Script de ejemplos prácticos

### 🔧 Optimizaciones

- Buffer I/O de 64KB para archivos grandes
- Procesamiento streaming (no carga archivo completo en memoria)
- Timeout por jugador (evita bloqueos)
- GPU memory growth habilitado
- Batch prefetch para eficiencia
- Garbage collection automático

### 📊 Formato de Datos

#### Mapas de Calor
```
┌─────────────────────┐
│  BLANCAS (8x8)     │  ← Movimientos impares
├─────────────────────┤
│  NEGRAS (8x8)      │  ← Movimientos pares
└─────────────────────┘

Codificación:
🔵 Azul = Rápido (< 5%)
🟢 Verde = Normal (5-15%)
🟡 Amarillo = Lento (15-30%)
🔴 Rojo = Muy lento (> 30%)
```

#### Estructura de Salida
```
output/
├── player_pgns/       # PGNs extraídos
├── board_images/      # Imágenes de tablero
└── heatmap_images/    # Mapas de calor
```

### 📈 Métricas de Calidad

- ✅ Tests: 4/4 pasados (100%)
- ✅ Compatibilidad: Python 3.10+
- ✅ GPU: TensorFlow 2.20.0 con CUDA
- ✅ Documentación: 25KB de docs técnicas

### 🎯 Casos de Uso

1. **Identificación de jugador**: ¿Partida de Magnus o Hikaru?
2. **Análisis temporal**: Patrones de tiempo de decisión
3. **Detección de anomalías**: Partidas con comportamiento inusual
4. **Clustering**: Agrupar jugadores por estilo similar

### 🔄 Compatibilidad

- Mantiene compatibilidad con sistema base (v1.0)
- Notebook original `0001_siamese_nn.ipynb` sigue funcionando
- Scripts legacy no afectados

### 📝 Notas de Migración

Para usuarios del sistema v1.0:
- Los scripts originales siguen disponibles
- El nuevo sistema es complementario, no reemplaza
- Puede usarse el pipeline antiguo o el nuevo según necesidad

---

## [1.0.0] - 2025-10-24

### Características Iniciales

- Sistema base de estilometría de ajedrez
- `parse_games_to_images.py`: Generación de imágenes de tablero
- `0001_siamese_nn.ipynb`: Red Siamese single-channel
- Transparencia temporal en imágenes de tablero
- Procesamiento básico de archivos PGN

---

**Versión actual:** 2.0.0  
**Última actualización:** 2025-10-25
