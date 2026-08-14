# Cumplimiento

### **1. Red Siamesa (2 partes)**
**Archivo:** `labs/notebooks/100_chess_siamese.ipynb`
- **Línea 886-889:** Clase `DistanceLayer` que calcula distancias entre anchor-positive y anchor-negative
- **Línea 892-918:** Clase `SiameseModel` que implementa el modelo siamés con triplet loss
- **Línea 1140-1142:** Construcción de las 3 ramas (anchor, positive, negative) compartiendo pesos
- **Línea 1145:** Cálculo de distancias `ap_distance` y `an_distance`

---

### **2. Distinguir entre 100 jugadores**
**Archivo:** `labs/notebooks/100_chess_siamese.ipynb`
- **Línea 122:** `"players_limit": 100` - Configuración para limitar a 100 jugadores
- **Línea 314:** Output: `"Jugadores válidos: 100"`
- **Línea 402-403:** Filtrado de los top 100 jugadores con más partidas
- **Línea 411-418:** Función `discover_game_samples()` que descubre muestras de jugadores

---

### **3. Sistema Ancla-Positivo-Negativo (Triplet Loss)**
**Archivo:** `labs/notebooks/100_chess_siamese.ipynb`
- **Línea 695-724:** Clase `TripletFactory` que genera tripletes
- **Línea 711:** Selección del jugador ancla: `anchor_player = rng.choice(self.players)`
- **Línea 715:** Selección de anchor y positive del mismo jugador: `anchor_sample, positive_sample = rng.sample(positives, 2)`
- **Línea 716-720:** Selección de negativo de un jugador diferente
- **Línea 727-738:** Función `serialize_triplets()` que prepara los datos (anchor_board, anchor_heat, positive, negative)
- **Línea 818-819:** Generación de 8192 tripletes de entrenamiento y 2048 de validación

---

### **4. Inferencia - Comparar con estándar de cada jugador**
**Archivo:** `labs/notebooks/100_chess_siamese.ipynb`
- **Línea 1253-1259:** Código de validación que compara embeddings y calcula distancias
- **Línea 1256:** Identificación de pares del mismo jugador (intra-player)
- **Línea 1258:** Identificación de pares de diferentes jugadores (inter-player)
- **Línea 1277-1278:** Resumen estadístico de distancias intra-player vs inter-player

**Archivo:** `labs/src/embedding_exporter.py`
- **Línea 155-204:** Clase `EmbeddingExporter` para inferencia
- **Línea 177-204:** Método `run()` que genera embeddings y centroides por jugador

---

### **5. Partida nueva vs referencia media de N jugadores**
**Archivo:** `labs/notebooks/100_chess_siamese.ipynb`
- **Línea 1301-1340:** Función `export_embedding_cache()` que exporta embeddings y centroides
- **Línea 1316-1323:** Cálculo de centroides por jugador (promedio de vectores)
- **Línea 1317-1318:** `centroid = np.mean(stacked, axis=0)` - Promedio latente

**Archivo:** `labs/src/embedding_exporter.py`
- **Línea 226-250:** Método `_write_manifest()` que guarda centroides por jugador
- **Línea 241-244:** Cálculo y guardado de centroides: `centroid = stacked.mean(axis=0)`

---

### **6. Clasificación por distancia mínima**
**Archivo:** `labs/notebooks/100_chess_siamese.ipynb`
- **Línea 1255:** Cálculo de distancia L2: `dist = float(np.linalg.norm(val_embeddings[i] - val_embeddings[j]))`
- **Línea 1256-1259:** Clasificación basada en si la distancia es mínima para el jugador correcto

---

### **7. Support Set - Opciones A y B**
**A. Elegir referencia a ojo:**
**Archivo:** `labs/notebooks/100_chess_siamese.ipynb`
- **Línea 132:** `"players_whitelist": None` - Permite filtrar jugadores manualmente
- **Línea 353-354:** Lógica para whitelist de jugadores específicos

**B. Usar Centroide latente (promedio de vectores):**
**Archivo:** `labs/notebooks/100_chess_siamese.ipynb`
- **Línea 1316-1323:** Implementación de centroides por jugador
- **Línea 1317:** `centroid = np.mean(stacked, axis=0)` - Promedio de embeddings

**Archivo:** `labs/src/embedding_exporter.py`
- **Línea 240-244:** Cálculo de centroides como promedio de vectores del jugador

---

### **8. Transfer Learning con ResNet en vez de GASF**
**Archivo:** `labs/notebooks/100_chess_siamese.ipynb`
- **Línea 836-841:** Uso de `ResNet50` con pesos de ImageNet
- **Línea 838:** `weights="imagenet"` - Transfer learning
- **Línea 841:** `base_cnn.trainable = False` - Congelado de pesos preentrenados
- **Línea 843-851:** Encoder de tablero usando ResNet50 + GlobalAveragePooling

**Archivo:** `labs/src/embedding_exporter.py`
- **Línea 107-135:** Clase `ResNetFeatureExtractor` con ResNet18
- **Línea 129:** Uso de `ResNet18_Weights.IMAGENET1K_V1`
- **Línea 132:** `model = resnet18(weights=weight_enum)` - Transfer learning
- **Línea 206-216:** Método `_embed_sample()` que procesa imágenes con ResNet

---

### **9. (Espacio reservado para TODOS)**
**Estado:** Actualmente hay una tarea.
- **Diagrama:** Actualizar el diagrama proporcionado por Matencio siendo más claro con el uso de la red Siamesa.

---

### **11. Preparar embeddings previamente**
**Archivo:** `labs/notebooks/100_chess_siamese.ipynb`
- **Línea 133:** `"embedding_cache_dir": LABS_DIR / "dataset" / "embedding"` - Directorio de caché
- **Línea 1301-1340:** Función `export_embedding_cache()` que exporta embeddings pre-calculados
- **Línea 1305:** `embeddings = model.predict(ds, verbose=1)` - Generación de embeddings
- **Línea 1309-1313:** Guardado de embeddings individuales por jugador/partida
- **Línea 1319-1320:** Guardado de centroides por jugador
- **Línea 1326-1334:** Generación de manifest.json con metadata

**Archivo:** `labs/src/embedding_exporter.py` (Script dedicado)
- **Línea 155-204:** Clase `EmbeddingExporter` completa para exportar embeddings
- **Línea 177-204:** Método `run()` que procesa todas las muestras
- **Línea 189-193:** Guardado de vectores por muestra
- **Línea 201-204:** Generación de manifest con información de centroides
- **Línea 218-223:** Método `_save_vector()` que guarda embeddings en formato .npy

---

### Arquitectura del Modelo
- **Base:** ResNet50 (ImageNet) congelado para tableros + CNN custom para heatmaps
- **Temporal:** Bidirectional GRU (256 unidades) para secuencias de bloques
- **Embedding:** Vector de 256 dimensiones normalizado L2
- **Loss:** Triplet Loss con margen 0.4

### Dataset
- **Jugadores:** 100 (top por cantidad de partidas)
- **Partidas por jugador:** Hasta 40 (mínimo 4)
- **Split:** 80% train, 20% validación
- **Tripletes:** 8192 train, 2048 validación

### Embeddings
- **Formato:** Numpy arrays (.npy)
- **Estructura:** `embeddings/{player}/{game_id}.npy` + `embeddings/{player}/centroid.npy`
- **Manifest:** JSON con metadata y rutas a centroides
