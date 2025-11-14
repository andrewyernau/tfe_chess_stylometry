# Código Actualizado para 0002b_dual_channel_cnn.ipynb

## ⚠️ IMPORTANTE: Reemplazar TODO el código de testing existente

Este código corrige:
1. ✅ Error `Invalid dtype: object` - Convierte arrays a float32
2. ✅ Nueva estrategia de testing con jugadores de entrenamiento + nuevos
3. ✅ Emparejamiento correcto tablero-heatmap (1 tablero = 1 heatmap por partida)

---

## Sección 12: Testing Real del Modelo (CÓDIGO COMPLETO)

```python
## 12. Testing Real del Modelo con Jugadores Nuevos y del Entrenamiento

import random
import json
import datetime
import os
import sys
sys.path.append('/home/andrewyernau/dev/jupyter/labs')

# =============================================================================
# 12.1 CONFIGURACIÓN DEL TEST
# =============================================================================

# Constantes para el test
NUM_PLAYERS_TEST = 5  # Número de jugadores NUEVOS (fuera del entrenamiento)
NUM_PLAYERS_TRAIN = 5  # Número de jugadores del ENTRENAMIENTO (para verificar overfitting)
SELECTED_EVENT = "Rated Blitz game"  # Mismo evento usado en entrenamiento
TEST_THRESHOLD = 1.0  # Ajustar según distribución de distancias

print("\n" + "=" * 80)
print("🎯 CONFIGURACIÓN DEL TEST")
print("=" * 80)
print(f"Evento seleccionado: {SELECTED_EVENT}")
print(f"Jugadores nuevos a probar: {NUM_PLAYERS_TEST}")
print(f"Jugadores del entrenamiento: {NUM_PLAYERS_TRAIN}")
print(f"Threshold: {TEST_THRESHOLD}")

# =============================================================================
# 12.2 GENERAR DATOS DE JUGADORES NUEVOS (Fuera del entrenamiento)
# =============================================================================

from pipeline_stylometry_by_event import ChessStylometryPipelineByEvent
from pathlib import Path

print("\n" + "=" * 80)
print("📦 GENERANDO DATOS DE JUGADORES NUEVOS")
print("=" * 80)

# Configurar paths
massive_pgn = Path("/home/andrewyernau/dev/jupyter/data/lichess_db_standard_rated_2013-01.pgn")
output_base = Path("/home/andrewyernau/dev/jupyter/labs/output")

# Obtener jugadores YA entrenados
trained_players = set(dataset_by_player.keys())
print(f"Jugadores ya entrenados: {len(trained_players)}")
print(f"Ejemplos: {list(trained_players)[:5]}")

# Generar datos de jugadores NUEVOS (fuera del entrenamiento)
test_pipeline = ChessStylometryPipelineByEvent(
    massive_pgn=massive_pgn,
    output_base=output_base,
    event_type=SELECTED_EVENT,
    num_players=NUM_PLAYERS_TEST,  # Solo los jugadores de test
    games_per_player=30,
    min_threshold=0.7,
    start_move=15,
    end_move=23,
    compression_factor=1,
    use_relative_time=True
)

print(f"\n🔍 Extrayendo {NUM_PLAYERS_TEST} jugadores nuevos del evento '{SELECTED_EVENT}'...")

# IMPORTANTE: Solo extraer jugadores que NO estén en el entrenamiento
# Esto lo haremos manualmente filtrando después

# Ejecutar pipeline solo para obtener nuevos jugadores
test_results = test_pipeline.run()

# =============================================================================
# 12.3 CARGAR DATOS DE JUGADORES NUEVOS
# =============================================================================

print("\n" + "=" * 80)
print("📂 CARGANDO DATOS DE JUGADORES NUEVOS")
print("=" * 80)

event_safe = SELECTED_EVENT.replace(" ", "_").replace("/", "_")
event_dir = output_base / "events" / event_safe
board_images_dir = event_dir / "board_images"
heatmap_images_dir = event_dir / "heatmap_images"

# Cargar imágenes de jugadores NUEVOS
new_players_data = {}

for player_dir in sorted(board_images_dir.glob("*")):
    if not player_dir.is_dir():
        continue
    
    player_name = player_dir.name
    
    # FILTRAR: Solo jugadores que NO estén en el entrenamiento
    if player_name in trained_players:
        continue
    
    # Cargar pares (board, heatmap) para este jugador nuevo
    board_files = sorted(player_dir.glob("game_*.png"))
    heatmap_dir = heatmap_images_dir / player_name
    
    if not heatmap_dir.exists():
        continue
    
    pairs = []
    for board_file in board_files:
        game_num = board_file.stem.split('_')[1]  # game_0001.png -> 0001
        heatmap_file = heatmap_dir / f"game_{game_num}.png"
        
        if not heatmap_file.exists():
            continue
        
        # Cargar imágenes
        board = cv2.imread(str(board_file), cv2.IMREAD_GRAYSCALE)
        heatmap = cv2.imread(str(heatmap_file), cv2.IMREAD_COLOR)
        
        if board is None or heatmap is None:
            continue
        
        # Asegurar que estén en el formato correcto
        board = cv2.resize(board, IMG_SHAPE) if board.shape[:2] != IMG_SHAPE else board
        heatmap = cv2.resize(heatmap, IMG_SHAPE) if heatmap.shape[:2] != IMG_SHAPE else heatmap
        
        # Aplicar preprocessing de ResNet
        board_prep = resnet.preprocess_input(
            np.expand_dims(np.stack([board, board, board], axis=-1), axis=0)
        )[0]
        heatmap_prep = resnet.preprocess_input(np.expand_dims(heatmap, axis=0))[0]
        
        pairs.append((board_prep, heatmap_prep))
    
    if len(pairs) >= 2:  # Solo jugadores con al menos 2 partidas
        new_players_data[player_name] = pairs

print(f"\n✅ Jugadores nuevos cargados: {len(new_players_data)}")
print(f"Ejemplos: {list(new_players_data.keys())[:5]}")

# Seleccionar NUM_PLAYERS_TEST jugadores aleatorios de los nuevos
if len(new_players_data) < NUM_PLAYERS_TEST:
    print(f"⚠️  Solo hay {len(new_players_data)} jugadores nuevos, usando todos")
    test_new_players = list(new_players_data.keys())
else:
    test_new_players = random.sample(list(new_players_data.keys()), NUM_PLAYERS_TEST)

# Seleccionar NUM_PLAYERS_TRAIN jugadores del entrenamiento
test_train_players = random.sample(list(trained_players), 
                                   min(NUM_PLAYERS_TRAIN, len(trained_players)))

print(f"\n📊 Jugadores seleccionados para el test:")
print(f"  - Nuevos (fuera entrenamiento): {len(test_new_players)}")
print(f"  - Del entrenamiento: {len(test_train_players)}")

# =============================================================================
# 12.4 FUNCIÓN DE PREDICCIÓN (CORREGIDA)
# =============================================================================

def predecir_jugador(model, jugador_id, dataset_dict, is_new_player=False, threshold=1.0):
    """
    Predice la identidad de un jugador.
    
    Parameters
    ----------
    model : keras.Model
        Modelo entrenado
    jugador_id : str
        Nombre del jugador a probar
    dataset_dict : dict
        Diccionario con los datos (puede incluir jugadores nuevos y entrenados)
    is_new_player : bool
        Si es True, el jugador es nuevo (no visto en entrenamiento)
    threshold : float
        Umbral de distancia
    
    Returns
    -------
    dict
        Resultados de la predicción
    """
    if jugador_id not in dataset_dict:
        return {"error": f"Jugador {jugador_id} no encontrado"}
    
    pairs = dataset_dict[jugador_id]
    if len(pairs) < 2:
        return {"error": "No hay suficientes imágenes"}
    
    # Query: última partida
    query_board, query_heatmap = pairs[-1]
    
    # IMPORTANTE: Convertir a float32 y asegurar formato correcto
    query_board = np.asarray(query_board, dtype=np.float32)
    query_heatmap = np.asarray(query_heatmap, dtype=np.float32)
    
    # Añadir batch dimension
    query_board_batch = np.expand_dims(query_board, axis=0)
    query_heatmap_batch = np.expand_dims(query_heatmap, axis=0)
    
    # Obtener embedding
    query_emb = model.predict([query_board_batch, query_heatmap_batch], verbose=0)
    
    # Calcular distancias con TODOS los jugadores (entrenados)
    # Los jugadores nuevos NO deben estar en la galería de comparación
    resultados = []
    
    # Usar SOLO los jugadores del entrenamiento como galería
    for pid, pid_pairs in dataset_by_player.items():
        # Para el mismo jugador, usar partidas diferentes
        if pid == jugador_id and not is_new_player:
            gallery_pairs = pid_pairs[:-1]
        else:
            gallery_pairs = pid_pairs
        
        if not gallery_pairs:
            continue
        
        distancias = []
        for board, heatmap in gallery_pairs[:5]:  # Máximo 5 partidas
            # Convertir a float32
            board = np.asarray(board, dtype=np.float32)
            heatmap = np.asarray(heatmap, dtype=np.float32)
            
            board_batch = np.expand_dims(board, axis=0)
            heatmap_batch = np.expand_dims(heatmap, axis=0)
            
            gallery_emb = model.predict([board_batch, heatmap_batch], verbose=0)
            dist = np.linalg.norm(query_emb - gallery_emb)
            distancias.append(dist)
        
        resultados.append({
            'player_id': pid,
            'distancia': np.mean(distancias),
            'num_comparaciones': len(distancias)
        })
    
    resultados.sort(key=lambda x: x['distancia'])
    mejor_match = resultados[0]
    
    # Para jugadores nuevos, NO deberían ser clasificados como conocidos
    es_correcto = (not is_new_player) and (mejor_match['player_id'] == jugador_id)
    es_conocido = mejor_match['distancia'] < threshold
    
    return {
        'jugador_real': jugador_id,
        'prediccion': mejor_match['player_id'],
        'distancia_minima': mejor_match['distancia'],
        'es_correcto': es_correcto,
        'is_new_player': is_new_player,
        'confianza': max(0, 1 - (mejor_match['distancia'] / threshold)),
        'top_5_matches': resultados[:5],
        'es_conocido': es_conocido
    }

# =============================================================================
# 12.5 EJECUTAR TESTS
# =============================================================================

print("\n" + "=" * 80)
print("🧪 EJECUTANDO TESTS")
print("=" * 80)

resultados_pruebas = []

# TEST 1: Jugadores NUEVOS (no deberían ser reconocidos)
print("\n📍 TEST 1: Jugadores NUEVOS (fuera del entrenamiento)")
print("-" * 80)

# Combinar datasets para la función de predicción
combined_dataset = {**dataset_by_player, **new_players_data}

for idx, player in enumerate(test_new_players, 1):
    print(f"\n[{idx}/{len(test_new_players)}] 🔍 Probando NUEVO: {player}")
    resultado = predecir_jugador(
        embedding_model, 
        player, 
        combined_dataset, 
        is_new_player=True,
        threshold=TEST_THRESHOLD
    )
    
    if 'error' not in resultado:
        resultados_pruebas.append(resultado)
        
        # Un jugador nuevo DEBERÍA tener distancia > threshold (desconocido)
        status = "✅ CORRECTO" if not resultado['es_conocido'] else "❌ FALSO POSITIVO"
        print(f"  {status}")
        print(f"  📏 Distancia mínima: {resultado['distancia_minima']:.4f}")
        print(f"  🎯 Match más cercano: {resultado['prediccion']}")
        print(f"  📊 Clasificado como: {'Conocido' if resultado['es_conocido'] else '⭐ Desconocido'}")

# TEST 2: Jugadores DEL ENTRENAMIENTO (deberían ser reconocidos correctamente)
print("\n\n📍 TEST 2: Jugadores DEL ENTRENAMIENTO (verificar overfitting)")
print("-" * 80)

for idx, player in enumerate(test_train_players, 1):
    print(f"\n[{idx}/{len(test_train_players)}] 🔍 Probando ENTRENADO: {player}")
    resultado = predecir_jugador(
        embedding_model,
        player,
        combined_dataset,
        is_new_player=False,
        threshold=TEST_THRESHOLD
    )
    
    if 'error' not in resultado:
        resultados_pruebas.append(resultado)
        
        status = "✅" if resultado['es_correcto'] else "❌"
        print(f"  {status} Predicción: {resultado['prediccion']}")
        print(f"  📏 Distancia: {resultado['distancia_minima']:.4f}")
        print(f"  📊 Confianza: {resultado['confianza']:.2%}")
        
        if not resultado['es_correcto']:
            print(f"  ⚠️  Top 3 matches:")
            for i, m in enumerate(resultado['top_5_matches'][:3], 1):
                print(f"      {i}. {m['player_id']}: {m['distancia']:.4f}")

# =============================================================================
# 12.6 MÉTRICAS GLOBALES
# =============================================================================

print("\n" + "=" * 80)
print("📊 RESULTADOS GLOBALES")
print("=" * 80)

# Separar resultados por tipo
resultados_nuevos = [r for r in resultados_pruebas if r['is_new_player']]
resultados_entrenados = [r for r in resultados_pruebas if not r['is_new_player']]

# Métricas para jugadores NUEVOS
if resultados_nuevos:
    # Para nuevos: éxito = NO ser clasificado como conocido
    aciertos_nuevos = sum(1 for r in resultados_nuevos if not r['es_conocido'])
    tasa_rechazo = aciertos_nuevos / len(resultados_nuevos)
    dist_media_nuevos = np.mean([r['distancia_minima'] for r in resultados_nuevos])
    
    print(f"\n🆕 JUGADORES NUEVOS:")
    print(f"  Total probados: {len(resultados_nuevos)}")
    print(f"  Correctamente rechazados: {aciertos_nuevos}/{len(resultados_nuevos)}")
    print(f"  Tasa de rechazo: {tasa_rechazo:.2%} ({'✅ BUENO' if tasa_rechazo > 0.7 else '⚠️  MEJORAR'})")
    print(f"  Distancia media: {dist_media_nuevos:.4f}")
    print(f"  Falsos positivos: {len(resultados_nuevos) - aciertos_nuevos}")

# Métricas para jugadores ENTRENADOS
if resultados_entrenados:
    aciertos_entrenados = sum(1 for r in resultados_entrenados if r['es_correcto'])
    accuracy = aciertos_entrenados / len(resultados_entrenados)
    conf_media = np.mean([r['confianza'] for r in resultados_entrenados])
    dist_media_entrenados = np.mean([r['distancia_minima'] for r in resultados_entrenados])
    
    print(f"\n🎓 JUGADORES ENTRENADOS:")
    print(f"  Total probados: {len(resultados_entrenados)}")
    print(f"  Correctos: {aciertos_entrenados}/{len(resultados_entrenados)}")
    print(f"  Accuracy: {accuracy:.2%} ({'✅ BUENO' if accuracy > 0.7 else '⚠️  OVERFITTING?'})")
    print(f"  Confianza media: {conf_media:.2%}")
    print(f"  Distancia media: {dist_media_entrenados:.4f}")

# Comparar distancias
if resultados_nuevos and resultados_entrenados:
    print(f"\n📉 COMPARACIÓN:")
    print(f"  Distancia media NUEVOS: {dist_media_nuevos:.4f}")
    print(f"  Distancia media ENTRENADOS: {dist_media_entrenados:.4f}")
    separacion = dist_media_nuevos - dist_media_entrenados
    print(f"  Separación: {separacion:.4f} ({'✅ BUENA' if separacion > 0.3 else '⚠️  AJUSTAR THRESHOLD'})")
    print(f"  Threshold actual: {TEST_THRESHOLD:.4f}")
    threshold_sugerido = (dist_media_nuevos + dist_media_entrenados) / 2
    print(f"  Threshold sugerido: {threshold_sugerido:.4f}")

print("\n" + "=" * 80)
```

---

## Notas Importantes

### ✅ Correcciones Aplicadas:

1. **Error `Invalid dtype: object`**:
   ```python
   # Antes (causaba error):
   query_emb = model.predict([query_board_batch, query_heatmap_batch], verbose=0)
   
   # Después (correcto):
   query_board = np.asarray(query_board, dtype=np.float32)
   query_heatmap = np.asarray(query_heatmap, dtype=np.float32)
   query_emb = model.predict([query_board_batch, query_heatmap_batch], verbose=0)
   ```

2. **Nueva estrategia de testing**:
   - Genera datos de jugadores NUEVOS usando el pipeline
   - Filtra para asegurar que no estén en el entrenamiento
   - Prueba con jugadores nuevos (deberían ser rechazados)
   - Prueba con jugadores entrenados (deberían ser reconocidos)

3. **Emparejamiento correcto**:
   - Ahora el pipeline genera 1 tablero = 1 heatmap por partida
   - El tablero muestra transparencia temporal (últimos movimientos más visibles)
   - El heatmap usa tiempos relativos correctamente

### 📋 Variables Necesarias:

- `embedding_model` - Modelo entrenado
- `dataset_by_player` - Dataset de entrenamiento
- `IMG_SHAPE` - Tamaño de imagen
- `resnet` - Para preprocessing

### 🎯 Interpretación de Resultados:

**Jugadores NUEVOS**:
- ✅ **Bueno**: Tasa de rechazo > 70% (no los identifica como conocidos)
- ❌ **Malo**: Tasa de rechazo < 50% (overfitting, los identifica incorrectamente)

**Jugadores ENTRENADOS**:
- ✅ **Bueno**: Accuracy > 70%
- ⚠️ **Sospechoso**: Accuracy = 100% (posible overfitting)
- ❌ **Malo**: Accuracy < 50% (modelo no aprendió bien)

**Separación de Distancias**:
- ✅ **Buena**: Distancia nuevos - distancia entrenados > 0.3
- ⚠️ **Ajustar**: Separación < 0.2 (necesitas cambiar threshold o reentrenar)
