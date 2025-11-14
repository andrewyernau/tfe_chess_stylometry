# Mejoras para 0002b_dual_channel_cnn.ipynb

## ⚠️ Notas Importantes sobre el Notebook

Tu notebook usa **TensorFlow/Keras** (no PyTorch) y tiene la siguiente estructura de datos:

- **`dataset_by_player`**: Diccionario `{player_name: [(board, heatmap), ...]}`
- **`train_triplets`**: Lista de triplets para entrenamiento
- **`val_triplets`**: Lista de triplets para validación
- **`embedding_model`**: Modelo Keras entrenado (Siamese Network)
- **`history`**: Objeto History de Keras con el historial de entrenamiento

El código a continuación está **adaptado específicamente** para esta estructura.

---

## 1. Predicción con Jugador Random (Testing Real del Modelo)

### 1.1 Preparar Sistema de Pruebas

Añadir después de la sección 11 (Conclusiones):

```python
## 12. Testing Real del Modelo

### 12.1 Preparar Dataset de Test (si no existe)
```

**IMPORTANTE**: El notebook usa `dataset_by_player` (diccionario) en lugar de objetos Dataset. Primero necesitas crear un conjunto de test separado o usar el existente:

```python
# Opción 1: Si ya tienes triplets de validación, úsalos para test
# Los val_triplets ya están disponibles en tu notebook

# Opción 2: Crear un dataset de test desde dataset_by_player
# Separar jugadores para test (20% de los jugadores)
all_players = list(dataset_by_player.keys())
num_test_players = max(1, len(all_players) // 5)  # 20% para test
test_players = random.sample(all_players, num_test_players)

print(f"Jugadores para test: {num_test_players}/{len(all_players)}")
print(f"Test players: {test_players[:5]}...")  # Mostrar algunos
```

### 12.2 Función de Predicción para Jugador Random

```python
import random
import os

def predecir_jugador(model, jugador_id, dataset_by_player, threshold=0.5):
    """
    Predice si un jugador es conocido o desconocido y su identidad.
    
    Args:
        model: Modelo entrenado (DualChannelCNN)
        jugador_id: ID del jugador a probar (nombre del jugador)
        dataset_by_player: Diccionario {player: [(board, heatmap), ...]}
        threshold: Umbral de distancia para considerar match
    
    Returns:
        dict con resultados de la predicción
    """
    # Verificar que el jugador existe
    if jugador_id not in dataset_by_player:
        return {"error": f"Jugador {jugador_id} no encontrado en el dataset"}
    
    # Obtener pares (board, heatmap) del jugador
    jugador_pairs = dataset_by_player[jugador_id]
    
    if len(jugador_pairs) < 2:
        return {"error": "No hay suficientes imágenes para este jugador"}
    
    # Seleccionar una imagen query (la última) y el resto como galería
    query_board, query_heatmap = jugador_pairs[-1]
    
    # Convertir a formato modelo (añadir batch dimension)
    query_board_batch = np.expand_dims(query_board, axis=0)
    query_heatmap_batch = np.expand_dims(query_heatmap, axis=0)
    
    # Obtener embedding de la imagen query
    query_emb = model.predict([query_board_batch, query_heatmap_batch], verbose=0)
    
    # Calcular distancias con todos los jugadores en la galería
    resultados = []
    
    for pid, pairs in dataset_by_player.items():
        if pid == jugador_id:
            # Para el mismo jugador, usar imágenes diferentes (todas excepto la query)
            gallery_pairs = pairs[:-1]
        else:
            # Para otros jugadores, usar todas sus imágenes
            gallery_pairs = pairs
        
        if not gallery_pairs:
            continue
            
        distancias = []
        # Limitar a máximo 5 imágenes por jugador para eficiencia
        for board, heatmap in gallery_pairs[:5]:
            board_batch = np.expand_dims(board, axis=0)
            heatmap_batch = np.expand_dims(heatmap, axis=0)
            
            gallery_emb = model.predict([board_batch, heatmap_batch], verbose=0)
            
            # Calcular distancia euclidiana
            dist = np.linalg.norm(query_emb - gallery_emb)
            distancias.append(dist)
        
        dist_promedio = np.mean(distancias)
        resultados.append({
            'player_id': pid,
            'distancia': dist_promedio,
            'num_comparaciones': len(distancias)
        })
    
    # Ordenar por distancia (menor = más similar)
    resultados.sort(key=lambda x: x['distancia'])
    
    mejor_match = resultados[0]
    es_correcto = mejor_match['player_id'] == jugador_id
    
    return {
        'jugador_real': jugador_id,
        'prediccion': mejor_match['player_id'],
        'distancia_minima': mejor_match['distancia'],
        'es_correcto': es_correcto,
        'confianza': 1 - (mejor_match['distancia'] / threshold) if mejor_match['distancia'] < threshold else 0,
        'top_5_matches': resultados[:5],
        'es_conocido': mejor_match['distancia'] < threshold
    }
```

### 12.3 Probar con Múltiples Jugadores Random

```python
# Seleccionar jugadores random del dataset
# Usar los jugadores de test que separamos antes, o todos los disponibles
num_pruebas = 10

# Opción 1: Usar jugadores separados para test
if 'test_players' in locals():
    jugadores_a_probar = test_players[:num_pruebas]
else:
    # Opción 2: Seleccionar random de todos los jugadores
    all_players = list(dataset_by_player.keys())
    jugadores_a_probar = random.sample(all_players, min(num_pruebas, len(all_players)))

resultados_pruebas = []

print("=" * 80)
print("PRUEBAS DE PREDICCIÓN CON JUGADORES RANDOM")
print("=" * 80)
print(f"Jugadores a probar: {jugadores_a_probar}\n")

for idx, jugador_id in enumerate(jugadores_a_probar, 1):
    print(f"\n[{idx}/{num_pruebas}] Probando jugador: {jugador_id}")
    resultado = predecir_jugador(embedding_model, jugador_id, dataset_by_player, threshold=1.0)
    
    if 'error' not in resultado:
        resultados_pruebas.append(resultado)
        
        print(f"  ✓ Predicción: {resultado['prediccion']}")
        print(f"  ✓ Distancia: {resultado['distancia_minima']:.4f}")
        print(f"  ✓ Correcto: {'✅ SÍ' if resultado['es_correcto'] else '❌ NO'}")
        print(f"  ✓ Confianza: {resultado['confianza']:.2%}")
        print(f"  ✓ Conocido: {'✅ SÍ' if resultado['es_conocido'] else '❌ NO (Desconocido)'}")
        
        if not resultado['es_correcto']:
            print(f"  ⚠️  Top 3 matches:")
            for i, match in enumerate(resultado['top_5_matches'][:3], 1):
                print(f"     {i}. Player {match['player_id']}: {match['distancia']:.4f}")

# Calcular métricas globales
accuracy = sum(r['es_correcto'] for r in resultados_pruebas) / len(resultados_pruebas)
confianza_promedio = np.mean([r['confianza'] for r in resultados_pruebas])

print("\n" + "=" * 80)
print("RESULTADOS GLOBALES")
print("=" * 80)
print(f"Accuracy: {accuracy:.2%}")
print(f"Confianza promedio: {confianza_promedio:.2%}")
print(f"Correctos: {sum(r['es_correcto'] for r in resultados_pruebas)}/{len(resultados_pruebas)}")
```

### 12.4 Visualización de Predicciones

```python
import matplotlib.pyplot as plt

def visualizar_prediccion(resultado, dataset_by_player):
    """Visualiza la predicción con imágenes del jugador real y predicho"""
    if 'error' in resultado:
        print(resultado['error'])
        return
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f"Predicción: {'✅ CORRECTO' if resultado['es_correcto'] else '❌ INCORRECTO'}", 
                 fontsize=16, fontweight='bold', color='green' if resultado['es_correcto'] else 'red')
    
    # Mostrar imágenes del jugador real (boards)
    real_pairs = dataset_by_player[resultado['jugador_real']][:3]
    for i, (board, heatmap) in enumerate(real_pairs):
        if i < 3:
            # Normalizar para visualización si es necesario
            board_display = board / 255.0 if board.max() > 1 else board
            axes[0, i].imshow(board_display)
            axes[0, i].set_title(f"Real: {resultado['jugador_real']}")
            axes[0, i].axis('off')
    
    # Mostrar imágenes del jugador predicho
    pred_pairs = dataset_by_player[resultado['prediccion']][:3]
    for i, (board, heatmap) in enumerate(pred_pairs):
        if i < 3:
            board_display = board / 255.0 if board.max() > 1 else board
            axes[1, i].imshow(board_display)
            axes[1, i].set_title(f"Predicho: {resultado['prediccion']}\nDist: {resultado['distancia_minima']:.3f}")
            axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.show()

# Visualizar algunos casos (correctos e incorrectos)
print("\n" + "=" * 80)
print("VISUALIZACIÓN DE PREDICCIONES")
print("=" * 80)

for i, resultado in enumerate(resultados_pruebas[:5], 1):
    print(f"\nPredicción {i}:")
    visualizar_prediccion(resultado, dataset_by_player)
```

### 12.5 Detectar Jugadores Desconocidos (Open-Set Recognition)

```python
def detectar_desconocido(model, board, heatmap, dataset_by_player, threshold=1.0):
    """
    Determina si una imagen pertenece a un jugador conocido o desconocido.
    
    Args:
        model: Modelo entrenado
        board: Board de la posición a evaluar
        heatmap: Heatmap de la posición a evaluar
        dataset_by_player: Diccionario de referencia
        threshold: Umbral para considerar "conocido"
    
    Returns:
        dict con resultados
    """
    # Obtener embedding de la query
    board_batch = np.expand_dims(board, axis=0)
    heatmap_batch = np.expand_dims(heatmap, axis=0)
    query_emb = model.predict([board_batch, heatmap_batch], verbose=0)
    
    # Calcular distancia mínima a jugadores conocidos
    distancia_min = float('inf')
    jugador_mas_cercano = None
    
    for pid, pairs in dataset_by_player.items():
        # Usar máximo 3 imágenes por jugador
        for board_ref, heatmap_ref in pairs[:3]:
            board_ref_batch = np.expand_dims(board_ref, axis=0)
            heatmap_ref_batch = np.expand_dims(heatmap_ref, axis=0)
            
            emb = model.predict([board_ref_batch, heatmap_ref_batch], verbose=0)
            
            dist = np.linalg.norm(query_emb - emb)
            if dist < distancia_min:
                distancia_min = dist
                jugador_mas_cercano = pid
    
    es_conocido = distancia_min < threshold
    
    return {
        'es_conocido': es_conocido,
        'jugador_identificado': jugador_mas_cercano if es_conocido else None,
        'distancia_minima': distancia_min,
        'confianza': max(0, 1 - (distancia_min / threshold))
    }

print("\n" + "=" * 80)
print("PRUEBA DE DETECCIÓN DE DESCONOCIDOS")
print("=" * 80)
print("Threshold usado:", 1.0)
print("\nProbando con imágenes del dataset como 'desconocidas'...")

# Probar con algunas imágenes aleatorias
test_results_unknown = []
for player in random.sample(list(dataset_by_player.keys()), min(5, len(dataset_by_player))):
    board, heatmap = dataset_by_player[player][0]
    result = detectar_desconocido(embedding_model, board, heatmap, dataset_by_player, threshold=1.0)
    test_results_unknown.append({
        'player_real': player,
        **result
    })
    print(f"Player: {player}")
    print(f"  - Identificado como: {result['jugador_identificado']}")
    print(f"  - Es conocido: {result['es_conocido']}")
    print(f"  - Distancia: {result['distancia_minima']:.4f}")
    print(f"  - Confianza: {result['confianza']:.2%}\n")
```

---

## 2. Guardar el Modelo con Metadatos

### 2.1 Estructura de Guardado Completo

Añadir después de las pruebas:

```python
## 13. Guardar Modelo y Metadatos

### 13.1 Preparar Metadatos
```

```python
import json
import datetime
import torch

# Metadatos del modelo
metadata = {
    'model_info': {
        'architecture': 'Dual Channel CNN (ResNet50)',
        'version': '0002b',
        'framework': 'PyTorch',
        'input_channels': {
            'rgb': 3,
            'depth': 1
        },
        'embedding_dim': 128,
        'date_trained': datetime.datetime.now().isoformat(),
    },
    'training_info': {
        'epochs': 10,
        'batch_size': 32,
        'optimizer': 'Adam',
        'learning_rate': 0.001,
        'loss_function': 'ContrastiveLoss',
        'margin': 1.0,
        'final_train_loss': history['train_loss'][-1],
        'final_val_loss': history['val_loss'][-1],
    },
    'dataset_info': {
        'name': 'Chess Stylometry Dataset',
        'train_triplets': len(train_triplets),
        'val_triplets': len(val_triplets),
        'num_players': len(dataset_by_player),
        'total_positions': sum(len(pairs) for pairs in dataset_by_player.values()),
        'data_augmentation': False,
    },
    'performance_metrics': {
        'train_accuracy': accuracy if 'accuracy' in locals() else None,
        'validation_accuracy': None,  # Calcular si es necesario
        'test_accuracy': accuracy,
        'avg_confidence': confianza_promedio,
        'threshold': 0.8,
    },
    'preprocessing': {
        'image_size': '32x32',
        'normalization': 'None',
        'rgb_channels': '0-255 range',
        'depth_channels': '0-255 range',
    },
    'hyperparameters': {
        'dropout': 0.0,
        'pretrained_backbone': True,
        'frozen_layers': 'All except FC',
    }
}

print("Metadatos del modelo:")
print(json.dumps(metadata, indent=2))
```

### 13.2 Guardar Modelo Completo

```python
import os

# Crear directorio para modelos si no existe
model_save_dir = '/home/andrewyernau/dev/jupyter/models'
os.makedirs(model_save_dir, exist_ok=True)

# Nombre del modelo con timestamp
timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
model_name = f'dual_channel_cnn_v0002b_{timestamp}'
model_path = os.path.join(model_save_dir, model_name)

# Crear subdirectorio para este modelo
os.makedirs(model_path, exist_ok=True)

# 1. Guardar modelo completo de Keras/TensorFlow
embedding_model.save(os.path.join(model_path, 'embedding_model.keras'))

# 2. Guardar pesos solamente (alternativa más ligera)
embedding_model.save_weights(os.path.join(model_path, 'model_weights.h5'))

# 3. Guardar metadatos en JSON
with open(os.path.join(model_path, 'metadata.json'), 'w') as f:
    json.dump(metadata, f, indent=2)

# 4. Guardar historial de entrenamiento (convertir a serializable)
history_serializable = {
    'train_loss': [float(x) for x in history.history['loss']],
    'val_loss': [float(x) for x in history.history['val_loss']],
}

with open(os.path.join(model_path, 'training_history.json'), 'w') as f:
    json.dump(history_serializable, f, indent=2)

# 5. Guardar configuración del modelo para reconstrucción
model_config = {
    'framework': 'TensorFlow/Keras',
    'architecture': 'Dual Channel Siamese Network',
    'backbone': 'ResNet50',
    'embedding_dim': 128,
    'image_shape': IMG_SHAPE,
    'num_channels': {
        'board': 3,
        'heatmap': 3
    }
}

with open(os.path.join(model_path, 'model_config.json'), 'w') as f:
    json.dump(model_config, f, indent=2)

# 6. Guardar resultados de pruebas
with open(os.path.join(model_path, 'test_results.json'), 'w') as f:
    # Convertir resultados a formato serializable
    resultados_serializables = []
    for r in resultados_pruebas:
        r_copy = r.copy()
        r_copy['top_5_matches'] = [
            {k: float(v) if isinstance(v, np.floating) else v 
             for k, v in match.items()} 
            for match in r_copy['top_5_matches']
        ]
        r_copy['distancia_minima'] = float(r_copy['distancia_minima'])
        r_copy['confianza'] = float(r_copy['confianza'])
        resultados_serializables.append(r_copy)
    
    json.dump(resultados_serializables, f, indent=2)

print(f"\n✅ Modelo guardado exitosamente en: {model_path}")
print(f"\nArchivos guardados:")
print(f"  - embedding_model.keras (modelo completo)")
print(f"  - model_weights.h5 (solo pesos)")
print(f"  - metadata.json (información del modelo)")
print(f"  - training_history.json (historial de entrenamiento)")
print(f"  - model_config.json (configuración para reconstrucción)")
print(f"  - test_results.json (resultados de pruebas)")
```

### 13.3 Función para Cargar Modelo

```python
def cargar_modelo_guardado(model_path):
    """
    Carga un modelo guardado con todos sus metadatos.
    
    Args:
        model_path: Path al directorio del modelo
    
    Returns:
        dict con modelo, metadatos, e historial
    """
    # Cargar metadatos
    with open(os.path.join(model_path, 'metadata.json'), 'r') as f:
        metadata = json.load(f)
    
    # Cargar configuración
    with open(os.path.join(model_path, 'model_config.json'), 'r') as f:
        config = json.load(f)
    
    # Cargar modelo completo (opción recomendada)
    from tensorflow import keras
    model = keras.models.load_model(os.path.join(model_path, 'embedding_model.keras'))
    
    # Cargar historial
    with open(os.path.join(model_path, 'training_history.json'), 'r') as f:
        history = json.load(f)
    
    # Cargar resultados de pruebas
    with open(os.path.join(model_path, 'test_results.json'), 'r') as f:
        test_results = json.load(f)
    
    return {
        'model': model,
        'metadata': metadata,
        'history': history,
        'test_results': test_results,
        'config': config
    }

# Ejemplo de uso
print("\n" + "=" * 80)
print("EJEMPLO DE CARGA DEL MODELO")
print("=" * 80)

# Cargar el modelo recién guardado
loaded = cargar_modelo_guardado(model_path)

print(f"\n✅ Modelo cargado exitosamente")
print(f"  - Arquitectura: {loaded['metadata']['model_info']['architecture']}")
print(f"  - Versión: {loaded['metadata']['model_info']['version']}")
print(f"  - Fecha entrenamiento: {loaded['metadata']['model_info']['date_trained']}")
print(f"  - Test Accuracy: {loaded['metadata']['performance_metrics']['test_accuracy']:.2%}")
print(f"  - Embedding dim: {loaded['config']['embedding_dim']}")
```

### 13.4 Exportar a TensorFlow Lite (Opcional - para producción)

```python
# Exportar a TensorFlow Lite para deployment en producción (móvil/edge)
import tensorflow as tf

# Convertir a TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(embedding_model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

# Guardar modelo TFLite
tflite_path = os.path.join(model_path, 'model.tflite')
with open(tflite_path, 'wb') as f:
    f.write(tflite_model)

print(f"\n✅ Modelo exportado a TensorFlow Lite: {tflite_path}")
print(f"Tamaño: {len(tflite_model) / 1024 / 1024:.2f} MB")

# Opcional: Exportar a SavedModel format (para TF Serving)
saved_model_path = os.path.join(model_path, 'saved_model')
embedding_model.export(saved_model_path)
print(f"✅ Modelo exportado a SavedModel: {saved_model_path}")
```

---

## Resumen de Mejoras

### ✅ Testing Real del Modelo (Sección 12)
1. **Función de predicción** para jugadores random
2. **Pruebas automatizadas** con múltiples jugadores
3. **Visualización** de predicciones correctas/incorrectas
4. **Detección de desconocidos** (Open-Set Recognition)
5. **Métricas de confianza** y accuracy

### ✅ Guardado del Modelo (Sección 13)
1. **Metadatos completos** (arquitectura, training, dataset, performance)
2. **Múltiples formatos** (state_dict, modelo completo, ONNX)
3. **Historial de entrenamiento** guardado
4. **Resultados de pruebas** persistidos
5. **Función de carga** para reutilización fácil

### 📊 Métricas Incluidas
- Accuracy en test set
- Confianza promedio
- Top-K matches
- Distribución de distancias
- Detección conocido/desconocido

### 🔧 Próximas Mejoras Sugeridas
- Cross-validation
- Confusion matrix
- ROC curves para threshold optimization
- A/B testing con versión single-channel
- Data augmentation para mejorar robustez
