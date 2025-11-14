# Ejemplo Rápido: Testing y Guardado del Modelo

Este es un código completo y listo para copiar/pegar en tu notebook `0002b_dual_channel_cnn.ipynb`.

## Código Completo para Añadir al Final del Notebook

```python
## 12. Testing Real del Modelo con Jugadores Random

import random
import json
import datetime
import os

# =============================================================================
# 12.1 FUNCIÓN DE PREDICCIÓN
# =============================================================================

def predecir_jugador(model, jugador_id, dataset_by_player, threshold=1.0):
    """Predice la identidad de un jugador y calcula confianza"""
    
    if jugador_id not in dataset_by_player:
        return {"error": f"Jugador {jugador_id} no encontrado"}
    
    jugador_pairs = dataset_by_player[jugador_id]
    if len(jugador_pairs) < 2:
        return {"error": "No hay suficientes imágenes"}
    
    # Query: última imagen del jugador
    query_board, query_heatmap = jugador_pairs[-1]
    query_board_batch = np.expand_dims(query_board, axis=0)
    query_heatmap_batch = np.expand_dims(query_heatmap, axis=0)
    query_emb = model.predict([query_board_batch, query_heatmap_batch], verbose=0)
    
    # Calcular distancias con todos los jugadores
    resultados = []
    for pid, pairs in dataset_by_player.items():
        gallery_pairs = pairs[:-1] if pid == jugador_id else pairs
        if not gallery_pairs:
            continue
        
        distancias = []
        for board, heatmap in gallery_pairs[:5]:
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
    
    return {
        'jugador_real': jugador_id,
        'prediccion': mejor_match['player_id'],
        'distancia_minima': mejor_match['distancia'],
        'es_correcto': mejor_match['player_id'] == jugador_id,
        'confianza': max(0, 1 - (mejor_match['distancia'] / threshold)),
        'top_5_matches': resultados[:5],
        'es_conocido': mejor_match['distancia'] < threshold
    }

# =============================================================================
# 12.2 PROBAR CON 10 JUGADORES RANDOM
# =============================================================================

print("\n" + "=" * 80)
print("🎯 TESTING DEL MODELO CON JUGADORES RANDOM")
print("=" * 80)

num_pruebas = 10
all_players = list(dataset_by_player.keys())
jugadores_a_probar = random.sample(all_players, min(num_pruebas, len(all_players)))

resultados_pruebas = []

for idx, jugador_id in enumerate(jugadores_a_probar, 1):
    print(f"\n[{idx}/{num_pruebas}] 🔍 Probando: {jugador_id}")
    resultado = predecir_jugador(embedding_model, jugador_id, dataset_by_player, threshold=1.0)
    
    if 'error' not in resultado:
        resultados_pruebas.append(resultado)
        
        status = "✅" if resultado['es_correcto'] else "❌"
        print(f"  {status} Predicción: {resultado['prediccion']}")
        print(f"  📏 Distancia: {resultado['distancia_minima']:.4f}")
        print(f"  📊 Confianza: {resultado['confianza']:.2%}")
        print(f"  🎭 Conocido: {'Sí' if resultado['es_conocido'] else 'No'}")
        
        if not resultado['es_correcto']:
            print(f"  ⚠️  Top 3 matches:")
            for i, m in enumerate(resultado['top_5_matches'][:3], 1):
                print(f"      {i}. {m['player_id']}: {m['distancia']:.4f}")

# Métricas globales
accuracy = sum(r['es_correcto'] for r in resultados_pruebas) / len(resultados_pruebas)
confianza_promedio = np.mean([r['confianza'] for r in resultados_pruebas])

print("\n" + "=" * 80)
print("📊 RESULTADOS GLOBALES")
print("=" * 80)
print(f"✅ Accuracy: {accuracy:.2%}")
print(f"📈 Confianza promedio: {confianza_promedio:.2%}")
print(f"🎯 Correctos: {sum(r['es_correcto'] for r in resultados_pruebas)}/{len(resultados_pruebas)}")

# =============================================================================
# 12.3 VISUALIZAR PREDICCIONES
# =============================================================================

def visualizar_prediccion(resultado, dataset_by_player):
    """Visualiza comparación entre jugador real y predicho"""
    if 'error' in resultado:
        print(resultado['error'])
        return
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    status = '✅ CORRECTO' if resultado['es_correcto'] else '❌ INCORRECTO'
    color = 'green' if resultado['es_correcto'] else 'red'
    fig.suptitle(f"Predicción: {status}", fontsize=16, fontweight='bold', color=color)
    
    # Imágenes del jugador real
    real_pairs = dataset_by_player[resultado['jugador_real']][:3]
    for i, (board, heatmap) in enumerate(real_pairs):
        if i < 3:
            board_display = board / 255.0 if board.max() > 1 else board
            axes[0, i].imshow(board_display)
            axes[0, i].set_title(f"Real: {resultado['jugador_real']}")
            axes[0, i].axis('off')
    
    # Imágenes del jugador predicho
    pred_pairs = dataset_by_player[resultado['prediccion']][:3]
    for i, (board, heatmap) in enumerate(pred_pairs):
        if i < 3:
            board_display = board / 255.0 if board.max() > 1 else board
            axes[1, i].imshow(board_display)
            title = f"Predicho: {resultado['prediccion']}\nDist: {resultado['distancia_minima']:.3f}"
            axes[1, i].set_title(title)
            axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.show()

print("\n" + "=" * 80)
print("🖼️  VISUALIZACIÓN DE PREDICCIONES")
print("=" * 80)

for i, resultado in enumerate(resultados_pruebas[:3], 1):
    print(f"\nPredicción {i}:")
    visualizar_prediccion(resultado, dataset_by_player)

# =============================================================================
# 13. GUARDAR MODELO Y METADATOS
# =============================================================================

print("\n" + "=" * 80)
print("💾 GUARDANDO MODELO Y METADATOS")
print("=" * 80)

# Crear directorio para modelos
model_save_dir = '/home/andrewyernau/dev/jupyter/models'
os.makedirs(model_save_dir, exist_ok=True)

timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
model_name = f'dual_channel_cnn_v0002b_{timestamp}'
model_path = os.path.join(model_save_dir, model_name)
os.makedirs(model_path, exist_ok=True)

# 13.1 Preparar metadatos
metadata = {
    'model_info': {
        'architecture': 'Dual Channel CNN (ResNet50)',
        'version': '0002b',
        'framework': 'TensorFlow/Keras',
        'embedding_dim': 128,
        'date_trained': datetime.datetime.now().isoformat(),
    },
    'training_info': {
        'epochs': EPOCHS,
        'batch_size': BATCH_SIZE,
        'optimizer': 'Adam',
        'learning_rate': 0.001,
        'loss_function': 'TripletSemiHardLoss',
        'final_train_loss': float(history.history['loss'][-1]),
        'final_val_loss': float(history.history['val_loss'][-1]),
    },
    'dataset_info': {
        'name': 'Chess Stylometry Dataset',
        'train_triplets': len(train_triplets),
        'val_triplets': len(val_triplets),
        'num_players': len(dataset_by_player),
        'total_positions': sum(len(pairs) for pairs in dataset_by_player.values()),
    },
    'performance_metrics': {
        'test_accuracy': float(accuracy),
        'avg_confidence': float(confianza_promedio),
        'threshold': 1.0,
    },
    'preprocessing': {
        'image_size': str(IMG_SHAPE),
        'normalization': 'ResNet preprocessing',
    }
}

# 13.2 Guardar modelo
embedding_model.save(os.path.join(model_path, 'embedding_model.keras'))
embedding_model.save_weights(os.path.join(model_path, 'model_weights.h5'))

# 13.3 Guardar metadatos
with open(os.path.join(model_path, 'metadata.json'), 'w') as f:
    json.dump(metadata, f, indent=2)

# 13.4 Guardar historial
history_serializable = {
    'train_loss': [float(x) for x in history.history['loss']],
    'val_loss': [float(x) for x in history.history['val_loss']],
}
with open(os.path.join(model_path, 'training_history.json'), 'w') as f:
    json.dump(history_serializable, f, indent=2)

# 13.5 Guardar configuración
model_config = {
    'framework': 'TensorFlow/Keras',
    'architecture': 'Dual Channel Siamese Network',
    'backbone': 'ResNet50',
    'embedding_dim': 128,
    'image_shape': IMG_SHAPE,
}
with open(os.path.join(model_path, 'model_config.json'), 'w') as f:
    json.dump(model_config, f, indent=2)

# 13.6 Guardar resultados de pruebas
resultados_serializables = []
for r in resultados_pruebas:
    r_copy = r.copy()
    r_copy['top_5_matches'] = [
        {k: float(v) if isinstance(v, (np.floating, float)) else v 
         for k, v in match.items()} 
        for match in r_copy['top_5_matches']
    ]
    r_copy['distancia_minima'] = float(r_copy['distancia_minima'])
    r_copy['confianza'] = float(r_copy['confianza'])
    resultados_serializables.append(r_copy)

with open(os.path.join(model_path, 'test_results.json'), 'w') as f:
    json.dump(resultados_serializables, f, indent=2)

print(f"\n✅ Modelo guardado en: {model_path}")
print(f"\n📁 Archivos guardados:")
print(f"  ✓ embedding_model.keras")
print(f"  ✓ model_weights.h5")
print(f"  ✓ metadata.json")
print(f"  ✓ training_history.json")
print(f"  ✓ model_config.json")
print(f"  ✓ test_results.json")

# =============================================================================
# 13.7 FUNCIÓN PARA CARGAR EL MODELO
# =============================================================================

def cargar_modelo_guardado(model_path):
    """Carga un modelo guardado con todos sus metadatos"""
    from tensorflow import keras
    
    with open(os.path.join(model_path, 'metadata.json'), 'r') as f:
        metadata = json.load(f)
    
    with open(os.path.join(model_path, 'model_config.json'), 'r') as f:
        config = json.load(f)
    
    model = keras.models.load_model(os.path.join(model_path, 'embedding_model.keras'))
    
    with open(os.path.join(model_path, 'training_history.json'), 'r') as f:
        history = json.load(f)
    
    with open(os.path.join(model_path, 'test_results.json'), 'r') as f:
        test_results = json.load(f)
    
    return {
        'model': model,
        'metadata': metadata,
        'history': history,
        'test_results': test_results,
        'config': config
    }

# Ejemplo de carga
print("\n" + "=" * 80)
print("🔄 EJEMPLO DE CARGA DEL MODELO")
print("=" * 80)

loaded = cargar_modelo_guardado(model_path)

print(f"\n✅ Modelo cargado exitosamente")
print(f"  📦 Arquitectura: {loaded['metadata']['model_info']['architecture']}")
print(f"  🏷️  Versión: {loaded['metadata']['model_info']['version']}")
print(f"  📅 Fecha: {loaded['metadata']['model_info']['date_trained']}")
print(f"  🎯 Test Accuracy: {loaded['metadata']['performance_metrics']['test_accuracy']:.2%}")

print("\n" + "=" * 80)
print("✅ PROCESO COMPLETADO")
print("=" * 80)
```

## Resumen

Este código añade al notebook:

1. ✅ **Testing con jugadores random** (10 pruebas)
2. ✅ **Visualización de predicciones** (correctas e incorrectas)
3. ✅ **Guardado completo del modelo** con metadatos
4. ✅ **Función de carga** para reutilización
5. ✅ **Métricas de performance** (accuracy, confianza)

**Umbral usado**: `threshold=1.0` (ajusta según tus necesidades basándote en la distribución de distancias del notebook)

**Variables que usa**:
- `embedding_model` - Tu modelo entrenado
- `dataset_by_player` - Diccionario de jugadores
- `train_triplets`, `val_triplets` - Para metadatos
- `history` - Historial de Keras
- `IMG_SHAPE`, `EPOCHS`, `BATCH_SIZE` - Parámetros

¡Copia este código al final de tu notebook y ejecútalo! 🚀
