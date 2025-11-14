# ⚠️ Errores Comunes y Soluciones

## Error: `NameError: name 'test_dataset' is not defined`

**Causa**: Tu notebook NO usa objetos `Dataset`, usa diccionarios y listas.

**Solución**: Usa las variables correctas:
```python
# ❌ INCORRECTO
jugadores = list(test_dataset.person_ids)

# ✅ CORRECTO
jugadores = list(dataset_by_player.keys())
```

---

## Error: Acceso a atributos inexistentes

**Causa**: `dataset_by_player` es un diccionario, no un objeto con métodos.

**Solución**:
```python
# ❌ INCORRECTO
for pid, img in dataset:
    ...

# ✅ CORRECTO
for pid, pairs in dataset_by_player.items():
    for board, heatmap in pairs:
        ...
```

---

## Error: Modelo no definido o nombre incorrecto

**Causa**: El modelo en tu notebook se llama `embedding_model`, no `model`.

**Solución**:
```python
# ❌ INCORRECTO
predictions = model.predict([board, heatmap])

# ✅ CORRECTO
predictions = embedding_model.predict([board, heatmap])
```

---

## Error: PyTorch vs TensorFlow

**Causa**: Tu notebook usa TensorFlow/Keras, no PyTorch.

**Solución**:
```python
# ❌ INCORRECTO (PyTorch)
import torch
with torch.no_grad():
    emb = model(input)

# ✅ CORRECTO (TensorFlow/Keras)
import tensorflow as tf
emb = embedding_model.predict([board, heatmap], verbose=0)
```

---

## Error: Formato de entrada al modelo

**Causa**: El modelo espera dos entradas (board y heatmap) con batch dimension.

**Solución**:
```python
# ❌ INCORRECTO
emb = embedding_model.predict(board)

# ✅ CORRECTO
board_batch = np.expand_dims(board, axis=0)
heatmap_batch = np.expand_dims(heatmap, axis=0)
emb = embedding_model.predict([board_batch, heatmap_batch])
```

---

## Error: Threshold muy bajo o muy alto

**Causa**: El umbral debe ajustarse a la distribución de distancias de tu modelo.

**Recomendación**:
1. Ejecuta el análisis de distancias que ya tienes en el notebook
2. Observa la media de distancias positivas vs negativas
3. Ajusta el threshold entre esos valores

```python
# En tu notebook ya tienes:
# pos_dists (distancias de pares del mismo jugador)
# neg_dists (distancias de pares de diferentes jugadores)

# Observa estos valores:
print(f"Distancia media positiva: {pos_dists.mean():.4f}")
print(f"Distancia media negativa: {neg_dists.mean():.4f}")

# Usa un threshold intermedio, por ejemplo:
threshold = (pos_dists.mean() + neg_dists.mean()) / 2
print(f"Threshold recomendado: {threshold:.4f}")
```

---

## Error: No hay suficientes jugadores para test

**Causa**: Dataset pequeño o no se separaron jugadores para test.

**Solución**:
```python
# Verificar cantidad de jugadores
print(f"Total jugadores: {len(dataset_by_player)}")

# Si son pocos, usar menos pruebas
num_pruebas = min(10, len(dataset_by_player))
jugadores_a_probar = random.sample(list(dataset_by_player.keys()), num_pruebas)
```

---

## Error: Jugador con pocas imágenes

**Causa**: Algunos jugadores solo tienen 1 imagen.

**Solución**:
```python
# Filtrar jugadores con suficientes imágenes
jugadores_validos = [
    p for p, pairs in dataset_by_player.items() 
    if len(pairs) >= 2
]

jugadores_a_probar = random.sample(jugadores_validos, min(num_pruebas, len(jugadores_validos)))
```

---

## Error: JSON serialization con numpy

**Causa**: Numpy floats no son serializables directamente a JSON.

**Solución**:
```python
# ❌ INCORRECTO
json.dump({'value': np.float32(0.5)}, f)

# ✅ CORRECTO
json.dump({'value': float(0.5)}, f)

# O para diccionarios completos:
def convert_to_serializable(obj):
    if isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    return obj
```

---

## Error: Modelo no se guarda correctamente

**Causa**: Versiones incompatibles o formato incorrecto.

**Solución**:
```python
# Guardar en múltiples formatos por seguridad

# Formato nuevo (Keras 3)
embedding_model.save('model.keras')

# Formato legacy (si hay problemas)
embedding_model.save('model.h5')

# Solo pesos (más ligero)
embedding_model.save_weights('weights.h5')
```

---

## Error: Visualización no muestra imágenes

**Causa**: Valores fuera de rango [0, 1] o formato incorrecto.

**Solución**:
```python
# Normalizar para visualización
if board.max() > 1:
    board_display = board / 255.0
else:
    board_display = board

# Asegurar rango correcto
board_display = np.clip(board_display, 0, 1)

plt.imshow(board_display)
```

---

## Checklist antes de ejecutar el código

- [ ] ✅ Variables correctas: `embedding_model`, `dataset_by_player`, `history`
- [ ] ✅ Imports necesarios: `random`, `json`, `datetime`, `os`, `numpy`, `matplotlib`
- [ ] ✅ Threshold ajustado según tu distribución de distancias
- [ ] ✅ Directorio de guardado existe: `/home/andrewyernau/dev/jupyter/models`
- [ ] ✅ Suficientes jugadores con >= 2 imágenes para testing
- [ ] ✅ GPU con suficiente memoria (si aplica)

---

## Debugging rápido

Si algo falla, ejecuta esto para verificar el estado:

```python
print("=" * 80)
print("VERIFICACIÓN DEL ENTORNO")
print("=" * 80)

# Verificar modelo
print(f"✓ Modelo: {type(embedding_model)}")
print(f"  - Inputs: {embedding_model.input_shape}")
print(f"  - Output: {embedding_model.output_shape}")

# Verificar dataset
print(f"\n✓ Dataset:")
print(f"  - Total jugadores: {len(dataset_by_player)}")
print(f"  - Total posiciones: {sum(len(p) for p in dataset_by_player.values())}")
print(f"  - Jugadores con >= 2 imgs: {sum(1 for p in dataset_by_player.values() if len(p) >= 2)}")

# Verificar un sample
sample_player = list(dataset_by_player.keys())[0]
sample_board, sample_heatmap = dataset_by_player[sample_player][0]
print(f"\n✓ Sample data:")
print(f"  - Board shape: {sample_board.shape}")
print(f"  - Heatmap shape: {sample_heatmap.shape}")
print(f"  - Board range: [{sample_board.min():.2f}, {sample_board.max():.2f}]")

# Verificar historial
print(f"\n✓ Training history:")
print(f"  - Epochs: {len(history.history['loss'])}")
print(f"  - Final train loss: {history.history['loss'][-1]:.4f}")
print(f"  - Final val loss: {history.history['val_loss'][-1]:.4f}")

print("\n" + "=" * 80)
```

---

## ¿Más problemas?

Si encuentras otros errores:

1. **Lee el mensaje de error completo** - indica la línea exacta
2. **Verifica los nombres de variables** - usa las de tu notebook
3. **Comprueba los tipos de datos** - `type(variable)`
4. **Revisa las dimensiones** - `variable.shape` para arrays
5. **Usa verbose=0** en `.predict()` para reducir output

¡La mayoría de errores son por nombres de variables incorrectos! 🔍
