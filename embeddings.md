# Embeddings en estilometría de ajedrez (actualizado a notebook 104)

## 1) Qué representa un embedding

Un embedding es un vector denso (L2-normalizado) que resume el estilo de una partida para una ventana temporal concreta.
En el notebook 104, cada embedding se genera a partir de una secuencia de jugadas del jugador objetivo:

- tableros RGB por jugada,
- heatmaps de tiempo de decisión por jugada,
- fusión temporal + proyección a 256D.

## 2) Qué NO valida un embedding

- **No** se considera válida la prueba "si reconstruye imagen, entonces embedding bueno" como criterio principal.
- La reconstrucción puede ser útil como diagnóstico auxiliar, pero el objetivo real es **separabilidad estilométrica**.

## 3) Criterios correctos de calidad

Para identificación de jugador, se evalúa con partidas no vistas (hold-out):

1. **Top-1 / Top-3** por centroide más cercano.
2. **kNN** sobre embeddings de train como baseline métrico adicional.
3. Distancia media a centroide propio vs centroide ajeno más cercano.
4. Margen efectivo: `mean(other_nearest) - mean(own)` (debe ser positivo).

## 4) Inferencia de una partida arbitraria

Flujo recomendado:

1. Extraer secuencia con el mismo preprocesado del entrenamiento.
2. Obtener embedding de la partida.
3. Calcular distancia a los centroides de jugadores conocidos.
4. Clasificar por distancia mínima (y opcionalmente reportar Top-k).

Si no se sabe qué color corresponde al jugador objetivo, evaluar ambos lados (`white` y `black`) y comparar consistencia.

## 5) Errores típicos que degradan rendimiento

- mezclar jugadas del rival dentro de la señal del jugador,
- no normalizar perspectiva por color,
- usar pipeline de imagen distinto entre train e inferencia,
- evaluar solo en train/val sin hold-out real,
- confiar en una única métrica agregada.

## 6) Checklist mínimo antes de reportar benchmark

- [ ] Hold-out separado temporalmente de train.
- [ ] Top-1 y Top-3 reportados.
- [ ] Matriz de confusión por jugador.
- [ ] Margen own/other positivo.
- [ ] Comparación contra baseline simple (kNN o centroide).
