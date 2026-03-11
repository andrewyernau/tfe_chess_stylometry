# TODO técnico (benchmark 104)

## Completado

- [x] Crear notebook `104_chess_siamese_robust.ipynb` con pipeline más claro y auditable.
- [x] Corregir generación de muestras para enfocarse en jugadas del jugador objetivo.
- [x] Añadir normalización de perspectiva por color (white/black).
- [x] Sustituir tripletes estáticos por entrenamiento métrico batch-hard.
- [x] Implementar evaluación hold-out con métricas Top-k + kNN + análisis de margen.

## Siguiente iteración recomendada

- [ ] Ejecutar corrida larga completa (sin interrupciones) y guardar curva por época.
- [ ] Añadir ablation study: board-only vs heat-only vs multimodal.
- [ ] Probar múltiples prototipos por jugador (sub-centers) en vez de un solo centroide.
- [ ] Calibrar scores/distancias para convertir salida en confianza interpretable.
- [ ] Añadir protocolo open-set (rechazo de jugador desconocido).

## Notas

Las decisiones de arquitectura y evaluación para esta fase están documentadas dentro del notebook 104 y en `embeddings.md`.
