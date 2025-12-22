#!/usr/bin/env python3
"""
Modelo CNN con múltiples canales independientes para estilometría de ajedrez.

Cada bloque genera 2 imágenes (tablero + heatmap).
Para N bloques: 2*N canales de entrada procesados independientemente.
"""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Force CPU execution

from keras import layers, Model
from typing import Tuple, List


def build_multi_channel_model(
    num_players: int, 
    num_blocks: int, 
    img_shape: Tuple[int, int] = (192, 192)
) -> Model:
    """
    Construye modelo con N canales independientes.
    
    Cada bloque aporta 2 canales:
    - Canal RGB (tablero del bloque)
    - Canal Grayscale (heatmap del bloque)
    
    Parameters
    ----------
    num_players : int
        Número de jugadores a clasificar
    num_blocks : int
        Número de bloques temporales
    img_shape : tuple
        Dimensiones de las imágenes (alto, ancho)
    
    Returns
    -------
    Model
        Modelo compilado listo para entrenar
    """
    total_channels = num_blocks * 2
    
    # Crear inputs para cada bloque
    board_inputs = []
    heat_inputs = []
    
    for block_idx in range(num_blocks):
        board_input = layers.Input(
            shape=(*img_shape, 3), 
            name=f'board_input_block_{block_idx+1:02d}'
        )
        heat_input = layers.Input(
            shape=(*img_shape, 1), 
            name=f'heat_input_block_{block_idx+1:02d}'
        )
        board_inputs.append(board_input)
        heat_inputs.append(heat_input)
    
    # Procesar cada canal independientemente
    board_features = []
    heat_features = []
    
    for block_idx in range(num_blocks):
        # Encoder para tablero del bloque
        x_board = layers.Conv2D(32, 3, activation='relu', padding='same')(board_inputs[block_idx])
        x_board = layers.MaxPooling2D(2)(x_board)
        x_board = layers.Conv2D(64, 3, activation='relu', padding='same')(x_board)
        x_board = layers.MaxPooling2D(2)(x_board)
        x_board = layers.Conv2D(128, 3, activation='relu', padding='same')(x_board)
        x_board = layers.GlobalAveragePooling2D()(x_board)
        board_features.append(x_board)
        
        # Encoder para heatmap del bloque
        x_heat = layers.Conv2D(32, 3, activation='relu', padding='same')(heat_inputs[block_idx])
        x_heat = layers.MaxPooling2D(2)(x_heat)
        x_heat = layers.Conv2D(64, 3, activation='relu', padding='same')(x_heat)
        x_heat = layers.MaxPooling2D(2)(x_heat)
        x_heat = layers.Conv2D(128, 3, activation='relu', padding='same')(x_heat)
        x_heat = layers.GlobalAveragePooling2D()(x_heat)
        heat_features.append(x_heat)
    
    # Fusionar todos los features
    all_features = board_features + heat_features
    merged = layers.concatenate(all_features)
    
    # Clasificador
    x = layers.Dense(512, activation='relu')(merged)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    output = layers.Dense(num_players, activation='softmax')(x)
    
    # Combinar todos los inputs
    all_inputs = board_inputs + heat_inputs
    
    model = Model(inputs=all_inputs, outputs=output)
    
    return model


def load_multi_channel_dataset(
    event_dir,
    players: List[str],
    num_blocks: int,
    img_shape: Tuple[int, int] = (192, 192)
):
    """
    Carga dataset con estructura de múltiples canales por bloque.
    
    Retorna N listas de arrays, donde N = num_blocks * 2
    
    Parameters
    ----------
    event_dir : Path
        Directorio del evento con las imágenes
    players : list
        Lista de nombres de jugadores
    num_blocks : int
        Número de bloques
    img_shape : tuple
        Dimensiones de las imágenes
    
    Returns
    -------
    tuple
        (board_channels_list, heat_channels_list, labels, player_to_id)
    """
    import cv2
    import numpy as np
    from collections import defaultdict
    
    # Lista de arrays para cada canal
    board_channels = [[] for _ in range(num_blocks)]
    heat_channels = [[] for _ in range(num_blocks)]
    labels = []
    
    player_to_id = {p: idx for idx, p in enumerate(players)}
    
    for player in players:
        player_board_dir = event_dir / 'board_images' / player
        player_heat_dir = event_dir / 'heatmap_images' / player
        
        if not player_board_dir.exists() or not player_heat_dir.exists():
            continue
        
        # Agrupar por partida
        game_ids = set()
        for f in player_board_dir.glob("game_*_block_*.png"):
            game_id = f.stem.split('_')[1]
            game_ids.add(game_id)
        
        for game_id in sorted(game_ids):
            # Verificar que existan todos los bloques
            blocks_complete = True
            for block_idx in range(num_blocks):
                block_label = f"block_{block_idx+1:02d}"
                board_file = player_board_dir / f"game_{game_id}_{block_label}.png"
                heat_file = player_heat_dir / f"game_{game_id}_{block_label}.png"
                
                if not board_file.exists() or not heat_file.exists():
                    blocks_complete = False
                    break
            
            if not blocks_complete:
                continue
            
            # Cargar todos los bloques
            for block_idx in range(num_blocks):
                block_label = f"block_{block_idx+1:02d}"
                board_file = player_board_dir / f"game_{game_id}_{block_label}.png"
                heat_file = player_heat_dir / f"game_{game_id}_{block_label}.png"
                
                # Cargar y normalizar tablero
                board = cv2.imread(str(board_file))
                board = cv2.cvtColor(board, cv2.COLOR_BGR2RGB)
                board = cv2.resize(board, img_shape) / 255.0
                board_channels[block_idx].append(board)
                
                # Cargar y normalizar heatmap
                heat = cv2.imread(str(heat_file), cv2.IMREAD_GRAYSCALE)
                heat = cv2.resize(heat, img_shape) / 255.0
                heat = np.expand_dims(heat, axis=-1)
                heat_channels[block_idx].append(heat)
            
            # Añadir label (una vez por partida completa)
            labels.append(player_to_id[player])
    
    # Convertir a arrays numpy
    board_arrays = [np.array(channel) for channel in board_channels]
    heat_arrays = [np.array(channel) for channel in heat_channels]
    labels_array = np.array(labels)
    
    return board_arrays, heat_arrays, labels_array, player_to_id


if __name__ == '__main__':
    # Ejemplo de uso
    import numpy as np
    
    num_players = 98
    num_blocks = 15
    
    model = build_multi_channel_model(num_players, num_blocks)
    
    print(f"\nModelo con {num_blocks} bloques ({num_blocks * 2} canales totales):")
    print(f"  - {num_blocks} canales RGB (tableros)")
    print(f"  - {num_blocks} canales Grayscale (heatmaps)")
    print(f"  - {num_players} jugadores a clasificar\n")
    
    model.summary()