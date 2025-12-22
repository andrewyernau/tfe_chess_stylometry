#!/usr/bin/env python3
"""
Generador de datos eficiente en memoria para multi-channel CNN.

En lugar de cargar todas las imágenes en RAM, este generador:
- Carga solo un batch a la vez
- Lee imágenes desde disco bajo demanda
- Reduce el uso de memoria de GB a MB
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict
from functools import lru_cache
import keras


# Funciones de caché globales (fuera de la clase para que funcione lru_cache)
@lru_cache(maxsize=4096)
def _cached_load_board(path: str, img_shape: Tuple[int, int]) -> np.ndarray:
    """Carga y cachea una imagen de tablero"""
    board = cv2.imread(path)
    board = cv2.cvtColor(board, cv2.COLOR_BGR2RGB)
    board = cv2.resize(board, img_shape) / 255.0
    return board


@lru_cache(maxsize=4096)
def _cached_load_heat(path: str, img_shape: Tuple[int, int]) -> np.ndarray:
    """Carga y cachea un heatmap"""
    heat = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    heat = cv2.resize(heat, img_shape) / 255.0
    return np.expand_dims(heat, axis=-1)


class MultiChannelDataGenerator(keras.utils.Sequence):
    """
    Generador de datos para CNN con múltiples bloques.
    
    Carga imágenes bajo demanda para evitar colapsos de memoria.
    """
    
    def __init__(
        self,
        event_dir: Path,
        players: List[str],
        num_blocks: int,
        img_shape: Tuple[int, int] = (192, 192),
        batch_size: int = 32,
        shuffle: bool = True,
        **kwargs
    ):
        """
        Parameters
        ----------
        event_dir : Path
            Directorio del evento con las imágenes
        players : list
            Lista de nombres de jugadores
        num_blocks : int
            Número de bloques por partida
        img_shape : tuple
            Dimensiones (alto, ancho) de las imágenes
        batch_size : int
            Tamaño del batch
        shuffle : bool
            Si mezclar los datos al inicio de cada época
        """
        super().__init__(**kwargs)
        self.event_dir = event_dir
        self.players = players
        self.num_blocks = num_blocks
        self.img_shape = img_shape
        self.batch_size = batch_size
        self.shuffle = shuffle
        
        # Mapeo de jugadores a IDs
        self.player_to_id = {p: idx for idx, p in enumerate(players)}
        
        # Descubrir todas las partidas válidas
        self.samples = self._discover_samples()
        print(f"Generador inicializado: {len(self.samples)} partidas, {len(players)} jugadores")
        
        # Índices para shuffling
        self.indexes = np.arange(len(self.samples))
        if self.shuffle:
            np.random.shuffle(self.indexes)
    
    def _discover_samples(self) -> List[Tuple[str, str]]:
        """
        Descubre todas las partidas completas (con todos los bloques).
        
        Returns
        -------
        list
            Lista de tuplas (player, game_id)
        """
        samples = []
        
        for player in self.players:
            player_board_dir = self.event_dir / 'board_images' / player
            player_heat_dir = self.event_dir / 'heatmap_images' / player
            
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
                for block_idx in range(self.num_blocks):
                    block_label = f"block_{block_idx+1:02d}"
                    board_file = player_board_dir / f"game_{game_id}_{block_label}.png"
                    heat_file = player_heat_dir / f"game_{game_id}_{block_label}.png"
                    
                    if not board_file.exists() or not heat_file.exists():
                        blocks_complete = False
                        break
                
                if blocks_complete:
                    samples.append((player, game_id))
        
        return samples
    
    def __len__(self) -> int:
        """Número de batches por época"""
        return int(np.ceil(len(self.samples) / self.batch_size))
    
    def __getitem__(self, index: int) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
        """
        Genera un batch de datos.
        
        Parameters
        ----------
        index : int
            Índice del batch
        
        Returns
        -------
        tuple
            (inputs_dict, labels) donde inputs_dict tiene claves como
            'board_input_block_01', 'heat_input_block_01', etc.
        """
        # Obtener índices del batch
        batch_indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        
        # Cargar muestras del batch
        batch_samples = [self.samples[i] for i in batch_indexes]
        
        # Generar datos
        return self._load_batch(batch_samples)
    
    def _load_batch(self, batch_samples: List[Tuple[str, str]]) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
        """
        Carga un batch de imágenes desde disco.
        
        Parameters
        ----------
        batch_samples : list
            Lista de tuplas (player, game_id)
        
        Returns
        -------
        tuple
            (inputs_dict, labels)
        """
        batch_size = len(batch_samples)
        
        # Inicializar arrays para cada canal
        board_batches = {
            f'board_input_block_{i+1:02d}': np.zeros((batch_size, *self.img_shape, 3), dtype=np.float32)
            for i in range(self.num_blocks)
        }
        heat_batches = {
            f'heat_input_block_{i+1:02d}': np.zeros((batch_size, *self.img_shape, 1), dtype=np.float32)
            for i in range(self.num_blocks)
        }
        labels = np.zeros(batch_size, dtype=np.int32)
        
        # Cargar cada muestra
        for i, (player, game_id) in enumerate(batch_samples):
            player_board_dir = self.event_dir / 'board_images' / player
            player_heat_dir = self.event_dir / 'heatmap_images' / player
            
            # Cargar todos los bloques
            for block_idx in range(self.num_blocks):
                block_label = f"block_{block_idx+1:02d}"
                board_file = player_board_dir / f"game_{game_id}_{block_label}.png"
                heat_file = player_heat_dir / f"game_{game_id}_{block_label}.png"
                
                # Cargar tablero (RGB) con caché
                board = _cached_load_board(str(board_file), self.img_shape)
                board_batches[f'board_input_block_{block_idx+1:02d}'][i] = board
                
                # Cargar heatmap (Grayscale) con caché
                heat = _cached_load_heat(str(heat_file), self.img_shape)
                heat_batches[f'heat_input_block_{block_idx+1:02d}'][i] = heat
            
            # Label
            labels[i] = self.player_to_id[player]
        
        # Combinar diccionarios
        inputs_dict = {**board_batches, **heat_batches}
        
        return inputs_dict, labels
    
    def on_epoch_end(self):
        """Mezclar índices al final de cada época"""
        if self.shuffle:
            np.random.shuffle(self.indexes)
    
    def get_sample_count(self) -> int:
        """Retorna el número total de muestras"""
        return len(self.samples)
    
    def get_player_to_id(self) -> Dict[str, int]:
        """Retorna el mapeo de jugadores a IDs"""
        return self.player_to_id


def create_train_val_generators(
    event_dir: Path,
    players: List[str],
    num_blocks: int,
    img_shape: Tuple[int, int] = (192, 192),
    batch_size: int = 32,
    validation_split: float = 0.2,
    seed: int = 314
) -> Tuple[MultiChannelDataGenerator, MultiChannelDataGenerator]:
    """
    Crea generadores de entrenamiento y validación.
    
    Parameters
    ----------
    event_dir : Path
        Directorio del evento
    players : list
        Lista de jugadores
    num_blocks : int
        Número de bloques
    img_shape : tuple
        Dimensiones de imágenes
    batch_size : int
        Tamaño del batch
    validation_split : float
        Proporción de datos para validación
    seed : int
        Semilla para reproducibilidad
    
    Returns
    -------
    tuple
        (train_generator, val_generator)
    """
    np.random.seed(seed)
    
    # Crear generador completo
    full_gen = MultiChannelDataGenerator(
        event_dir=event_dir,
        players=players,
        num_blocks=num_blocks,
        img_shape=img_shape,
        batch_size=batch_size,
        shuffle=False
    )
    
    # Dividir samples
    total_samples = full_gen.get_sample_count()
    val_size = int(total_samples * validation_split)
    
    indices = np.arange(total_samples)
    np.random.shuffle(indices)
    
    train_indices = indices[val_size:]
    val_indices = indices[:val_size]
    
    # Crear generadores train/val
    train_samples = [full_gen.samples[i] for i in train_indices]
    val_samples = [full_gen.samples[i] for i in val_indices]
    
    train_gen = MultiChannelDataGenerator(
        event_dir=event_dir,
        players=players,
        num_blocks=num_blocks,
        img_shape=img_shape,
        batch_size=batch_size,
        shuffle=True
    )
    train_gen.samples = train_samples
    train_gen.indexes = np.arange(len(train_samples))
    
    val_gen = MultiChannelDataGenerator(
        event_dir=event_dir,
        players=players,
        num_blocks=num_blocks,
        img_shape=img_shape,
        batch_size=batch_size,
        shuffle=False
    )
    val_gen.samples = val_samples
    val_gen.indexes = np.arange(len(val_samples))
    
    print(f"Train: {len(train_samples)} muestras, {len(train_gen)} batches")
    print(f"Val: {len(val_samples)} muestras, {len(val_gen)} batches")
    
    return train_gen, val_gen


if __name__ == '__main__':
    # Ejemplo de uso
    from pathlib import Path
    
    event_dir = Path("dataset/by_event/Tata_Steel_Masters_2025")
    players = ["Carlsen_Magnus", "Nakamura_Hikaru"]
    
    # Crear generador
    gen = MultiChannelDataGenerator(
        event_dir=event_dir,
        players=players,
        num_blocks=15,
        img_shape=(192, 192),
        batch_size=4
    )
    
    print(f"\nGenerador creado:")
    print(f"  Total muestras: {gen.get_sample_count()}")
    print(f"  Batches: {len(gen)}")
    
    # Probar un batch
    inputs, labels = gen[0]
    print(f"\nPrimer batch:")
    print(f"  Inputs: {list(inputs.keys())}")
    print(f"  Board shape: {inputs['board_input_block_01'].shape}")
    print(f"  Heat shape: {inputs['heat_input_block_01'].shape}")
    print(f"  Labels: {labels.shape}")
