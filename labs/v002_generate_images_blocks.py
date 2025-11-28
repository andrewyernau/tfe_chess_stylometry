#!/usr/bin/env python3
"""
Generación sincronizada de imágenes con FRAGMENTACIÓN EN BLOQUES (Versión 002).

CAMBIOS V002:
- Genera imágenes en BLOQUES de 5 movimientos
- 15 jugadas por jugador = 3 bloques por jugador
- Bloques: (15-20), (20-25), (25-30) = movimientos 15-30, 30-45, 45-60
- 6 imágenes por partida: 3 tableros + 3 heatmaps
- Mejor visualización de transparencia en cada bloque
"""

import chess.pgn
from pathlib import Path
from typing import Dict, Tuple, Optional, List
import numpy as np
import cv2
import sys

# Asegurar que el directorio labs está en el path para multiprocessing
labs_dir = Path(__file__).parent.resolve()
if str(labs_dir) not in sys.path:
    sys.path.insert(0, str(labs_dir))

# Importar módulos necesarios al nivel del módulo (antes de multiprocessing)
from parse_games_to_images import overlay_temporal_sequence
from v002_generate_decision_heatmaps import GrayscaleHeatmapGenerator


# Configuración de bloques
# Cada tupla: (start_move, end_move, label)
BLOCK_CONFIG = [
    (15, 30, "block_01"),   # Jugadas 15-30 (movimientos 15-30)
    (30, 45, "block_02"),   # Jugadas 30-45 (movimientos 30-45) 
    (45, 60, "block_03"),   # Jugadas 45-60 (movimientos 45-60)
]


def generate_images_blocks(
    pgn_path: Path,
    output_board: Path,
    output_heat: Path,
    blocks: List[Tuple[int, int, str]] = BLOCK_CONFIG,
    compression_factor: int = 1,
    use_relative_time: bool = True,
    verbose: bool = True
) -> Dict[str, int]:
    """
    Genera tableros y heatmaps fragmentados en bloques.
    
    Para cada partida genera:
    - N bloques de tablero (1 imagen por bloque)
    - N bloques de heatmap (1 imagen por bloque)
    
    Nomenclatura: game_XXXX_block_YY.png
    
    Parameters
    ----------
    pgn_path : Path
        Archivo PGN del jugador
    output_board : Path
        Directorio de salida para tableros
    output_heat : Path
        Directorio de salida para heatmaps
    blocks : List[Tuple[int, int, str]]
        Lista de bloques: (start, end, label)
    compression_factor : int
        Factor de compresión (1 = 192x192, 2 = 96x96)
    use_relative_time : bool
        Usar tiempos relativos en heatmap
    verbose : bool
        Mostrar progreso
    
    Returns
    -------
    dict
        Estadísticas de procesamiento
    """
    output_board.mkdir(parents=True, exist_ok=True)
    output_heat.mkdir(parents=True, exist_ok=True)
    
    heatmap_gen = GrayscaleHeatmapGenerator(
        grid_size=(8, 8),
        output_size=192,
        use_relative=use_relative_time
    )
    
    stats = {
        'games_processed': 0,
        'games_skipped': 0,
        'boards_created': 0,
        'heatmaps_created': 0,
        'errors': 0
    }
    
    if verbose:
        print(f"\nGeneración con BLOQUES:")
        print(f"  PGN: {pgn_path.name}")
        print(f"  Bloques por partida: {len(blocks)}")
        for start, end, label in blocks:
            print(f"    - {label}: movimientos {start}-{end}")
        print(f"  Compresión: {compression_factor}x\n")
    
    with open(pgn_path) as pgn_file:
        game_num = 0
        
        while True:
            game = chess.pgn.read_game(pgn_file)
            if game is None:
                break
            
            game_num += 1
            
            try:
                moves = list(game.mainline_moves())
                
                # Validar que tiene suficientes movimientos para el último bloque
                max_move_needed = max(end for _, end, _ in blocks)
                if len(moves) < max_move_needed:
                    stats['games_skipped'] += 1
                    continue
                
                # Procesar cada bloque
                blocks_success = []
                
                for start_move, end_move, block_label in blocks:
                    # Generar tablero para este bloque
                    board_created = _generate_board_block(
                        game, output_board, game_num, block_label,
                        start_move, end_move, compression_factor
                    )
                    
                    if not board_created:
                        # Si falla un bloque, limpiar TODO y saltar partida
                        _cleanup_game_all_blocks(
                            output_board, output_heat, game_num, blocks
                        )
                        stats['games_skipped'] += 1
                        break
                    
                    # Generar heatmap para este bloque
                    heatmap_created = _generate_heatmap_block(
                        game, output_heat, game_num, block_label,
                        start_move, end_move, heatmap_gen
                    )
                    
                    if not heatmap_created:
                        # Si falla heatmap, limpiar TODO
                        _cleanup_game_all_blocks(
                            output_board, output_heat, game_num, blocks
                        )
                        stats['games_skipped'] += 1
                        break
                    
                    blocks_success.append(block_label)
                
                # Si todos los bloques fueron exitosos
                if len(blocks_success) == len(blocks):
                    stats['games_processed'] += 1
                    stats['boards_created'] += len(blocks)
                    stats['heatmaps_created'] += len(blocks)
                
            except Exception as e:
                _cleanup_game_all_blocks(output_board, output_heat, game_num, blocks)
                stats['errors'] += 1
                if verbose:
                    print(f"  Partida {game_num} error: {e}")
                continue
            
            if verbose and game_num % 10 == 0:
                print(f"  Procesadas: {game_num} | "
                      f"Válidas: {stats['games_processed']} | "
                      f"Saltadas: {stats['games_skipped']}")
    
    if verbose:
        print(f"\n✓ Procesamiento completado:")
        print(f"  Partidas procesadas: {stats['games_processed']}")
        print(f"  Tableros creados: {stats['boards_created']}")
        print(f"  Heatmaps creados: {stats['heatmaps_created']}")
        print(f"  Partidas saltadas: {stats['games_skipped']}")
        print(f"  Errores: {stats['errors']}\n")
    
    return stats


def _generate_board_block(
    game: chess.pgn.Game,
    output_dir: Path,
    game_num: int,
    block_label: str,
    start_move: int,
    end_move: int,
    compression: int
) -> bool:
    """
    Genera imagen de tablero para un bloque específico.
    
    Aplica transparencia temporal dentro del bloque:
    - Primer movimiento del bloque: más transparente
    - Último movimiento del bloque: 100% visible
    
    Returns
    -------
    bool
        True si se creó correctamente
    """
    moves = list(game.mainline_moves())
    
    board_sequence = []
    temp_board = game.board()
    
    # Iterar hasta end_move para recoger todos los estados del bloque
    for move_idx, move in enumerate(moves):
        halfmove_num = move_idx + 1  # Numeración empieza en 1
        
        # Llegar hasta el inicio del bloque
        if halfmove_num < start_move:
            temp_board.push(move)
            continue
        
        # Detener después del final del bloque
        if halfmove_num > end_move:
            break
        
        # Guardar estado DESPUÉS de hacer el movimiento
        temp_board.push(move)
        board_sequence.append(temp_board.copy())
    
    if not board_sequence:
        return False
    
    temporal_img = overlay_temporal_sequence(
        board_sequence,
        compression_factor=compression,
        board_size=400
    )
    
    output_file = output_dir / f"game_{game_num:04d}_{block_label}.png"
    cv2.imwrite(str(output_file), temporal_img)
    
    return True


def _generate_heatmap_block(
    game: chess.pgn.Game,
    output_dir: Path,
    game_num: int,
    block_label: str,
    start_move: int,
    end_move: int,
    heatmap_gen
) -> bool:
    """
    Genera heatmap para un bloque específico.
    
    Returns
    -------
    bool
        True si se creó correctamente
    """
    try:
        heatmap_img = heatmap_gen.generate_heatmap(
            game,
            start_move=start_move,
            end_move=end_move
        )
        
        if heatmap_img is None:
            return False
        
        output_file = output_dir / f"game_{game_num:04d}_{block_label}.png"
        cv2.imwrite(str(output_file), heatmap_img)
        
        return True
        
    except Exception:
        return False


def _cleanup_game_all_blocks(
    output_board: Path,
    output_heat: Path,
    game_num: int,
    blocks: List[Tuple[int, int, str]]
):
    """Elimina todas las imágenes de todos los bloques de una partida."""
    for _, _, block_label in blocks:
        # Eliminar tablero
        board_file = output_board / f"game_{game_num:04d}_{block_label}.png"
        if board_file.exists():
            board_file.unlink()
        
        # Eliminar heatmap
        heat_file = output_heat / f"game_{game_num:04d}_{block_label}.png"
        if heat_file.exists():
            heat_file.unlink()


def process_player_blocks(
    player_pgn: Path,
    output_base: Path,
    blocks: List[Tuple[int, int, str]] = BLOCK_CONFIG,
    compression: int = 1,
    use_relative: bool = True
) -> Tuple[int, int]:
    """
    Procesa un jugador completo con fragmentación en bloques.
    
    Parameters
    ----------
    player_pgn : Path
        Archivo PGN del jugador
    output_base : Path
        Directorio base
    blocks : List[Tuple[int, int, str]]
        Configuración de bloques
    compression : int
        Factor de compresión
    use_relative : bool
        Usar tiempos relativos
    
    Returns
    -------
    tuple
        (boards_created, heatmaps_created)
    """
    player_name = player_pgn.stem
    
    output_board = output_base / 'board_images' / player_name
    output_heat = output_base / 'heatmap_images' / player_name
    
    stats = generate_images_blocks(
        player_pgn,
        output_board,
        output_heat,
        blocks,
        compression,
        use_relative,
        verbose=True
    )
    
    return stats['boards_created'], stats['heatmaps_created']


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Generación sincronizada con bloques (v002)'
    )
    parser.add_argument('pgn_file', type=Path, help='Archivo PGN')
    parser.add_argument('--output-board', type=Path, required=True, help='Dir tableros')
    parser.add_argument('--output-heat', type=Path, required=True, help='Dir heatmaps')
    parser.add_argument('--compression', type=int, default=1, help='Factor compresión')
    parser.add_argument('--relative', action='store_true', help='Tiempos relativos')
    
    args = parser.parse_args()
    
    stats = generate_images_blocks(
        args.pgn_file,
        args.output_board,
        args.output_heat,
        BLOCK_CONFIG,
        args.compression,
        args.relative
    )
    
    print(f"\n✓ Completado: {stats['boards_created']} tableros, "
          f"{stats['heatmaps_created']} heatmaps")
