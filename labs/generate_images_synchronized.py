#!/usr/bin/env python3
"""
Generación sincronizada de imágenes de tablero y heatmaps.

Garantiza que cada partida procesada genera AMBOS:
- Imágenes de tablero
- Heatmap de decisiones

Si una partida falla en cualquiera, se cancelan AMBOS.
"""

import chess.pgn
from pathlib import Path
from typing import Dict, Tuple, Optional
import numpy as np
import cv2


def generate_images_synchronized(
    pgn_path: Path,
    output_board: Path,
    output_heat: Path,
    start_move: int,
    end_move: int,
    compression_factor: int = 1,
    use_relative_time: bool = True,
    verbose: bool = True
) -> Dict[str, int]:
    """
    Genera tableros y heatmaps de forma sincronizada.
    
    Garantías:
    - Si partida X genera tableros → genera heatmaps
    - Si partida X falla en tableros → NO genera heatmaps  
    - Mismo número de imágenes para tableros y heatmaps
    
    Parameters
    ----------
    pgn_path : Path
        Archivo PGN del jugador
    output_board : Path
        Directorio de salida para tableros
    output_heat : Path
        Directorio de salida para heatmaps
    start_move : int
        Movimiento inicial a procesar
    end_move : int
        Movimiento final a procesar
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
    # Importar módulos necesarios
    from parse_games_to_images import (
        board_to_png_array,
        extract_board_sequence,
        overlay_temporal_sequence
    )
    from generate_decision_heatmaps import (
        DecisionTimeExtractor,
        DecisionHeatmapGenerator
    )
    
    # Crear directorios
    output_board.mkdir(parents=True, exist_ok=True)
    output_heat.mkdir(parents=True, exist_ok=True)
    
    # Inicializar generador de heatmaps
    heatmap_gen = DecisionHeatmapGenerator(
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
        print(f"\nGeneración sincronizada:")
        print(f"  PGN: {pgn_path.name}")
        print(f"  Movimientos: {start_move}-{end_move}")
        print(f"  Compresión: {compression_factor}x\n")
    
    with open(pgn_path) as pgn_file:
        game_num = 0
        
        while True:
            game = chess.pgn.read_game(pgn_file)
            if game is None:
                break
            
            game_num += 1
            
            try:
                # 1. VALIDAR: suficientes movimientos
                moves = list(game.mainline_moves())
                if len(moves) < end_move:
                    stats['games_skipped'] += 1
                    continue
                
                # 2. GENERAR TABLERO (uno solo por partida)
                board_created = _generate_boards_for_game(
                    game, output_board, game_num,
                    start_move, end_move, compression_factor
                )
                
                if board_created == 0:
                    stats['games_skipped'] += 1
                    continue
                
                stats['boards_created'] += 1
                
                # 3. GENERAR HEATMAP
                heatmap_created = _generate_heatmap_for_game(
                    game, output_heat, game_num,
                    start_move, end_move, heatmap_gen
                )
                
                if heatmap_created:
                    stats['heatmaps_created'] += 1
                else:
                    # Si falla heatmap, ELIMINAR tablero creado
                    _cleanup_game_boards(output_board, game_num)
                    stats['boards_created'] -= 1
                    stats['games_skipped'] += 1
                    continue
                
                stats['games_processed'] += 1
                
            except Exception as e:
                # Error en cualquier parte: limpiar TODO
                _cleanup_game_images(output_board, output_heat, game_num)
                stats['errors'] += 1
                if verbose:
                    print(f"  ⚠ Partida {game_num} error: {e}")
                continue
            
            # Progreso
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


def _generate_boards_for_game(
    game: chess.pgn.Game,
    output_dir: Path,
    game_num: int,
    start_move: int,
    end_move: int,
    compression: int
) -> int:
    """
    Genera UNA SOLA imagen de tablero por partida con transparencia temporal.
    
    La imagen muestra:
    - Movimientos antiguos: muy transparentes (apenas visibles)
    - Movimiento más reciente: 100% visible
    
    Returns
    -------
    int
        1 si se creó correctamente, 0 si falló
    """
    from parse_games_to_images import overlay_temporal_sequence
    
    moves = list(game.mainline_moves())
    
    # Generar secuencia de tableros desde start_move hasta end_move
    board_sequence = []
    temp_board = game.board()
    
    for move_idx, move in enumerate(moves[:end_move]):
        temp_board.push(move)
        move_num = move_idx + 1
        
        if move_num >= start_move:
            board_sequence.append(temp_board.copy())
    
    if not board_sequence:
        return 0
    
    # Aplicar transparencia temporal: el último board es el más visible
    temporal_img = overlay_temporal_sequence(
        board_sequence,
        compression_factor=compression,
        board_size=400
    )
    
    # Guardar UNA SOLA imagen por partida
    output_file = output_dir / f"game_{game_num:04d}.png"
    cv2.imwrite(str(output_file), temporal_img)
    
    return 1


def _generate_heatmap_for_game(
    game: chess.pgn.Game,
    output_dir: Path,
    game_num: int,
    start_move: int,
    end_move: int,
    heatmap_gen
) -> bool:
    """
    Genera heatmap para una partida.
    
    Returns
    -------
    bool
        True si se creó correctamente
    """
    try:
        # Generar heatmap
        heatmap_img = heatmap_gen.generate_heatmap(
            game,
            start_move=start_move,
            end_move=end_move
        )
        
        if heatmap_img is None:
            return False
        
        # Guardar
        output_file = output_dir / f"game_{game_num:04d}.png"
        cv2.imwrite(str(output_file), heatmap_img)
        
        return True
        
    except Exception:
        return False


def _cleanup_game_boards(output_dir: Path, game_num: int):
    """Elimina el tablero de una partida."""
    board_file = output_dir / f"game_{game_num:04d}.png"
    if board_file.exists():
        board_file.unlink()


def _cleanup_game_images(output_board: Path, output_heat: Path, game_num: int):
    """Elimina todas las imágenes (tableros + heatmaps) de una partida."""
    # Eliminar tableros
    _cleanup_game_boards(output_board, game_num)
    
    # Eliminar heatmap
    heatmap_file = output_heat / f"game_{game_num:04d}.png"
    if heatmap_file.exists():
        heatmap_file.unlink()


def process_player_synchronized(
    player_pgn: Path,
    output_base: Path,
    start_move: int,
    end_move: int,
    compression: int = 1,
    use_relative: bool = True
) -> Tuple[int, int]:
    """
    Procesa un jugador completo de forma sincronizada.
    
    Parameters
    ----------
    player_pgn : Path
        Archivo PGN del jugador
    output_base : Path
        Directorio base (contendrá board_images/ y heatmap_images/)
    start_move : int
        Movimiento inicial
    end_move : int
        Movimiento final
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
    
    stats = generate_images_synchronized(
        player_pgn,
        output_board,
        output_heat,
        start_move,
        end_move,
        compression,
        use_relative,
        verbose=True
    )
    
    return stats['boards_created'], stats['heatmaps_created']


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Generación sincronizada de imágenes')
    parser.add_argument('pgn_file', type=Path, help='Archivo PGN')
    parser.add_argument('--output-board', type=Path, required=True, help='Dir tableros')
    parser.add_argument('--output-heat', type=Path, required=True, help='Dir heatmaps')
    parser.add_argument('--start-move', type=int, default=15, help='Movimiento inicial')
    parser.add_argument('--end-move', type=int, default=23, help='Movimiento final')
    parser.add_argument('--compression', type=int, default=1, help='Factor compresión')
    parser.add_argument('--relative', action='store_true', help='Tiempos relativos')
    
    args = parser.parse_args()
    
    stats = generate_images_synchronized(
        args.pgn_file,
        args.output_board,
        args.output_heat,
        args.start_move,
        args.end_move,
        args.compression,
        args.relative
    )
    
    print(f"\n✓ Completado: {stats['boards_created']} tableros, "
          f"{stats['heatmaps_created']} heatmaps")
