#!/usr/bin/env python3
"""
Utilidad de sincronización de imágenes del pipeline.

Asegura que cada partida tenga AMBOS tableros Y heatmaps.
Elimina archivos huérfanos y renumera para mantener consistencia.
"""

from pathlib import Path
from typing import Dict, Set
import re


def extract_game_id(filename: str) -> str:
    """
    Extrae el game_id de un nombre de archivo.
    
    Examples:
        game_0001_move_15.png -> 0001
        game_0042_white.png -> 0042
    """
    match = re.search(r'game_(\d+)', filename)
    return match.group(1) if match else None


def synchronize_player_images(board_dir: Path, heatmap_dir: Path, verbose: bool = True):
    """
    Sincroniza tableros y heatmaps de un jugador.
    
    Elimina archivos huérfanos donde:
    - Hay tableros pero no heatmap
    - Hay heatmap pero no tableros
    
    Parameters
    ----------
    board_dir : Path
        Directorio de tableros del jugador
    heatmap_dir : Path
        Directorio de heatmaps del jugador
    verbose : bool
        Mostrar mensajes detallados
    
    Returns
    -------
    dict
        Estadísticas de sincronización
    """
    if not board_dir.exists() or not heatmap_dir.exists():
        return {'error': 'Directorios no existen'}
    
    # Encontrar game_ids de tableros
    board_games = set()
    for board_file in board_dir.glob('game_*.png'):
        game_id = extract_game_id(board_file.name)
        if game_id:
            board_games.add(game_id)
    
    # Encontrar game_ids de heatmaps
    heat_games = set()
    for heat_file in heatmap_dir.glob('game_*.png'):
        game_id = extract_game_id(heat_file.name)
        if game_id:
            heat_games.add(game_id)
    
    # Identificar huérfanos
    orphan_boards = board_games - heat_games
    orphan_heats = heat_games - board_games
    
    if verbose:
        print(f"\n  Game IDs en tableros: {len(board_games)}")
        print(f"  Game IDs en heatmaps: {len(heat_games)}")
        print(f"  Tableros huérfanos (sin heatmap): {len(orphan_boards)}")
        print(f"  Heatmaps huérfanos (sin tablero): {len(orphan_heats)}")
    
    # Eliminar huérfanos
    deleted_boards = 0
    deleted_heats = 0
    
    for game_id in orphan_boards:
        # Eliminar todos los tableros de esta partida
        for board_file in board_dir.glob(f'game_{game_id}_*.png'):
            board_file.unlink()
            deleted_boards += 1
    
    for game_id in orphan_heats:
        # Eliminar todos los heatmaps de esta partida
        for heat_file in heatmap_dir.glob(f'game_{game_id}*.png'):
            heat_file.unlink()
            deleted_heats += 1
    
    if verbose and (deleted_boards > 0 or deleted_heats > 0):
        print(f"  Eliminados: {deleted_boards} tableros, {deleted_heats} heatmaps")
    
    # Games sincronizados
    synced_games = board_games & heat_games
    
    if verbose:
        print(f"  ✓ Partidas sincronizadas: {len(synced_games)}")
    
    return {
        'total_boards': len(board_games),
        'total_heats': len(heat_games),
        'orphan_boards': len(orphan_boards),
        'orphan_heats': len(orphan_heats),
        'deleted_boards': deleted_boards,
        'deleted_heats': deleted_heats,
        'synced_games': len(synced_games)
    }


def synchronize_all_players(output_base: Path, verbose: bool = True):
    """
    Sincroniza tableros y heatmaps de TODOS los jugadores.
    
    Parameters
    ----------
    output_base : Path
        Directorio base de salida del pipeline
    verbose : bool
        Mostrar mensajes detallados
    
    Returns
    -------
    dict
        Estadísticas por jugador
    """
    board_base = output_base / 'board_images'
    heat_base = output_base / 'heatmap_images'
    
    if not board_base.exists() or not heat_base.exists():
        print("⚠ Directorios de imágenes no existen")
        return {}
    
    # Encontrar todos los jugadores
    players = []
    for player_dir in board_base.iterdir():
        if player_dir.is_dir():
            players.append(player_dir.name)
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"SINCRONIZACIÓN DE IMÁGENES")
        print(f"{'='*70}")
        print(f"Jugadores encontrados: {len(players)}\n")
    
    # Sincronizar cada jugador
    results = {}
    total_deleted_boards = 0
    total_deleted_heats = 0
    
    for player in players:
        if verbose:
            print(f"{player}:")
        
        board_dir = board_base / player
        heat_dir = heat_base / player
        
        stats = synchronize_player_images(board_dir, heat_dir, verbose)
        results[player] = stats
        
        total_deleted_boards += stats.get('deleted_boards', 0)
        total_deleted_heats += stats.get('deleted_heats', 0)
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"RESUMEN DE SINCRONIZACIÓN")
        print(f"{'='*70}")
        print(f"Total eliminados: {total_deleted_boards} tableros, {total_deleted_heats} heatmaps")
        print(f"✓ Sincronización completada\n")
    
    return results


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Sincronizar tableros y heatmaps')
    parser.add_argument('output_base', type=Path, help='Directorio base de salida')
    parser.add_argument('--quiet', action='store_true', help='Modo silencioso')
    
    args = parser.parse_args()
    
    results = synchronize_all_players(args.output_base, verbose=not args.quiet)
    
    # Mostrar resumen
    total_synced = sum(r.get('synced_games', 0) for r in results.values())
    print(f"\n✓ Total de partidas sincronizadas: {total_synced}")
