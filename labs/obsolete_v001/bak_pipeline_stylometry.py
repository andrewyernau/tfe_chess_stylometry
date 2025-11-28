#!/usr/bin/env python3
"""
Pipeline integrado para estilometría conductual de ajedrez.

Combina extracción de partidas por jugador, generación de imágenes de tablero
y mapas de calor de tiempos de decisión.
"""

import argparse
from pathlib import Path
from typing import List, Optional
import sys

# Importar módulos del proyecto
from extract_player_games import PlayerGameExtractor
from generate_decision_heatmaps import process_pgn_to_heatmaps
from parse_games_to_images import process_pgn_file


class ChessStylometryPipeline:
    """Pipeline completo de procesamiento para estilometría de ajedrez."""
    
    def __init__(
        self,
        massive_pgn: Path,
        output_base: Path,
        num_players: int = 20,
        games_per_player: int = 30,
        min_threshold: float = 0.7,
        player_timeout: float = 300.0,
        start_move: int = 15,
        end_move: int = 23,
        compression_factor: int = 2,
        seed: Optional[int] = None
    ):
        """
        Parameters
        ----------
        massive_pgn : Path
            Archivo PGN masivo (ej: lichess_db.pgn)
        output_base : Path
            Directorio base de salida
        num_players : int
            Número de jugadores a extraer aleatoriamente
        games_per_player : int
            Número objetivo de partidas por jugador
        min_threshold : float
            Umbral mínimo de partidas (0.0-1.0)
        player_timeout : float
            Timeout local por jugador (se reinicia al encontrar partida)
        start_move : int
            Movimiento inicial para imágenes
        end_move : int
            Movimiento final para imágenes
        compression_factor : int
            Factor de compresión de tableros
        seed : int, optional
            Seed para reproducibilidad
        """
        self.massive_pgn = massive_pgn
        self.output_base = output_base
        self.num_players = num_players
        self.games_per_player = games_per_player
        self.min_threshold = min_threshold
        self.player_timeout = player_timeout
        self.start_move = start_move
        self.end_move = end_move
        self.compression = compression_factor
        self.seed = seed
        # Normalize and resolve common relative paths so notebooks and CLI both work
        # Accept strings as well as Path objects
        try:
            self.massive_pgn = Path(self.massive_pgn)
        except Exception:
            self.massive_pgn = Path(str(self.massive_pgn))

        if not self.massive_pgn.exists():
            # Try resolving relative 
            candidate = (Path(__file__).resolve().parent / self.massive_pgn)
            if candidate.exists():
                print(f"Resolved PGN path relative to pipeline file: {candidate}")
                self.massive_pgn = candidate
            else:
                # Try resolving relative to repo root (one level up from labs/)
                repo_candidate = (Path(__file__).resolve().parents[1] / self.massive_pgn)
                if repo_candidate.exists():
                    print(f"Resolved PGN path relative to repository root: {repo_candidate}")
                    self.massive_pgn = repo_candidate
                else:
                    # Leave as-is
                    print(
                        f"Warning: PGN path '{self.massive_pgn}' does not exist in the current working directory.\n"
                        f"Tried: cwd -> {Path.cwd()},\n"
                        f"       pipeline file parent -> {candidate},\n"
                        f"       repo root -> {repo_candidate}\n"
                        f"If your PGN lives under 'labs/dataset', pass an absolute path or use Path('..') from the notebook."
                    )
        
        # Directorios de salida
        self.player_pgns_dir = output_base / "player_pgns"
        self.board_images_dir = output_base / "board_images"
        self.heatmap_images_dir = output_base / "heatmap_images"
    
    def run(self) -> dict:
        """
        Ejecuta el pipeline completo.
        
        Returns
        -------
        dict
            Estadísticas de ejecución por jugador
        """
        print(f"\n{'#'*70}")
        print(f"# PIPELINE DE ESTILOMETRÍA CONDUCTUAL DE AJEDREZ")
        print(f"{'#'*70}\n")
        
        # FASE 1: Extracción de partidas por jugador
        print(f"\n{'='*70}")
        print(f"FASE 1: EXTRACCIÓN DE PARTIDAS POR JUGADOR")
        print(f"{'='*70}\n")
        
        extractor = PlayerGameExtractor(
            pgn_path=self.massive_pgn,
            num_players=self.num_players,
            games_per_player=self.games_per_player,
            min_games_threshold=self.min_threshold,
            player_timeout=self.player_timeout
        )
        
        valid_players = extractor.extract_games()
        
        if not valid_players:
            print("❌ No se encontraron jugadores con suficientes partidas")
            return {}
        
        # Guardar PGNs por jugador
        extractor.save_player_games(self.player_pgns_dir, valid_players)
        
        # FASE 2-3: Generación de imágenes EN PARALELO
        print(f"\n{'='*70}")
        print(f"FASE 2-3: GENERACIÓN DE IMÁGENES (PARALELO)")
        print(f"{'='*70}")
        
        import multiprocessing as mp
        print(f"CPU cores disponibles: {mp.cpu_count()}")
        print(f"Usando procesamiento paralelo para máximo rendimiento\n")
        
        # Procesar tableros y heatmaps en paralelo
        from parse_games_to_images import process_pgn_file
        from generate_decision_heatmaps import process_pgn_to_heatmaps
        
        # Procesar en paralelo usando starmap (evita problemas de pickle)
        board_stats = {}
        
        print("\n🚀 Generando tableros en paralelo...")
        board_tasks = []
        for player in valid_players.keys():
            player_pgn = self.player_pgns_dir / f"{player.replace(' ', '_')}.pgn"
            player_output_board = self.board_images_dir / player.replace(' ', '_')
            board_tasks.append((
                player_pgn,
                player_output_board,
                self.start_move,
                self.end_move,
                self.compression
            ))
        
        # Procesar tableros
        with mp.Pool(mp.cpu_count()) as pool:
            board_results = pool.starmap(process_pgn_file, board_tasks)
        
        # Asociar resultados con jugadores
        for i, player in enumerate(valid_players.keys()):
            boards = board_results[i]
            board_stats[player] = {'boards': boards}
            print(f"  ✓ {player}: {boards} tableros")
        
        print("\n🚀 Generando heatmaps en paralelo...")
        heatmap_tasks = []
        for player in valid_players.keys():
            player_pgn = self.player_pgns_dir / f"{player.replace(' ', '_')}.pgn"
            player_output_heat = self.heatmap_images_dir / player.replace(' ', '_')
            heatmap_tasks.append((
                player_pgn,
                player_output_heat,
                self.start_move,
                self.end_move,
                (8, 8),  # grid_size (tablero 8x8)
                224,     # output_size (224x224 píxeles)
                True     # use_relative
            ))
        
        # Procesar heatmaps
        with mp.Pool(mp.cpu_count()) as pool:
            heatmap_results = pool.starmap(process_pgn_to_heatmaps, heatmap_tasks)
        
        # Asociar resultados con jugadores
        for i, player in enumerate(valid_players.keys()):
            heatmaps = heatmap_results[i]
            board_stats[player]['heatmaps'] = heatmaps
            print(f"  ✓ {player}: {heatmaps} heatmaps")
        
        # NUEVO: Sincronizar tableros y heatmaps
        print(f"\n{'='*70}")
        print(f"SINCRONIZACIÓN DE IMÁGENES")
        print(f"{'='*70}")
        
        from synchronize_images import synchronize_all_players
        sync_results = synchronize_all_players(self.output_base, verbose=True)
        
        # Actualizar estadísticas con datos sincronizados
        for player, sync_stats in sync_results.items():
            if player in board_stats:
                board_stats[player]['synced_games'] = sync_stats.get('synced_games', 0)
                board_stats[player]['deleted_boards'] = sync_stats.get('deleted_boards', 0)
                board_stats[player]['deleted_heats'] = sync_stats.get('deleted_heats', 0)
        
        # RESUMEN FINAL
        print(f"\n{'#'*70}")
        print(f"# RESUMEN FINAL DEL PIPELINE")
        print(f"{'#'*70}\n")
        
        print(f"Jugadores procesados: {len(valid_players)}")
        print(f"\nDetalle por jugador:")
        
        for player, stats in board_stats.items():
            total_games = valid_players[player].total_games
            boards = stats.get('boards', 0)
            heatmaps = stats.get('heatmaps', 0)
            
            print(f"\n  {player}:")
            print(f"    • Partidas extraídas: {total_games}")
            print(f"    • Imágenes de tablero: {boards}")
            print(f"    • Mapas de calor: {heatmaps}")
        
        print(f"\nDirectorios de salida:")
        print(f"  • PGNs por jugador: {self.player_pgns_dir}")
        print(f"  • Imágenes de tablero: {self.board_images_dir}")
        print(f"  • Mapas de calor: {self.heatmap_images_dir}")
        
        print(f"\n{'#'*70}\n")
        
        return board_stats


def main():
    parser = argparse.ArgumentParser(
        description="Pipeline integrado de estilometría conductual de ajedrez",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplo de uso:
  python pipeline_stylometry.py \\
    --pgn-file dataset/generated/lichess_db.pgn \\
    --players Magnus DrNykterstein Hikaru \\
    --min-games 20 \\
    --max-games 50 \\
    --output output/stylometry_results
        """
    )
    
    parser.add_argument(
        "--pgn-file",
        type=Path,
        default=Path("dataset/generated/lichess_db.pgn"),
        help="Archivo PGN masivo"
    )
    
    parser.add_argument(
        "--num-players",
        type=int,
        default=20,
        help="Número de jugadores a extraer aleatoriamente (default: 20)"
    )
    
    parser.add_argument(
        "--games-per-player",
        type=int,
        default=30,
        help="Número objetivo de partidas por jugador (default: 30)"
    )
    
    parser.add_argument(
        "--min-threshold",
        type=float,
        default=0.7,
        help="Umbral mínimo de partidas (0.0-1.0) (default: 0.7)"
    )
    
    parser.add_argument(
        "--player-timeout",
        type=float,
        default=300.0,
        help="Timeout local por jugador en segundos (se reinicia al encontrar partida) (default: 300)"
    )
    
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output"),
        help="Directorio base de salida (default: output)"
    )
    
    parser.add_argument(
        "--start-move",
        type=int,
        default=15,
        help="Movimiento inicial (default: 15)"
    )
    
    parser.add_argument(
        "--end-move",
        type=int,
        default=23,
        help="Movimiento final (default: 23)"
    )
    
    parser.add_argument(
        "--compression",
        type=int,
        default=2,
        help="Factor de compresión de tableros (default: 2)"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Seed para reproducibilidad (default: aleatorio)"
    )
    
    args = parser.parse_args()
    
    # Validaciones
    if not args.pgn_file.exists():
        print(f"❌ Error: Archivo PGN no encontrado: {args.pgn_file}")
        return 1
    
    # Establecer seed
    if args.seed is not None:
        import random
        random.seed(args.seed)
        print(f"🎲 Seed establecido: {args.seed}")
    
    # Crear pipeline
    pipeline = ChessStylometryPipeline(
        massive_pgn=args.pgn_file,
        output_base=args.output,
        num_players=args.num_players,
        games_per_player=args.games_per_player,
        min_threshold=args.min_threshold,
        player_timeout=args.player_timeout,
        start_move=args.start_move,
        end_move=args.end_move,
        compression_factor=args.compression,
        seed=args.seed
    )
    
    # Ejecutar pipeline
    try:
        stats = pipeline.run()
        return 0 if stats else 1
    except Exception as e:
        print(f"\n❌ Error en el pipeline: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
