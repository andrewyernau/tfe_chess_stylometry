#!/usr/bin/env python3
"""
Versión del pipeline optimizada para ejecutar desde notebooks Jupyter.

Esta versión evita problemas de multiprocessing anidado que ocurren
cuando se ejecuta desde notebooks.
"""

import sys
from pathlib import Path
from typing import Optional

# Importar módulos del proyecto
from extract_player_games import PlayerGameExtractor
from generate_decision_heatmaps import process_pgn_to_heatmaps
from parse_games_to_images import process_pgn_file


class ChessStylometryPipelineNotebook:
    """Pipeline para notebooks - evita multiprocessing anidado."""
    
    def __init__(
        self,
        massive_pgn: Path,
        output_base: Path,
        num_players: int = 20,
        games_per_player: int = 30,
        min_threshold: float = 0.7,
        start_move: int = 15,
        end_move: int = 23,
        compression_factor: int = 2,
        seed: Optional[int] = None,
        max_workers: int = 36
    ):
        """
        Parameters
        ----------
        max_workers : int
            Número máximo de workers paralelos (default: 36)
            Reduce este número si experimentas problemas de memoria
        """
        self.massive_pgn = Path(massive_pgn)
        self.output_base = Path(output_base)
        self.num_players = num_players
        self.games_per_player = games_per_player
        self.min_threshold = min_threshold
        self.start_move = start_move
        self.end_move = end_move
        self.compression = compression_factor
        self.seed = seed
        self.max_workers = max_workers
        
        # Directorios de salida
        self.player_pgns_dir = output_base / "player_pgns"
        self.board_images_dir = output_base / "board_images"
        self.heatmap_images_dir = output_base / "heatmap_images"
    
    def run(self) -> dict:
        """
        Ejecuta el pipeline completo con paralelización optimizada para notebooks.
        
        Returns
        -------
        dict
            Estadísticas de ejecución por jugador
        """
        print(f"\n{'#'*70}")
        print(f"# PIPELINE DE ESTILOMETRÍA (MODO NOTEBOOK)")
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
            player_timeout=600.0  # Mayor timeout para ejecución en notebook
        )
        
        valid_players = extractor.extract_games()
        
        if not valid_players:
            print("❌ No se encontraron jugadores con suficientes partidas")
            return {}
        
        # Guardar PGNs por jugador
        extractor.save_player_games(self.player_pgns_dir, valid_players)
        
        # FASE 2-3: Generación de imágenes (procesamiento secuencial para evitar problemas)
        print(f"\n{'='*70}")
        print(f"FASE 2-3: GENERACIÓN DE IMÁGENES")
        print(f"{'='*70}")
        print(f"Procesando {len(valid_players)} jugadores...\n")
        
        board_stats = {}
        
        for i, player in enumerate(valid_players.keys(), 1):
            player_pgn = self.player_pgns_dir / f"{player.replace(' ', '_')}.pgn"
            player_output_board = self.board_images_dir / player.replace(' ', '_')
            player_output_heat = self.heatmap_images_dir / player.replace(' ', '_')
            
            print(f"[{i}/{len(valid_players)}] Procesando {player}...")
            
            # Generar tableros
            boards = process_pgn_file(
                pgn_path=player_pgn,
                output_dir=player_output_board,
                start_move=self.start_move,
                end_move=self.end_move,
                compression_factor=self.compression
            )
            
            # Generar heatmaps
            heatmaps = process_pgn_to_heatmaps(
                pgn_path=player_pgn,
                output_dir=player_output_heat,
                start_move=self.start_move,
                end_move=self.end_move,
                grid_size=(8, 8),
                cell_size=24,
                use_relative=True
            )
            
            board_stats[player] = {
                'boards': boards,
                'heatmaps': heatmaps
            }
            
            print(f"  ✓ {boards} tableros, {heatmaps} heatmaps")
        
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


def run_pipeline_from_notebook(
    pgn_file: str = "dataset/generated/lichess_db.pgn",
    output_dir: str = "output/stylometry_results",
    num_players: int = 5,
    games_per_player: int = 10,
    seed: int = 42,
    max_workers: int = 36
):
    """
    Función helper para ejecutar el pipeline desde un notebook.
    
    Ejemplo de uso en notebook:
    
    ```python
    from pipeline_stylometry_notebook import run_pipeline_from_notebook
    
    stats = run_pipeline_from_notebook(
        pgn_file="dataset/generated/lichess_db.pgn",
        num_players=5,
        games_per_player=10,
        seed=42
    )
    ```
    
    Parameters
    ----------
    pgn_file : str
        Ruta al archivo PGN masivo
    output_dir : str
        Directorio de salida
    num_players : int
        Número de jugadores a procesar
    games_per_player : int
        Partidas por jugador
    seed : int
        Seed para reproducibilidad
    max_workers : int
        Máximo de workers paralelos
    
    Returns
    -------
    dict
        Estadísticas del pipeline
    """
    import random
    
    if seed is not None:
        random.seed(seed)
    
    pipeline = ChessStylometryPipelineNotebook(
        massive_pgn=Path(pgn_file),
        output_base=Path(output_dir),
        num_players=num_players,
        games_per_player=games_per_player,
        min_threshold=0.7,
        start_move=15,
        end_move=23,
        compression_factor=2,
        seed=seed,
        max_workers=max_workers
    )
    
    return pipeline.run()


if __name__ == "__main__":
    # Ejemplo de uso directo
    stats = run_pipeline_from_notebook(
        num_players=3,
        games_per_player=5
    )
    print("Pipeline completado:", stats)
