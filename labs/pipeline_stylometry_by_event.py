#!/usr/bin/env python3
"""
Pipeline de Estilometría con Clasificación por Eventos.

Versión mejorada que:
- Clasifica por tipo de evento (Bullet, Blitz, etc.)
- Genera imágenes de forma sincronizada (garantiza consistencia)
- Lee PGN completo (sin límite de tiempo)
- Estructura jerárquica por evento
"""

from pathlib import Path
from typing import Dict, Optional
import multiprocessing as mp
from functools import partial


class ChessStylometryPipelineByEvent:
    """Pipeline completo con clasificación por eventos."""
    
    def __init__(
        self,
        massive_pgn: Path,
        output_base: Path,
        event_type: str,
        num_players: int = 20,
        games_per_player: int = 30,
        min_threshold: float = 0.7,
        start_move: int = 15,
        end_move: int = 23,
        compression_factor: int = 1,
        use_relative_time: bool = True
    ):
        """
        Parameters
        ----------
        massive_pgn : Path
            Archivo PGN masivo
        output_base : Path
            Directorio base de salida
        event_type : str
            Tipo de evento a procesar (ej: "Rated Bullet game")
        num_players : int
            Número de jugadores a extraer (0 = todos)
        games_per_player : int
            Partidas objetivo por jugador
        min_threshold : float
            Umbral mínimo (0.0-1.0)
        start_move : int
            Movimiento inicial para imágenes
        end_move : int
            Movimiento final para imágenes
        compression_factor : int
            Factor de compresión de tableros
        use_relative_time : bool
            Usar tiempos relativos en heatmaps
        """
        self.massive_pgn = massive_pgn
        self.output_base = output_base
        self.event_type = event_type
        self.num_players = num_players
        self.games_per_player = games_per_player
        self.min_threshold = min_threshold
        self.start_move = start_move
        self.end_move = end_move
        self.compression = compression_factor
        self.use_relative = use_relative_time
        
        # Estructura jerárquica: output/events/Event_Name/
        event_safe = event_type.replace(" ", "_").replace("/", "_")
        self.event_dir = output_base / "events" / event_safe
        self.player_pgns_dir = self.event_dir / "player_pgns"
        self.board_images_dir = self.event_dir / "board_images"
        self.heatmap_images_dir = self.event_dir / "heatmap_images"
    
    def run(self) -> Dict:
        """
        Ejecuta el pipeline completo.
        
        Returns
        -------
        dict
            Estadísticas de ejecución por jugador
        """
        print(f"\n{'#'*70}")
        print(f"# PIPELINE DE ESTILOMETRÍA POR EVENTO")
        print(f"{'#'*70}\n")
        print(f"Evento: {self.event_type}")
        print(f"Output: {self.event_dir}\n")
        
        # VERIFICAR si ya existen PGNs de jugadores
        existing_pgns = list(self.player_pgns_dir.glob("*.pgn")) if self.player_pgns_dir.exists() else []
        
        if existing_pgns:
            print(f"{'='*70}")
            print(f"✓ JUGADORES YA EXTRAÍDOS (SALTANDO FASE 1)")
            print(f"{'='*70}\n")
            print(f"Encontrados {len(existing_pgns)} jugadores en: {self.player_pgns_dir}")
            print(f"Saltando extracción de partidas (usar datos existentes)\n")
            
            # Cargar jugadores desde archivos existentes
            from extract_player_games_by_event_parallel import PlayerGameStats
            valid_players = {}
            
            for pgn_file in existing_pgns:
                player_name = pgn_file.stem.replace('_', ' ')
                valid_players[player_name] = PlayerGameStats(player_name=player_name)
                print(f"  ✓ {player_name}")
            
            print(f"\n{'='*70}\n")
        else:
            # FASE 1: Extracción de partidas por jugador (filtrado por evento)
            print(f"{'='*70}")
            print(f"FASE 1: EXTRACCIÓN PARALELA DE PARTIDAS (FILTRADO POR EVENTO)")
            print(f"{'='*70}\n")
            
            from extract_player_games_by_event_parallel import ParallelPlayerExtractorByEvent
            
            extractor = ParallelPlayerExtractorByEvent(
                pgn_path=self.massive_pgn,
                event_type=self.event_type,
                num_players=self.num_players,
                games_per_player=self.games_per_player,
                min_games_threshold=self.min_threshold
            )
            
            valid_players = extractor.extract_games()
            
            if not valid_players:
                print("❌ No se encontraron jugadores con suficientes partidas")
                return {}
            
            # Guardar PGNs por jugador
            extractor.save_player_games(self.player_pgns_dir, valid_players)
        
        # FASE 2: Generación SINCRONIZADA de imágenes
        print(f"{'='*70}")
        print(f"FASE 2: GENERACIÓN SINCRONIZADA DE IMÁGENES")
        print(f"{'='*70}")
        print(f"Sincronización: Garantiza tableros ↔ heatmaps 1:1\n")
        
        board_stats = self._generate_images_parallel(valid_players)
        
        # RESUMEN FINAL
        print(f"\n{'#'*70}")
        print(f"# RESUMEN FINAL DEL PIPELINE")
        print(f"{'#'*70}\n")
        
        total_boards = sum(s['boards'] for s in board_stats.values())
        total_heats = sum(s['heatmaps'] for s in board_stats.values())
        
        print(f"Evento: {self.event_type}")
        print(f"Jugadores procesados: {len(board_stats)}")
        print(f"Total tableros: {total_boards}")
        print(f"Total heatmaps: {total_heats}")
        print(f"Ratio sincronización: {total_heats}/{total_boards//8:.0f} = "
              f"{total_heats/(total_boards//8)*100:.1f}%" if total_boards > 0 else "N/A")
        print(f"\nDirectorio de salida: {self.event_dir}")
        print(f"{'#'*70}\n")
        
        return board_stats
    
    def _generate_images_parallel(self, valid_players: Dict) -> Dict:
        """
        Genera imágenes en paralelo de forma sincronizada.
        
        Parameters
        ----------
        valid_players : dict
            Jugadores válidos con sus partidas
        
        Returns
        -------
        dict
            Estadísticas por jugador
        """
        from generate_images_synchronized import generate_images_synchronized
        
        print(f"CPU cores disponibles: {mp.cpu_count()}")
        print(f"Generación sincronizada para {len(valid_players)} jugadores\n")
        
        # Preparar tareas
        tasks = []
        for player_name in valid_players.keys():
            safe_name = player_name.replace(' ', '_').replace('/', '_')
            player_pgn = self.player_pgns_dir / f"{safe_name}.pgn"
            player_board_dir = self.board_images_dir / safe_name
            player_heat_dir = self.heatmap_images_dir / safe_name
            
            tasks.append((
                player_pgn,
                player_board_dir,
                player_heat_dir,
                self.start_move,
                self.end_move,
                self.compression,
                self.use_relative,
                False  # verbose=False en paralelo
            ))
        
        # Ejecutar en paralelo
        with mp.Pool(mp.cpu_count()) as pool:
            results = pool.starmap(generate_images_synchronized, tasks)
        
        # Procesar resultados
        board_stats = {}
        for i, player_name in enumerate(valid_players.keys()):
            stats = results[i]
            board_stats[player_name] = {
                'boards': stats['boards_created'],
                'heatmaps': stats['heatmaps_created'],
                'games_processed': stats['games_processed'],
                'games_skipped': stats['games_skipped']
            }
            
            print(f"  ✓ {player_name}:")
            print(f"      Partidas procesadas: {stats['games_processed']}")
            print(f"      Tableros: {stats['boards_created']}")
            print(f"      Heatmaps: {stats['heatmaps_created']}")
            
            # Verificar sincronización
            expected_heats = stats['games_processed']
            actual_heats = stats['heatmaps_created']
            if expected_heats != actual_heats:
                print(f"      ⚠ Desincronización: esperaba {expected_heats}, "
                      f"tiene {actual_heats}")
        
        return board_stats


def discover_available_events(pgn_path: Path, max_games: int = 50000) -> Dict:
    """
    Descubre eventos disponibles en el PGN.
    
    Helper function para notebooks.
    
    Parameters
    ----------
    pgn_path : Path
        Archivo PGN
    max_games : int
        Máximo de partidas a escanear (None = todas)
    
    Returns
    -------
    dict
        Eventos encontrados con estadísticas
    """
    from event_discovery_parallel import ParallelEventDiscovery
    
    # Usar versión paralela optimizada para RAM
    discovery = ParallelEventDiscovery(pgn_path, max_workers=8)
    events = discovery.discover_events(max_games=max_games)
    discovery.print_summary(events)
    
    return events


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Pipeline de estilometría por evento'
    )
    parser.add_argument('--pgn-file', type=Path, required=True, 
                       help='Archivo PGN masivo')
    parser.add_argument('--output', type=Path, required=True,
                       help='Directorio de salida')
    parser.add_argument('--event', required=True,
                       help='Tipo de evento')
    parser.add_argument('--num-players', type=int, default=20,
                       help='Número de jugadores (0=todos)')
    parser.add_argument('--games-per-player', type=int, default=30,
                       help='Partidas por jugador')
    parser.add_argument('--min-threshold', type=float, default=0.7,
                       help='Umbral mínimo (0.0-1.0)')
    parser.add_argument('--start-move', type=int, default=15,
                       help='Movimiento inicial')
    parser.add_argument('--end-move', type=int, default=23,
                       help='Movimiento final')
    parser.add_argument('--compression', type=int, default=1,
                       help='Factor de compresión')
    parser.add_argument('--relative', action='store_true',
                       help='Tiempos relativos en heatmaps')
    
    # Modo de descubrimiento
    parser.add_argument('--discover', action='store_true',
                       help='Solo descubrir eventos disponibles')
    
    args = parser.parse_args()
    
    if args.discover:
        # Solo descubrir eventos
        discover_available_events(args.pgn_file)
    else:
        # Ejecutar pipeline
        pipeline = ChessStylometryPipelineByEvent(
            massive_pgn=args.pgn_file,
            output_base=args.output,
            event_type=args.event,
            num_players=args.num_players,
            games_per_player=args.games_per_player,
            min_threshold=args.min_threshold,
            start_move=args.start_move,
            end_move=args.end_move,
            compression_factor=args.compression,
            use_relative_time=args.relative
        )
        
        stats = pipeline.run()
