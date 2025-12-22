#!/usr/bin/env python3
"""
Pipeline de Estilometría con fragmentación en bloques.

Genera imágenes de tablero y heatmaps divididos en bloques temporales.
Estructura jerárquica por evento, con validación de PGNs existentes.
"""

from pathlib import Path
from typing import Dict, Optional, List, Tuple
import multiprocessing as mp


# Número de bloques en que dividir el rango de movimientos
NUM_BLOCKS = 3


class ChessStylometryPipelineV002:
    """Pipeline completo con fragmentación en bloques."""
    
    def __init__(
        self,
        massive_pgn: Path,
        output_base: Path,
        event_type: str,
        num_players: int = 20,
        games_per_player: int = 30,
        min_threshold: float = 0.7,
        move_start: int = 15,
        move_end: int = 30,
        num_blocks: int = NUM_BLOCKS,
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
            Tipo de evento a procesar
        num_players : int
            Número de jugadores a extraer (0 = todos)
        games_per_player : int
            Partidas objetivo por jugador
        min_threshold : float
            Umbral mínimo (0.0-1.0)
        move_start : int
            Jugada inicial del rango a analizar (ej: 15 = jugada 15 de ambos colores)
        move_end : int
            Jugada final del rango a analizar (ej: 30 = jugada 30 de ambos colores)
        num_blocks : int
            Número de bloques en que dividir el rango (default: 3)
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
        self.move_start = move_start
        self.move_end = move_end
        self.num_blocks = num_blocks
        self.compression = compression_factor
        self.use_relative = use_relative_time
        
        # Convertir jugadas completas a movimientos individuales (halfmoves)
        # Jugada 15 = movimientos 29-30 (blancas+negras)
        # Fórmula: movimiento = jugada * 2 - 1 (para blancas de esa jugada)
        start_halfmove = (move_start * 2) - 1
        end_halfmove = move_end * 2
        
        # Calcular bloques automáticamente dividiendo el rango
        total_halfmoves = end_halfmove - start_halfmove + 1
        halfmoves_per_block = total_halfmoves // num_blocks
        
        self.blocks = []
        for i in range(num_blocks):
            block_start = start_halfmove + (i * halfmoves_per_block)
            block_end = block_start + halfmoves_per_block - 1
            block_label = f"block_{i+1:02d}"
            self.blocks.append((block_start, block_end, block_label))
        
        # Estructura jerárquica
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
        print(f"\n{'='*70}")
        print(f"Pipeline de Estilometría - Bloques")
        print(f"{'='*70}\n")
        print(f"Evento: {self.event_type}")
        print(f"Bloques configurados: {len(self.blocks)}")
        for start, end, label in self.blocks:
            print(f"  - {label}: movimientos {start}-{end}")
        print(f"Output: {self.event_dir}\n")
        
        # FASE 0: Verificar PGNs existentes
        existing_pgns = []
        if self.player_pgns_dir.exists():
            existing_pgns = [f for f in self.player_pgns_dir.glob("*.pgn") if f.stat().st_size > 0]
        
        if existing_pgns:
            print(f"{'='*70}")
            print(f"Jugadores ya extraídos (saltando fase 1)")
            print(f"{'='*70}\n")
            print(f"Encontrados {len(existing_pgns)} jugadores en: {self.player_pgns_dir}")
            print(f"Usando datos existentes...\n")
            
            # Cargar jugadores desde archivos existentes
            from extract_player_games_by_event_parallel import PlayerGameStats
            valid_players = {}
            
            for pgn_file in existing_pgns:
                player_name = pgn_file.stem.replace('_', ' ')
                valid_players[player_name] = PlayerGameStats(player_name=player_name)
                print(f"  - {player_name}")
            
            print(f"\n{'='*70}")
            print(f"Limpiando imágenes previas...")
            print(f"{'='*70}\n")
            
            # Borrar imágenes previas de tableros y heatmaps
            import shutil
            if self.board_images_dir.exists():
                shutil.rmtree(self.board_images_dir)
                print(f"  - Eliminado: {self.board_images_dir}")
            if self.heatmap_images_dir.exists():
                shutil.rmtree(self.heatmap_images_dir)
                print(f"  - Eliminado: {self.heatmap_images_dir}")
            
            print(f"\n{'='*70}\n")
        else:
            # FASE 1: Extracción de partidas por jugador
            print(f"{'='*70}")
            print(f"FASE 1: EXTRACCIÓN PARALELA DE PARTIDAS")
            print(f"{'='*70}\n")
            
            # Importar desde obsolete ya que no hemos recreado esto todavía
            import sys
            obsolete_path = Path(__file__).parent / "obsolete_v001" / "scripts"
            if str(obsolete_path) not in sys.path:
                sys.path.insert(0, str(obsolete_path))
            
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
                print("ERROR: No se encontraron jugadores con suficientes partidas")
                return {}
            
            # Validar que se encontraron los jugadores solicitados
            if self.num_players > 0 and len(valid_players) < self.num_players:
                print(f"ADVERTENCIA: Solo se encontraron {len(valid_players)} de {self.num_players} jugadores solicitados")
            
            extractor.save_player_games(self.player_pgns_dir, valid_players)
        
        # FASE 2: Generación de imágenes con bloques
        print(f"{'='*70}")
        print(f"FASE 2: GENERACIÓN DE IMÁGENES CON BLOQUES")
        print(f"{'='*70}\n")
        
        board_stats = self._generate_images_parallel(valid_players)
        
        # Resumen final
        print(f"\n{'='*70}")
        print(f"Resumen Final")
        print(f"{'='*70}\n")
        
        total_boards = sum(s['boards'] for s in board_stats.values())
        total_heats = sum(s['heatmaps'] for s in board_stats.values())
        
        print(f"Evento: {self.event_type}")
        print(f"Jugadores procesados: {len(board_stats)}")
        print(f"Bloques por partida: {len(self.blocks)}")
        print(f"Total tableros: {total_boards}")
        print(f"Total heatmaps: {total_heats}")
        total_games = sum(s['games_processed'] for s in board_stats.values()) if board_stats else 0
        print(f"Partidas completas: {total_games}")
        print(f"Ratio sincronización: {total_heats}/{total_boards} = "
              f"{total_heats/total_boards*100:.1f}%" if total_boards > 0 else "N/A")
        print(f"\nDirectorio de salida: {self.event_dir}")
        print(f"{'='*70}\n")
        
        return board_stats
    
    def _generate_images_parallel(self, valid_players: Dict) -> Dict:
        """
        Genera imágenes en paralelo con fragmentación en bloques.
        
        Parameters
        ----------
        valid_players : dict
            Jugadores válidos con sus partidas
        
        Returns
        -------
        dict
            Estadísticas por jugador
        """
        from generate_images_blocks import generate_images_blocks
        
        print(f"CPU cores disponibles: {mp.cpu_count()}")
        print(f"Generación con bloques para {len(valid_players)} jugadores")
        print(f"Imágenes por partida: {len(self.blocks) * 2} "
              f"({len(self.blocks)} tableros + {len(self.blocks)} heatmaps)\n")
        
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
                self.blocks,
                self.compression,
                self.use_relative,
                False  # verbose=False en paralelo
            ))
        
        # Ejecutar en paralelo
        with mp.Pool(mp.cpu_count()) as pool:
            results = pool.starmap(generate_images_blocks, tasks)
        
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
            
            print(f"  - {player_name}:")
            print(f"      Partidas procesadas: {stats['games_processed']}")
            print(f"      Tableros: {stats['boards_created']} "
                  f"({stats['boards_created']//len(self.blocks)} partidas x {len(self.blocks)} bloques)")
            print(f"      Heatmaps: {stats['heatmaps_created']}")
            
            # Verificar sincronización
            if stats['boards_created'] != stats['heatmaps_created']:
                print(f"      ADVERTENCIA: Desincronización - boards={stats['boards_created']}, "
                      f"heatmaps={stats['heatmaps_created']}")
        
        return board_stats


def discover_available_events(pgn_path: Path, max_games: int = 50000) -> Dict:
    """
    Descubre eventos disponibles en el PGN.
    
    Parameters
    ----------
    pgn_path : Path
        Archivo PGN
    max_games : int
        Máximo de partidas a escanear
    
    Returns
    -------
    dict
        Eventos encontrados con estadísticas
    """
    import sys
    obsolete_path = Path(__file__).parent / "obsolete_v001" / "scripts"
    if str(obsolete_path) not in sys.path:
        sys.path.insert(0, str(obsolete_path))
    
    from event_discovery_parallel import ParallelEventDiscovery
    
    discovery = ParallelEventDiscovery(pgn_path, max_workers=8)
    events = discovery.discover_events(max_games=max_games)
    discovery.print_summary(events)
    
    return events


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Pipeline de estilometría V002 con bloques'
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
    parser.add_argument('--move-start', type=int, default=15,
                       help='Movimiento inicial del rango')
    parser.add_argument('--move-end', type=int, default=30,
                       help='Movimiento final del rango')
    parser.add_argument('--num-blocks', type=int, default=3,
                       help='Número de bloques')
    parser.add_argument('--compression', type=int, default=1,
                       help='Factor de compresión')
    parser.add_argument('--relative', action='store_true',
                       help='Tiempos relativos en heatmaps')
    
    # Modo de descubrimiento
    parser.add_argument('--discover', action='store_true',
                       help='Solo descubrir eventos disponibles')
    
    args = parser.parse_args()
    
    if args.discover:
        discover_available_events(args.pgn_file)
    else:
        pipeline = ChessStylometryPipelineV002(
            massive_pgn=args.pgn_file,
            output_base=args.output,
            event_type=args.event,
            num_players=args.num_players,
            games_per_player=args.games_per_player,
            min_threshold=args.min_threshold,
            move_start=args.move_start,
            move_end=args.move_end,
            num_blocks=args.num_blocks,
            compression_factor=args.compression,
            use_relative_time=args.relative
        )
        
        stats = pipeline.run()
