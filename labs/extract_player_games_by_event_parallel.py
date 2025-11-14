#!/usr/bin/env python3
"""
Extractor PARALELO de jugadores por evento.

Versión optimizada que:
- Paraleliza descubrimiento de jugadores (usa todos los cores)
- Paraleliza extracción de partidas (usa todos los cores)
- Controla uso de RAM (streaming + auto-limitación)
- 10-20x más rápido que versión secuencial
"""

import chess.pgn
from pathlib import Path
from typing import Dict, List, Set, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict, Counter
import multiprocessing as mp
from functools import partial
import time
import random
import gc


def _create_player_stats():
    """Factory function for defaultdict (must be picklable)."""
    return {
        'games_as_white': 0,
        'games_as_black': 0,
        'total': 0
    }


@dataclass
class PlayerGameStats:
    """Estadísticas de partidas encontradas para un jugador."""
    player_name: str
    games_as_white: int = 0
    games_as_black: int = 0
    games: List[chess.pgn.Game] = field(default_factory=list)
    
    @property
    def total_games(self) -> int:
        return self.games_as_white + self.games_as_black
    
    def add_game(self, game: chess.pgn.Game, color: str):
        """Añade una partida y actualiza estadísticas."""
        self.games.append(game)
        if color.lower() == 'white':
            self.games_as_white += 1
        else:
            self.games_as_black += 1


def find_chunk_boundaries(pgn_path: Path, num_chunks: int) -> List[Tuple[int, int]]:
    """
    Encuentra límites de chunks alineados con partidas completas.
    Mismo algoritmo que event_discovery_parallel.py
    """
    file_size = pgn_path.stat().st_size
    chunk_size = file_size // num_chunks
    
    boundaries = []
    current_start = 0
    
    with open(pgn_path, 'rb') as f:
        for i in range(num_chunks):
            target_end = min(current_start + chunk_size, file_size)
            
            if target_end >= file_size:
                boundaries.append((current_start, file_size))
                break
            
            f.seek(target_end)
            
            while True:
                line = f.readline()
                if not line:
                    boundaries.append((current_start, file_size))
                    return boundaries
                
                if line.startswith(b'[Event '):
                    chunk_end = f.tell() - len(line)
                    boundaries.append((current_start, chunk_end))
                    current_start = chunk_end
                    break
    
    return boundaries


def discover_players_in_chunk(
    chunk_info: Tuple[int, int],
    pgn_path: Path,
    event_type: str
) -> Dict:
    """
    Descubre jugadores en un chunk del PGN.
    Versión optimizada con streaming.
    """
    start_byte, end_byte = chunk_info
    chunk_size = end_byte - start_byte
    
    player_frequency = Counter()
    games_scanned = 0
    games_of_event = 0
    
    BUFFER_SIZE = 1024 * 1024  # 1MB
    LOG_INTERVAL = 10000  # Log cada 10k partidas
    
    with open(pgn_path, 'r', encoding='utf-8', errors='ignore', buffering=BUFFER_SIZE) as f:
        f.seek(start_byte)
        
        class LimitedReader:
            def __init__(self, file_obj, max_bytes):
                self.file = file_obj
                self.bytes_read = 0
                self.max_bytes = max_bytes
            
            def readline(self):
                if self.bytes_read >= self.max_bytes:
                    return ''
                line = self.file.readline()
                self.bytes_read += len(line.encode('utf-8'))
                return line if self.bytes_read <= self.max_bytes else ''
        
        limited_reader = LimitedReader(f, chunk_size)
        
        while True:
            try:
                game = chess.pgn.read_game(limited_reader)
                if game is None:
                    break
                
                games_scanned += 1
                
                # Log progreso
                if games_scanned % LOG_INTERVAL == 0:
                    progress_mb = limited_reader.bytes_read / (1024**2)
                    chunk_mb = chunk_size / (1024**2)
                    print(f"    Chunk {start_byte//1024**2}MB: {games_scanned:,} partidas "
                          f"({progress_mb:.1f}/{chunk_mb:.1f} MB, "
                          f"{games_of_event:,} del evento)")
                
                # Filtrar por evento
                event = game.headers.get("Event", "")
                if event != event_type:
                    continue
                
                games_of_event += 1
                
                # Contar jugadores
                white = game.headers.get("White", "")
                black = game.headers.get("Black", "")
                
                if white:
                    player_frequency[white] += 1
                if black:
                    player_frequency[black] += 1
                    
            except Exception:
                continue
    
    return {
        'player_frequency': dict(player_frequency),
        'games_scanned': games_scanned,
        'games_of_event': games_of_event
    }


def extract_games_from_chunk(
    chunk_info: Tuple[int, int],
    pgn_path: Path,
    event_type: str,
    selected_players: Set[str],
    games_per_player: int
) -> Dict:
    """
    Extrae partidas de jugadores seleccionados en un chunk.
    Versión optimizada con streaming.
    """
    start_byte, end_byte = chunk_info
    chunk_size = end_byte - start_byte
    
    # Estadísticas por jugador
    player_stats = defaultdict(_create_player_stats)
    # Partidas extraídas (solo guardamos las que necesitamos)
    player_games = defaultdict(list)
    
    games_scanned = 0
    
    BUFFER_SIZE = 1024 * 1024  # 1MB
    
    with open(pgn_path, 'r', encoding='utf-8', errors='ignore', buffering=BUFFER_SIZE) as f:
        f.seek(start_byte)
        
        class LimitedReader:
            def __init__(self, file_obj, max_bytes):
                self.file = file_obj
                self.bytes_read = 0
                self.max_bytes = max_bytes
            
            def readline(self):
                if self.bytes_read >= self.max_bytes:
                    return ''
                line = self.file.readline()
                self.bytes_read += len(line.encode('utf-8'))
                return line if self.bytes_read <= self.max_bytes else ''
        
        limited_reader = LimitedReader(f, chunk_size)
        
        while True:
            try:
                game = chess.pgn.read_game(limited_reader)
                if game is None:
                    break
                
                games_scanned += 1
                
                # Filtrar por evento
                event = game.headers.get("Event", "")
                if event != event_type:
                    continue
                
                white = game.headers.get("White", "")
                black = game.headers.get("Black", "")
                
                # Extraer para blancas
                if white in selected_players:
                    if player_stats[white]['total'] < games_per_player:
                        player_stats[white]['games_as_white'] += 1
                        player_stats[white]['total'] += 1
                        player_games[white].append(('white', game))
                
                # Extraer para negras
                if black in selected_players:
                    if player_stats[black]['total'] < games_per_player:
                        player_stats[black]['games_as_black'] += 1
                        player_stats[black]['total'] += 1
                        player_games[black].append(('black', game))
                        
            except Exception:
                continue
    
    return {
        'player_stats': dict(player_stats),
        'player_games': dict(player_games),
        'games_scanned': games_scanned
    }


class ParallelPlayerExtractorByEvent:
    """Extractor PARALELO de jugadores por evento."""
    
    def __init__(
        self,
        pgn_path: Path,
        event_type: str,
        num_players: int = 20,
        games_per_player: int = 30,
        min_games_threshold: float = 0.7,
        num_workers: int = None,
        max_workers: int = None
    ):
        """
        Parameters
        ----------
        pgn_path : Path
            Ruta al archivo PGN masivo
        event_type : str
            Tipo de evento a filtrar (ej: "Rated Bullet game")
        num_players : int
            Número de jugadores a extraer
        games_per_player : int
            Número objetivo de partidas por jugador
        min_games_threshold : float
            Umbral mínimo (0.0-1.0)
        num_workers : int, optional
            Número de workers (None = auto según RAM)
        max_workers : int, optional
            Límite máximo de workers
        """
        self.pgn_path = pgn_path
        self.event_type = event_type
        self.num_players = num_players
        self.games_per_player = games_per_player
        self.min_games_threshold = min_games_threshold
        
        self.min_games = int(games_per_player * min_games_threshold)
        
        # Calcular workers según RAM
        if num_workers is None:
            cpu_count = mp.cpu_count()
            
            try:
                import psutil
                available_ram_gb = psutil.virtual_memory().available / (1024**3)
                # Máximo rendimiento: 1GB por worker
                max_workers_by_ram = max(1, int(available_ram_gb / 1))
            except ImportError:
                # Sin psutil: usar todos los cores
                max_workers_by_ram = cpu_count
            
            if max_workers:
                self.num_workers = min(cpu_count, max_workers_by_ram, max_workers)
            else:
                # Usar todos los cores si hay RAM suficiente
                self.num_workers = min(cpu_count, max_workers_by_ram)
        else:
            self.num_workers = num_workers
        
        # Estadísticas
        self.discovered_players: Dict[str, PlayerGameStats] = {}
        self.selected_players: Set[str] = set()
        self.games_scanned = 0
        self.games_of_event = 0
    
    def extract_games(self) -> Dict[str, PlayerGameStats]:
        """
        Extrae jugadores y sus partidas del evento especificado.
        Versión PARALELA optimizada para RAM.
        
        Returns
        -------
        dict
            Jugadores válidos con sus partidas
        """
        print(f"\n{'='*70}")
        print(f"EXTRACCIÓN PARALELA POR EVENTO (Optimizado RAM)")
        print(f"{'='*70}")
        print(f"Evento: {self.event_type}")
        print(f"Objetivo: {self.num_players} jugadores")
        print(f"Partidas/jugador: {self.games_per_player}")
        print(f"Mínimo: {self.min_games} partidas ({self.min_games_threshold*100:.0f}%)")
        print(f"Workers: {self.num_workers} (cores: {mp.cpu_count()})\n")
        
        start_time = time.time()
        
        # FASE 1: Descubrir jugadores del evento (PARALELO)
        print("FASE 1: Descubrimiento PARALELO de jugadores...")
        self._discover_players_parallel()
        
        # FASE 2: Extraer partidas (PARALELO)
        print(f"\nFASE 2: Extracción PARALELA de partidas...")
        self._extract_games_parallel()
        
        # FASE 3: Filtrar por umbral
        print(f"\nFASE 3: Filtrado por umbral...")
        valid_players = self._filter_by_threshold()
        
        elapsed = time.time() - start_time
        
        print(f"\n{'='*70}")
        print(f"RESUMEN DE EXTRACCIÓN")
        print(f"{'='*70}")
        print(f"Partidas escaneadas: {self.games_scanned:,}")
        print(f"Partidas del evento: {self.games_of_event:,}")
        print(f"Jugadores descubiertos: {len(self.discovered_players)}")
        print(f"Jugadores válidos: {len(valid_players)}")
        print(f"Tiempo: {elapsed:.1f}s")
        print(f"{'='*70}\n")
        
        return valid_players
    
    def _discover_players_parallel(self):
        """Descubre jugadores usando procesamiento paralelo."""
        # Dividir archivo en chunks
        boundaries = find_chunk_boundaries(self.pgn_path, self.num_workers)
        
        print(f"  Procesando {len(boundaries)} chunks en paralelo...")
        
        # Procesar chunks en paralelo
        discover_func = partial(
            discover_players_in_chunk,
            pgn_path=self.pgn_path,
            event_type=self.event_type
        )
        
        with mp.Pool(self.num_workers, maxtasksperchild=1) as pool:
            chunk_results = pool.map(discover_func, boundaries)
        
        # Fusionar resultados
        player_frequency = Counter()
        
        for result in chunk_results:
            self.games_scanned += result['games_scanned']
            self.games_of_event += result['games_of_event']
            
            for player, count in result['player_frequency'].items():
                player_frequency[player] += count
        
        # Liberar memoria
        del chunk_results
        gc.collect()
        
        print(f"  ✓ {self.games_scanned:,} partidas escaneadas")
        print(f"  ✓ {self.games_of_event:,} del evento '{self.event_type}'")
        print(f"  ✓ {len(player_frequency)} jugadores encontrados")
        
        # Seleccionar jugadores
        all_players = list(player_frequency.keys())
        
        if self.num_players == 0:
            self.selected_players = set(all_players)
        else:
            sample_size = min(len(all_players), self.num_players * 3)
            self.selected_players = set(random.sample(all_players, sample_size))
        
        print(f"\n✓ Jugadores seleccionados para extracción: {len(self.selected_players)}")
        
        # Inicializar estadísticas
        for player in self.selected_players:
            self.discovered_players[player] = PlayerGameStats(player_name=player)
    
    def _extract_games_parallel(self):
        """Extrae partidas usando procesamiento paralelo."""
        # Dividir archivo en chunks
        boundaries = find_chunk_boundaries(self.pgn_path, self.num_workers)
        
        print(f"  Procesando {len(boundaries)} chunks en paralelo...")
        
        # Procesar chunks en paralelo
        extract_func = partial(
            extract_games_from_chunk,
            pgn_path=self.pgn_path,
            event_type=self.event_type,
            selected_players=self.selected_players,
            games_per_player=self.games_per_player
        )
        
        with mp.Pool(self.num_workers, maxtasksperchild=1) as pool:
            chunk_results = pool.map(extract_func, boundaries)
        
        # Fusionar resultados
        for result in chunk_results:
            for player, stats in result['player_stats'].items():
                self.discovered_players[player].games_as_white += stats['games_as_white']
                self.discovered_players[player].games_as_black += stats['games_as_black']
            
            # Añadir partidas
            for player, games in result['player_games'].items():
                for color, game in games:
                    if self.discovered_players[player].total_games < self.games_per_player:
                        self.discovered_players[player].games.append(game)
        
        # Liberar memoria
        del chunk_results
        gc.collect()
        
        completed = sum(1 for p in self.discovered_players.values() 
                       if p.total_games >= self.min_games)
        print(f"  ✓ {completed} jugadores con ≥{self.min_games} partidas")
    
    def _filter_by_threshold(self) -> Dict[str, PlayerGameStats]:
        """Filtra jugadores que cumplan el umbral mínimo."""
        valid_players = {}
        
        for player, stats in self.discovered_players.items():
            if stats.total_games >= self.min_games:
                valid_players[player] = stats
                print(f"  ✓ {player}: {stats.total_games} partidas "
                      f"(W:{stats.games_as_white}, B:{stats.games_as_black})")
        
        # Si tenemos más de lo necesario, tomar muestra
        if self.num_players > 0 and len(valid_players) > self.num_players:
            print(f"\nSeleccionando {self.num_players} de {len(valid_players)} válidos...")
            selected = dict(random.sample(list(valid_players.items()), self.num_players))
            return selected
        
        return valid_players
    
    def save_player_games(self, output_dir: Path, players: Dict[str, PlayerGameStats]):
        """
        Guarda PGNs por jugador.
        
        Parameters
        ----------
        output_dir : Path
            Directorio donde guardar los PGNs
        players : dict
            Jugadores con sus partidas
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n{'='*70}")
        print(f"GUARDANDO PGNs POR JUGADOR")
        print(f"{'='*70}")
        print(f"Directorio: {output_dir}\n")
        
        for player_name, stats in players.items():
            # Sanitizar nombre
            safe_name = player_name.replace(' ', '_').replace('/', '_')
            output_file = output_dir / f"{safe_name}.pgn"
            
            with open(output_file, 'w') as f:
                for game in stats.games:
                    # Escribir partida
                    exporter = chess.pgn.FileExporter(f)
                    game.accept(exporter)
                    # Separador entre partidas
                    f.write('\n\n')
            
            print(f"  ✓ {safe_name}.pgn: {len(stats.games)} partidas")
        
        print(f"\n✓ {len(players)} archivos guardados")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Extrae jugadores y partidas (versión paralela)'
    )
    parser.add_argument('pgn_file', type=Path, help='Archivo PGN')
    parser.add_argument('--event', required=True, help='Tipo de evento')
    parser.add_argument('--num-players', type=int, default=20, help='Número de jugadores')
    parser.add_argument('--games-per-player', type=int, default=30, help='Partidas por jugador')
    parser.add_argument('--threshold', type=float, default=0.7, help='Umbral mínimo')
    parser.add_argument('--workers', type=int, help='Número de workers')
    parser.add_argument('--max-workers', type=int, help='Máximo de workers')
    
    args = parser.parse_args()
    
    extractor = ParallelPlayerExtractorByEvent(
        pgn_path=args.pgn_file,
        event_type=args.event,
        num_players=args.num_players,
        games_per_player=args.games_per_player,
        min_games_threshold=args.threshold,
        num_workers=args.workers,
        max_workers=args.max_workers
    )
    
    valid_players = extractor.extract_games()
    
    print(f"\n✓ Extracción completada: {len(valid_players)} jugadores válidos")
