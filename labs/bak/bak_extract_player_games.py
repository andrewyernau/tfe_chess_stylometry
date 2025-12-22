#!/usr/bin/env python3
"""
Descubridor y extractor automático de jugadores desde archivo PGN masivo.

Descubre jugadores aleatoriamente en el archivo PGN, extrae sus partidas,
y filtra por umbral mínimo de partidas encontradas.
"""

import chess.pgn
import re
from pathlib import Path
from typing import Dict, List, Set, Optional
from dataclasses import dataclass
from collections import defaultdict
import time
import argparse
import random
import multiprocessing as mp
from functools import partial
import os


@dataclass
class PlayerGameStats:
    """Estadísticas de partidas encontradas para un jugador."""
    player_name: str
    games_as_white: int = 0
    games_as_black: int = 0
    games: List[chess.pgn.Game] = None
    
    def __post_init__(self):
        if self.games is None:
            self.games = []
    
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


class PlayerGameExtractor:
    """Descubridor y extractor automático de jugadores desde PGN masivo."""
    
    def __init__(
        self,
        pgn_path: Path,
        num_players: int = 20,
        games_per_player: int = 30,
        min_games_threshold: float = 0.7,  # 70% mínimo
        player_timeout: float = 300.0,  # 5 minutos sin encontrar partida → siguiente jugador
        buffer_size: int = 8192 * 8
    ):
        """
        Parameters
        ----------
        pgn_path : Path
            Ruta al archivo PGN masivo (ej: lichess_db.pgn)
        num_players : int
            Número de jugadores a extraer aleatoriamente
        games_per_player : int
            Número objetivo de partidas por jugador
        min_games_threshold : float
            Umbral mínimo (0.0-1.0). Ej: 0.7 = al menos 70% de games_per_player
        player_timeout : float
            Timeout local por jugador (se reinicia al encontrar partida)
            Si pasa este tiempo sin encontrar partida, pasa al siguiente jugador
        buffer_size : int
            Tamaño del buffer de lectura (optimización I/O)
        """
        self.pgn_path = pgn_path
        self.num_players = num_players
        self.games_per_player = games_per_player
        self.min_games_threshold = min_games_threshold
        self.player_timeout = player_timeout
        self.buffer_size = buffer_size
        
        # Umbral mínimo de partidas
        self.min_games = int(games_per_player * min_games_threshold)
        
        # Estadísticas
        self.discovered_players: Dict[str, PlayerGameStats] = {}
        self.selected_players: Set[str] = set()
        self.games_scanned = 0
        
    def _normalize_name(self, name: str) -> str:
        """Normaliza nombre de jugador para comparación."""
        return name.strip().lower()
    
    def _scan_chunk(self, args):
        """Escanea un chunk del archivo PGN en paralelo."""
        start_byte, end_byte, chunk_id = args
        player_frequency = defaultdict(int)
        games_read = 0
        
        with open(self.pgn_path, 'r', encoding='utf-8', buffering=self.buffer_size) as pgn_file:
            pgn_file.seek(start_byte)
            
            # Skip to start of next game if not at beginning
            if start_byte != 0:
                line = pgn_file.readline()
                while line and not line.startswith('[Event '):
                    line = pgn_file.readline()
            
            while pgn_file.tell() < end_byte:
                try:
                    game = chess.pgn.read_game(pgn_file)
                    if game is None:
                        break
                    
                    games_read += 1
                    white = game.headers.get("White", "")
                    black = game.headers.get("Black", "")
                    
                    if white:
                        player_frequency[white] += 1
                    if black:
                        player_frequency[black] += 1
                        
                except Exception:
                    continue
        
        return dict(player_frequency), games_read, chunk_id
    
    def discover_players(self, sample_size: int = 50000) -> Set[str]:
        """
        Fase 1: Descubre jugadores escaneando muestra del archivo EN PARALELO.
        
        Parameters
        ----------
        sample_size : int
            Número de partidas a escanear para descubrir jugadores
        
        Returns
        -------
        Set[str]
            Conjunto de jugadores únicos descubiertos
        """
        print(f"\n{'='*70}")
        print(f"FASE 1: DESCUBRIMIENTO DE JUGADORES (PARALELO)")
        print(f"{'='*70}")
        print(f"Escaneando hasta {sample_size:,} partidas...")
        print(f"Buscando jugadores únicos...\n")
        
        # Determinar tamaño del archivo y dividir en chunks
        file_size = os.path.getsize(self.pgn_path)
        num_workers = min(mp.cpu_count(), 72)  # Usar todos los hilos disponibles
        
        # Calcular tamaño de chunk basado en sample_size estimado
        # Asumiendo ~500 bytes por partida en promedio
        estimated_bytes = sample_size * 500
        chunk_bytes = min(estimated_bytes // num_workers, file_size // num_workers)
        
        chunks = []
        current_pos = 0
        chunk_id = 0
        
        while current_pos < file_size and chunk_id < num_workers:
            end_pos = min(current_pos + chunk_bytes, file_size)
            chunks.append((current_pos, end_pos, chunk_id))
            current_pos = end_pos
            chunk_id += 1
        
        print(f"🚀 Procesamiento PARALELO activado:")
        print(f"   CPU cores: {mp.cpu_count()}")
        print(f"   Threads disponibles: 72")
        print(f"   Workers utilizados: {num_workers}")
        print(f"   Chunks: {len(chunks)}")
        print(f"   Tamaño del archivo: {file_size / (1024**3):.2f} GB\n")
        
        start_time = time.time()
        player_frequency = defaultdict(int)
        total_games = 0
        
        # Procesar chunks en paralelo
        with mp.Pool(num_workers) as pool:
            results = pool.map(self._scan_chunk, chunks)
            
            # Combinar resultados
            for freq_dict, games_count, cid in results:
                total_games += games_count
                for player, count in freq_dict.items():
                    player_frequency[player] += count
                print(f"  ✓ Chunk {cid}: {games_count:,} partidas procesadas")
        
        elapsed = time.time() - start_time
        rate = total_games / elapsed if elapsed > 0 else 0
        print(f"\n✓ Descubrimiento completado en {elapsed:.0f}s")
        print(f"  Partidas escaneadas: {total_games:,}")
        print(f"  Velocidad: {rate:.0f} partidas/s ({rate*60:.0f} partidas/min)")
        print(f"  Jugadores únicos encontrados: {len(player_frequency):,}")
        
        # Filtrar jugadores con al menos algunas partidas
        min_appearances = 5  # Al menos 5 apariciones para considerarlo
        valid_players = {
            player for player, count in player_frequency.items() 
            if count >= min_appearances
        }
        
        print(f"  Jugadores con ≥{min_appearances} partidas: {len(valid_players):,}")
        
        return valid_players
    
    def _extract_player_games_worker(self, args):
        """Worker para extraer partidas de un jugador en paralelo."""
        player_name, chunk_start, chunk_end = args
        stats = PlayerGameStats(player_name)
        games_scanned = 0
        
        with open(self.pgn_path, 'r', encoding='utf-8', buffering=self.buffer_size) as pgn_file:
            pgn_file.seek(chunk_start)
            
            # Skip to start of next game if not at beginning
            if chunk_start != 0:
                line = pgn_file.readline()
                while line and not line.startswith('[Event '):
                    line = pgn_file.readline()
            
            while stats.total_games < self.games_per_player and pgn_file.tell() < chunk_end:
                try:
                    game = chess.pgn.read_game(pgn_file)
                    if game is None:
                        break
                    
                    games_scanned += 1
                    
                    white = game.headers.get("White", "")
                    black = game.headers.get("Black", "")
                    
                    if white == player_name:
                        stats.add_game(game, 'white')
                    elif black == player_name:
                        stats.add_game(game, 'black')
                        
                except Exception:
                    continue
        
        return stats, games_scanned
    
    def _extract_multiple_players_from_chunk(self, args):
        """
        Worker que extrae partidas de MÚLTIPLES jugadores desde un chunk del archivo.
        Esto evita que cada jugador lea todo el archivo desde cero.
        
        Parameters
        ----------
        args : tuple
            (player_names, chunk_start, chunk_end, chunk_id)
        """
        player_names, chunk_start, chunk_end, chunk_id = args
        
        # Diccionario para acumular stats de cada jugador
        players_stats = {name: PlayerGameStats(name) for name in player_names}
        players_needs = {name: self.games_per_player for name in player_names}
        
        games_scanned = 0
        start_time = time.time()
        
        try:
            with open(self.pgn_path, 'r', encoding='utf-8', buffering=self.buffer_size) as pgn_file:
                # Ir al chunk
                if chunk_start > 0:
                    pgn_file.seek(chunk_start)
                    # Buscar inicio de partida
                    line = pgn_file.readline()
                    while line and not line.startswith('[Event '):
                        line = pgn_file.readline()
                
                # Procesar hasta el final del chunk
                while pgn_file.tell() < chunk_end:
                    # Check timeout general del chunk (máximo 10 min por chunk)
                    if time.time() - start_time > 600:
                        break
                    
                    try:
                        game = chess.pgn.read_game(pgn_file)
                        if game is None:
                            break
                        
                        games_scanned += 1
                        
                        white = game.headers.get("White", "")
                        black = game.headers.get("Black", "")
                        
                        # Verificar si es partida de alguno de nuestros jugadores
                        for player_name in player_names:
                            if players_needs[player_name] <= 0:
                                continue  # Ya tiene suficientes
                            
                            if white == player_name:
                                players_stats[player_name].add_game(game, 'white')
                                players_needs[player_name] -= 1
                            elif black == player_name:
                                players_stats[player_name].add_game(game, 'black')
                                players_needs[player_name] -= 1
                        
                        # Si todos los jugadores ya tienen suficientes, terminar
                        if all(need <= 0 for need in players_needs.values()):
                            break
                            
                    except Exception:
                        continue
        except Exception as e:
            print(f"      ⚠ Error en chunk {chunk_id}: {e}")
        
        # Retornar solo jugadores con partidas
        result = {}
        for name, stats in players_stats.items():
            if stats.total_games > 0:
                result[name] = stats
        
        return result, games_scanned, chunk_id
        """
        Extrae partidas de un jugador de forma secuencial con timeout.
        
        Parameters
        ----------
        player_name : str
            Nombre del jugador
        chunk_start : int
            Byte de inicio en el archivo (0 = desde el inicio)
        chunk_end : int
            Byte final en el archivo (None = hasta el final)
        """
        stats = PlayerGameStats(player_name)
        games_scanned = 0
        last_game_found_time = time.time()
        
        file_size = os.path.getsize(self.pgn_path)
        if chunk_end is None:
            chunk_end = file_size
        
        with open(self.pgn_path, 'r', encoding='utf-8', buffering=self.buffer_size) as pgn_file:
            # Ir al inicio del chunk
            if chunk_start > 0:
                pgn_file.seek(chunk_start)
                # Buscar inicio de siguiente partida
                line = pgn_file.readline()
                while line and not line.startswith('[Event '):
                    line = pgn_file.readline()
            
            while stats.total_games < self.games_per_player and pgn_file.tell() < chunk_end:
                # Verificar timeout (tiempo sin encontrar partidas del jugador)
                elapsed_since_last = time.time() - last_game_found_time
                if elapsed_since_last > self.player_timeout:
                    if stats.total_games == 0:
                        # No encontró ninguna partida en el timeout
                        break
                    # Si ya tiene algunas, podría ser que estemos en zona sin partidas
                    # Continuar buscando pero con menos paciencia
                    if elapsed_since_last > self.player_timeout * 2:
                        break
                
                try:
                    game = chess.pgn.read_game(pgn_file)
                    if game is None:
                        break
                    
                    games_scanned += 1
                    
                    white = game.headers.get("White", "")
                    black = game.headers.get("Black", "")
                    
                    if white == player_name:
                        stats.add_game(game, 'white')
                        last_game_found_time = time.time()  # Reset timeout
                    elif black == player_name:
                        stats.add_game(game, 'black')
                        last_game_found_time = time.time()  # Reset timeout
                        
                except Exception:
                    continue
        
        self.games_scanned += games_scanned
        
        if stats.total_games > 0:
            print(f"      ✓ {player_name}: {stats.total_games}/{self.games_per_player} partidas "
                  f"({games_scanned:,} escaneadas)")
        
        return stats if stats.total_games > 0 else None
    
    def _extract_player_games(
        self, 
        player_name: str,
        use_nested_parallel: bool = True
    ) -> Optional[PlayerGameStats]:
        """
        Extrae partidas de un jugador específico.
        
        Parameters
        ----------
        player_name : str
            Nombre del jugador
        use_nested_parallel : bool
            Si True, usa paralelización anidada (solo funciona si no estamos ya en un pool).
            Si False, usa procesamiento secuencial (para usar cuando ya estamos en un pool).
        
        Returns
        -------
        PlayerGameStats or None
            Estadísticas del jugador o None si no se encontró
        """
        # Si no podemos usar paralelización anidada (estamos en un pool), usar secuencial
        if not use_nested_parallel:
            return self._extract_player_games_sequential(player_name)
        
        # Intentar paralelización anidada
        try:
            file_size = os.path.getsize(self.pgn_path)
            num_workers = min(mp.cpu_count(), 36)  # Usar los 36 cores físicos
            chunk_size = file_size // num_workers
            
            # Crear chunks para procesar
            chunks = []
            for i in range(num_workers):
                start = i * chunk_size
                end = start + chunk_size if i < num_workers - 1 else file_size
                chunks.append((player_name, start, end))
            
            # Procesar en paralelo
            combined_stats = PlayerGameStats(player_name)
            total_scanned = 0
            
            with mp.Pool(num_workers) as pool:
                results = pool.map(self._extract_player_games_worker, chunks)
                
                for stats, scanned in results:
                    total_scanned += scanned
                    # Combinar partidas (evitar duplicados)
                    for game in stats.games:
                        if len(combined_stats.games) < self.games_per_player:
                            combined_stats.games.append(game)
                            if game.headers.get("White") == player_name:
                                combined_stats.games_as_white += 1
                            else:
                                combined_stats.games_as_black += 1
            
            self.games_scanned += total_scanned
            
            if combined_stats.total_games > 0:
                print(f"      ✓ {combined_stats.total_games}/{self.games_per_player} partidas "
                      f"({total_scanned:,} escaneadas en paralelo)")
            
            return combined_stats if combined_stats.total_games > 0 else None
            
        except AssertionError:
            # Si falla por pools anidados, usar versión secuencial
            return self._extract_player_games_sequential(player_name)
    
    def extract_games(self) -> Dict[str, PlayerGameStats]:
        """
        Fase 3: Extrae partidas garantizando num_players válidos.
        
        Returns
        -------
        Dict[str, PlayerGameStats]
            Exactamente num_players jugadores válidos (si es posible)
        """
        # Fase 1: Descubrir jugadores
        available_players = self.discover_players()
        
        if not available_players:
            print("❌ No se descubrieron jugadores")
            return {}
        
        # Fase 2 & 3: Seleccionar y extraer con reemplazo automático
        valid_players = {}
        available_pool = list(available_players)
        random.shuffle(available_pool)
        
        attempted_players = set()
        player_index = 0
        
        print(f"\n{'='*70}")
        print(f"FASES 2-3: EXTRACCIÓN MASIVA EN PARALELO")
        print(f"{'='*70}")
        print(f"Objetivo: {self.num_players} jugadores con ≥{self.min_games} partidas")
        print(f"Pool disponible: {len(available_pool)} jugadores")
        print(f"🚀 MODO ULTRA-PARALELO: Aprovechando 72 hilos + 128GB RAM")
        print(f"   • Archivo dividido en chunks procesados simultáneamente")
        print(f"   • Múltiples jugadores por chunk (evita re-lectura)")
        print(f"   • Timeout por jugador: {self.player_timeout:.0f}s")
        print(f"   • Umbral mínimo: {self.min_games} partidas ({self.min_games_threshold*100:.0f}%)\n")
        
        start_time = time.time()
        
        # División del archivo en chunks para procesamiento paralelo
        file_size = os.path.getsize(self.pgn_path)
        num_chunks = 72  # Usar todos los hilos disponibles
        chunk_size = file_size // num_chunks
        
        chunks = []
        for i in range(num_chunks):
            chunk_start = i * chunk_size
            chunk_end = chunk_start + chunk_size if i < num_chunks - 1 else file_size
            chunks.append((chunk_start, chunk_end, i))
        
        print(f"📂 Archivo dividido en {num_chunks} chunks de ~{chunk_size/(1024**2):.1f} MB c/u")
        print(f"💾 Tamaño total: {file_size/(1024**3):.2f} GB\n")
        
        # Estrategia: Procesar jugadores en OLEADAS
        # Cada oleada procesa TODOS los chunks en paralelo buscando un grupo de jugadores
        valid_players = {}
        attempted_players = set()
        player_index = 0
        oleada = 0
        
        # Determinar tamaño de oleada: cuántos jugadores buscar simultáneamente
        # Con 128 GB RAM, podemos permitirnos buscar muchos a la vez
        players_per_oleada = min(200, len(available_pool))  # Buscar hasta 200 jugadores a la vez
        
        while player_index < len(available_pool):
            oleada += 1
            
            # Seleccionar jugadores para esta oleada
            oleada_players = []
            while len(oleada_players) < players_per_oleada and player_index < len(available_pool):
                player = available_pool[player_index]
                player_index += 1
                if player not in attempted_players:
                    oleada_players.append(player)
                    attempted_players.add(player)
            
            if not oleada_players:
                break
            
            elapsed = time.time() - start_time
            print(f"\n{'━'*70}")
            print(f"🌊 OLEADA {oleada}")
            print(f"{'━'*70}")
            print(f"Buscando {len(oleada_players)} jugadores en {num_chunks} chunks paralelos...")
            print(f"Progreso global: {len(valid_players)}/{self.num_players} válidos | "
                  f"Tiempo: {elapsed/60:.1f}min | Intentados: {len(attempted_players)}")
            
            # Distribuir jugadores entre chunks
            # Cada chunk buscará TODOS los jugadores de la oleada
            chunk_tasks = [
                (oleada_players, start, end, cid) 
                for start, end, cid in chunks
            ]
            
            # Procesar todos los chunks en paralelo
            oleada_start = time.time()
            print(f"\n⚡ Procesando {num_chunks} chunks en paralelo (esto puede tardar)...")
            
            chunk_results = []
            try:
                with mp.Pool(num_chunks) as pool:
                    chunk_results = pool.map(self._extract_multiple_players_from_chunk, chunk_tasks)
            except Exception as e:
                print(f"⚠ Error en procesamiento paralelo: {e}")
                continue
            
            # Consolidar resultados de todos los chunks
            oleada_stats = {}
            total_scanned = 0
            
            for player_dict, games_scanned, chunk_id in chunk_results:
                total_scanned += games_scanned
                for player_name, stats in player_dict.items():
                    if player_name not in oleada_stats:
                        oleada_stats[player_name] = PlayerGameStats(player_name)
                    
                    # Combinar partidas (evitar duplicados por posición en archivo)
                    for game in stats.games:
                        if oleada_stats[player_name].total_games < self.games_per_player:
                            oleada_stats[player_name].games.append(game)
                            if game.headers.get("White") == player_name:
                                oleada_stats[player_name].games_as_white += 1
                            else:
                                oleada_stats[player_name].games_as_black += 1
            
            oleada_elapsed = time.time() - oleada_start
            
            print(f"\n✓ Oleada procesada en {oleada_elapsed:.1f}s")
            print(f"  Total partidas escaneadas: {total_scanned:,}")
            print(f"  Velocidad: {total_scanned/oleada_elapsed:.0f} partidas/seg")
            print(f"\n📊 Resultados de la oleada:")
            
            # Evaluar jugadores encontrados
            accepted_count = 0
            rejected_count = 0
            
            for player in oleada_players:
                if player in oleada_stats:
                    stats = oleada_stats[player]
                    if stats.total_games >= self.min_games:
                        valid_players[player] = stats
                        accepted_count += 1
                        print(f"   ✓ {player}: {stats.total_games} partidas "
                              f"(W:{stats.games_as_white} B:{stats.games_as_black})")
                    else:
                        rejected_count += 1
                        print(f"   ✗ {player}: {stats.total_games} partidas "
                              f"(mínimo: {self.min_games}) - RECHAZADO")
                else:
                    rejected_count += 1
                    print(f"   ✗ {player}: 0 partidas - NO ENCONTRADO")
            
            print(f"\n   Aceptados: {accepted_count} | Rechazados: {rejected_count}")
            print(f"   Total válidos acumulados: {len(valid_players)}/{self.num_players}")
            
            # Monitoreo de recursos
            try:
                import psutil
                cpu_percent = psutil.cpu_percent(interval=0.1)
                mem = psutil.virtual_memory()
                print(f"\n   💻 Recursos: CPU {cpu_percent:.1f}% | "
                      f"RAM {mem.percent:.1f}% ({mem.used/(1024**3):.1f}/{mem.total/(1024**3):.1f} GB)")
            except:
                pass
            
            # Si ya tenemos suficientes jugadores Y se especificó un límite, terminar
            if self.num_players > 0 and len(valid_players) >= self.num_players:
                print(f"\n🎯 ¡Objetivo alcanzado en Oleada {oleada}!")
                print(f"   Jugadores válidos: {len(valid_players)}/{self.num_players}")
                print(f"   Terminando búsqueda anticipadamente (early stopping)")
                break
            
            # Estimar si necesitamos más oleadas
            if player_index >= len(available_pool):
                print(f"\n⚠ Pool de jugadores agotado")
                print(f"   Jugadores válidos: {len(valid_players)}/{self.num_players}")
                break
        
        total_elapsed = time.time() - start_time
        
        print(f"\n{'='*70}")
        print(f"RESUMEN DE EXTRACCIÓN")
        print(f"{'='*70}")
        print(f"Tiempo total: {total_elapsed/60:.1f} minutos ({total_elapsed:.0f}s)")
        print(f"Partidas totales escaneadas: {self.games_scanned:,}")
        print(f"Jugadores intentados: {len(attempted_players)}")
        print(f"Jugadores válidos: {len(valid_players)}/{self.num_players}")
        
        if valid_players:
            total_games = sum(stats.total_games for stats in valid_players.values())
            avg_games = total_games / len(valid_players)
            print(f"Total de partidas: {total_games}")
            print(f"Promedio por jugador: {avg_games:.1f}")
            
            print(f"\nDetalle por jugador:")
            for player in sorted(valid_players.keys()):
                stats = valid_players[player]
                progress = (stats.total_games / self.games_per_player) * 100
                print(f"  {player}: {stats.total_games} partidas ({progress:.0f}%) "
                      f"(W:{stats.games_as_white}, B:{stats.games_as_black})")
        
        print(f"{'='*70}\n")
        
        return valid_players
    
    def save_player_games(
        self,
        output_dir: Path,
        valid_players: Optional[Dict[str, PlayerGameStats]] = None
    ):
        """
        Guarda partidas de jugadores válidos en archivos PGN separados.
        
        Parameters
        ----------
        output_dir : Path
            Directorio donde guardar los PGN por jugador
        valid_players : Dict[str, PlayerGameStats], optional
            Jugadores a guardar. Si None, usa todos los que cumplieron mínimo.
        """
        if valid_players is None:
            valid_players = {
                player: stats 
                for player, stats in self.stats.items()
                if stats.total_games >= self.min_games
            }
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n{'='*70}")
        print(f"GUARDANDO PARTIDAS POR JUGADOR")
        print(f"{'='*70}")
        print(f"Directorio: {output_dir}")
        print(f"Jugadores a guardar: {len(valid_players)}")
        print(f"{'='*70}\n")
        
        for player, stats in valid_players.items():
            output_file = output_dir / f"{player.replace(' ', '_')}.pgn"
            
            with open(output_file, 'w', encoding='utf-8') as f:
                for game in stats.games:
                    # Escribir partida en formato PGN
                    exporter = chess.pgn.FileExporter(f)
                    game.accept(exporter)
                    f.write('\n\n')
            
            print(f"✓ {output_file.name}: {stats.total_games} partidas guardadas")
        
        print(f"\n{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Descubre y extrae partidas de jugadores aleatorios desde PGN masivo"
    )
    
    parser.add_argument(
        "--pgn-file",
        type=Path,
        default=Path("dataset/generated/lichess_db.pgn"),
        help="Archivo PGN masivo (default: dataset/generated/lichess_db.pgn)"
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
        help="Umbral mínimo de partidas (0.0-1.0). Ej: 0.7 = 70%% (default: 0.7)"
    )
    
    parser.add_argument(
        "--player-timeout",
        type=float,
        default=300.0,
        help="Timeout local por jugador en segundos (se reinicia al encontrar partida) (default: 300)"
    )
    
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("dataset/player_games"),
        help="Directorio de salida (default: dataset/player_games)"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Seed para reproducibilidad (default: aleatorio)"
    )
    
    args = parser.parse_args()
    
    # Validar archivo PGN
    if not args.pgn_file.exists():
        print(f"❌ Error: Archivo PGN no encontrado: {args.pgn_file}")
        return 1
    
    # Establecer seed si se proporciona
    if args.seed is not None:
        random.seed(args.seed)
        print(f"🎲 Seed establecido: {args.seed}")
    
    # Crear extractor
    extractor = PlayerGameExtractor(
        pgn_path=args.pgn_file,
        num_players=args.num_players,
        games_per_player=args.games_per_player,
        min_games_threshold=args.min_threshold,
        player_timeout=args.player_timeout
    )
    
    # Extraer partidas
    valid_players = extractor.extract_games()
    
    # Guardar partidas
    if valid_players:
        extractor.save_player_games(args.output_dir, valid_players)
        return 0
    else:
        print("⚠ No se encontraron jugadores con el mínimo de partidas requerido")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
