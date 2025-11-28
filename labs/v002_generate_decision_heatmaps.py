#!/usr/bin/env python3
"""
Generador de mapas de calor de tiempo de decisión (Versión 002).

CAMBIOS PRINCIPALES V002:
- Heatmaps en ESCALA DE GRISES (negro=rápido, blanco=lento)
- Soporte para bloques de 5 movimientos
- Procesamiento de 15 jugadas por jugador (30 movimientos totales)
- Fragmentación en 3 bloques por jugador para mejor visualización
"""

import chess.pgn
import numpy as np
import cv2
from pathlib import Path
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
import argparse
import re
from io import StringIO


@dataclass
class MoveDecision:
    """Información de tiempo de decisión para un movimiento."""
    move_number: int
    player: str  # 'white' or 'black'
    clock_time: float
    decision_time: float
    decision_pct: float


class DecisionTimeExtractor:
    """Extrae tiempos de decisión desde partidas PGN."""
    
    @staticmethod
    def parse_clock_time(clk_string: str) -> Optional[float]:
        """Convierte string de reloj a segundos."""
        if not clk_string:
            return None
        
        match = re.match(r'(\d+):(\d+):(\d+)', clk_string)
        if match:
            hours, minutes, seconds = map(int, match.groups())
            return hours * 3600 + minutes * 60 + seconds
        
        match = re.match(r'(\d+):(\d+)', clk_string)
        if match:
            minutes, seconds = map(int, match.groups())
            return minutes * 60 + seconds
        
        return None
    
    @staticmethod
    def extract_time_control(game: chess.pgn.Game) -> Optional[Tuple[int, int]]:
        """Extrae configuración de tiempo."""
        tc_header = game.headers.get("TimeControl", "")
        match = re.match(r'(\d+)\+(\d+)', tc_header)
        if match:
            base, increment = map(int, match.groups())
            return (base, increment)
        return None
    
    def extract_decision_times(
        self,
        game: chess.pgn.Game,
        use_relative: bool = True
    ) -> List[MoveDecision]:
        """Extrae tiempos de decisión de todos los movimientos."""
        decisions = []
        
        time_control = self.extract_time_control(game)
        if time_control is None:
            return decisions
        
        initial_time, increment = time_control
        prev_clock = {'white': None, 'black': None}
        
        board = game.board()
        move_num = 0
        
        for node in game.mainline():
            move_num += 1
            color = 'white' if board.turn == chess.WHITE else 'black'
            board.push(node.move)
            
            comment = node.comment
            clk_match = re.search(r'\[%clk\s+(\d+:\d+:\d+|\d+:\d+)\]', comment)
            
            if clk_match:
                clock_str = clk_match.group(1)
                clock_time = self.parse_clock_time(clock_str)
                
                if clock_time is not None:
                    if prev_clock[color] is not None:
                        raw_decision = abs(prev_clock[color] - clock_time)
                        decision_time = max(0, raw_decision - increment)
                        
                        if use_relative and initial_time > 0:
                            decision_pct = (decision_time / initial_time) * 100
                        else:
                            decision_pct = decision_time
                        
                        decisions.append(MoveDecision(
                            move_number=move_num,
                            player=color,
                            clock_time=clock_time,
                            decision_time=decision_time,
                            decision_pct=decision_pct
                        ))
                    
                    prev_clock[color] = clock_time
        
        return decisions


class GrayscaleHeatmapGenerator:
    """
    Genera mapas de calor en ESCALA DE GRISES.
    
    NEGRO (0) = Decisión rápida (poco tiempo pensado)
    BLANCO (255) = Decisión lenta (mucho tiempo pensado)
    """
    
    def __init__(
        self,
        grid_size: Tuple[int, int] = (8, 8),
        output_size: int = 192,
        use_relative: bool = True
    ):
        """
        Parameters
        ----------
        grid_size : Tuple[int, int]
            Dimensiones del grid (8x8 para tablero)
        output_size : int
            Tamaño de salida en píxeles (192x192)
        use_relative : bool
            True = tiempos relativos (%), False = absolutos (segundos)
        """
        self.grid_rows, self.grid_cols = grid_size
        self.output_size = output_size
        self.use_relative = use_relative
        self.extractor = DecisionTimeExtractor()
    
    def generate_heatmap(
        self,
        game: chess.pgn.Game,
        start_move: int = 1,
        end_move: Optional[int] = None,
        percentile_clip: float = 95.0
    ) -> Optional[np.ndarray]:
        """
        Genera mapa de calor en ESCALA DE GRISES.
        
        Parameters
        ----------
        game : chess.pgn.Game
            Partida de ajedrez
        start_move : int
            Movimiento inicial
        end_move : int, optional
            Movimiento final
        percentile_clip : float
            Percentil para clipear outliers
        
        Returns
        -------
        np.ndarray or None
            Imagen en escala de grises (192x192x3 BGR), o None si no hay datos
        """
        decisions = self.extractor.extract_decision_times(game, self.use_relative)
        
        if not decisions:
            return None
        
        if end_move is None:
            end_move = max(d.move_number for d in decisions)
        
        filtered_decisions = [
            d for d in decisions 
            if start_move <= d.move_number <= end_move
        ]
        
        if not filtered_decisions:
            return None
        
        decision_grid = np.zeros((8, 8), dtype=np.float32)
        count_grid = np.zeros((8, 8), dtype=np.int32)
        
        # Crear mapa de decisiones por número de movimiento
        decision_map = {d.move_number: d for d in filtered_decisions}
        
        board = game.board()
        move_num = 0
        
        for node in game.mainline():
            move_num += 1
            move = node.move
            
            # Solo procesar movimientos en el rango del bloque
            if move_num < start_move:
                board.push(move)
                continue
            
            if move_num > end_move:
                break
            
            # Buscar decisión para este movimiento
            if move_num in decision_map:
                decision = decision_map[move_num]
                value = decision.decision_pct if self.use_relative else decision.decision_time
                
                # Pintar solo casilla ORIGEN (donde estaba la pieza antes de moverse)
                from_square = move.from_square
                from_col = from_square % 8
                from_row = 7 - (from_square // 8)
                
                decision_grid[from_row][from_col] += value
                count_grid[from_row][from_col] += 1
            
            board.push(move)
        
        nonzero_values = decision_grid[decision_grid > 0]
        
        if len(nonzero_values) == 0:
            return None
        
        vmax = np.percentile(nonzero_values, percentile_clip)
        vmin = 0
        
        # Normalizar a escala de grises [30, 255]
        # Mínimo 30 para jugadas rápidas (negro puro = sin movimiento)
        decision_grid_norm = np.zeros_like(decision_grid, dtype=np.uint8)
        mask_nonzero = decision_grid > 0
        decision_grid_norm[mask_nonzero] = np.clip(
            30 + ((decision_grid[mask_nonzero] / vmax) * 225), 
            30, 255
        ).astype(np.uint8)
        
        # Crear imagen en escala de grises (3 canales para compatibilidad)
        heatmap_gray = cv2.cvtColor(decision_grid_norm, cv2.COLOR_GRAY2BGR)
        
        # Celdas sin datos permanecen en NEGRO puro [0, 0, 0]
        mask_zero = decision_grid == 0
        heatmap_gray[mask_zero] = [0, 0, 0]
        
        # Redimensionar a tamaño del tablero
        heatmap_resized = cv2.resize(
            heatmap_gray, 
            (self.output_size, self.output_size), 
            interpolation=cv2.INTER_NEAREST
        )
        
        # Dibujar grid lines
        heatmap_with_grid = self._draw_grid_lines(heatmap_resized)
        
        return heatmap_with_grid
    
    def _draw_grid_lines(self, image: np.ndarray, line_color=(80, 80, 80), thickness=1) -> np.ndarray:
        """Dibuja líneas de grid."""
        img_copy = image.copy()
        cell_size = self.output_size // 8
        
        for i in range(1, 8):
            x = i * cell_size
            cv2.line(img_copy, (x, 0), (x, self.output_size), line_color, thickness)
        
        for i in range(1, 8):
            y = i * cell_size
            cv2.line(img_copy, (0, y), (self.output_size, y), line_color, thickness)
        
        return img_copy


def process_pgn_to_heatmaps(
    pgn_path: Path,
    output_dir: Path,
    start_move: int = 15,
    end_move: int = 45,
    use_relative: bool = True,
    max_games: Optional[int] = None,
    verbose: bool = True
) -> int:
    """
    Procesa archivo PGN y genera mapas de calor en escala de grises.
    
    Parameters
    ----------
    pgn_path : Path
        Ruta al archivo PGN
    output_dir : Path
        Directorio de salida
    start_move : int
        Movimiento inicial (default 15)
    end_move : int
        Movimiento final (default 45 para 15 jugadas por jugador)
    use_relative : bool
        Usar tiempos relativos
    max_games : int, optional
        Máximo de partidas a procesar
    verbose : bool
        Mostrar progreso
    
    Returns
    -------
    int
        Número de heatmaps generados
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    generator = GrayscaleHeatmapGenerator(
        grid_size=(8, 8),
        output_size=192,
        use_relative=use_relative
    )
    
    count = 0
    game_num = 0
    
    with open(pgn_path) as pgn_file:
        while True:
            if max_games and game_num >= max_games:
                break
            
            game = chess.pgn.read_game(pgn_file)
            if game is None:
                break
            
            game_num += 1
            
            try:
                heatmap = generator.generate_heatmap(
                    game,
                    start_move=start_move,
                    end_move=end_move
                )
                
                if heatmap is not None:
                    output_file = output_dir / f"game_{game_num:04d}.png"
                    cv2.imwrite(str(output_file), heatmap)
                    count += 1
                    
                    if verbose and count % 10 == 0:
                        print(f"  Generados: {count} heatmaps")
            
            except Exception as e:
                if verbose:
                    print(f"  Error en partida {game_num}: {e}")
                continue
    
    if verbose:
        print(f"\nTotal heatmaps generados: {count}")
    
    return count


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generador de heatmaps en escala de grises (v002)'
    )
    parser.add_argument('pgn_file', type=Path, help='Archivo PGN')
    parser.add_argument('--output', type=Path, required=True, help='Dir de salida')
    parser.add_argument('--start-move', type=int, default=15, help='Movimiento inicial')
    parser.add_argument('--end-move', type=int, default=45, help='Movimiento final')
    parser.add_argument('--relative', action='store_true', help='Tiempos relativos')
    parser.add_argument('--max-games', type=int, help='Máximo de partidas')
    
    args = parser.parse_args()
    
    count = process_pgn_to_heatmaps(
        args.pgn_file,
        args.output,
        args.start_move,
        args.end_move,
        args.relative,
        args.max_games
    )
    
    print(f"✓ Completado: {count} heatmaps generados")
