#!/usr/bin/env python3
"""
Generador de mapas de calor de tiempo de decisión para partidas de ajedrez.

Crea imágenes que visualizan el tiempo de decisión por movimiento usando
un gradiente de colores fríos (decisiones rápidas) a cálidos (decisiones lentas).
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
    clock_time: float  # Tiempo restante en segundos
    decision_time: float  # Tiempo empleado en la decisión (segundos)
    decision_pct: float  # Tiempo empleado como % del tiempo total


class DecisionTimeExtractor:
    """Extrae tiempos de decisión desde partidas PGN."""
    
    @staticmethod
    def parse_clock_time(clk_string: str) -> Optional[float]:
        """
        Convierte string de reloj a segundos.
        
        Ejemplos: '0:01:00' -> 60.0, '0:00:59' -> 59.0
        """
        if not clk_string:
            return None
        
        # Formato: H:MM:SS o MM:SS
        match = re.match(r'(\d+):(\d+):(\d+)', clk_string)
        if match:
            hours, minutes, seconds = map(int, match.groups())
            return hours * 3600 + minutes * 60 + seconds
        
        # Formato alternativo: MM:SS
        match = re.match(r'(\d+):(\d+)', clk_string)
        if match:
            minutes, seconds = map(int, match.groups())
            return minutes * 60 + seconds
        
        return None
    
    @staticmethod
    def extract_time_control(game: chess.pgn.Game) -> Optional[Tuple[int, int]]:
        """
        Extrae configuración de tiempo de la partida.
        
        Returns
        -------
        Tuple[int, int] or None
            (tiempo_inicial_segundos, incremento_segundos) o None si no disponible
        """
        tc_header = game.headers.get("TimeControl", "")
        
        # Formato: "base+increment" (ej: "60+0", "180+2")
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
        """
        Extrae tiempos de decisión de todos los movimientos.
        
        Parameters
        ----------
        game : chess.pgn.Game
            Partida de ajedrez con información de tiempo
        use_relative : bool
            Si True, calcula tiempos relativos (% del tiempo total)
            Si False, usa tiempos absolutos (segundos)
        
        Returns
        -------
        List[MoveDecision]
            Lista de decisiones con información temporal
        """
        decisions = []
        
        # Obtener configuración de tiempo
        time_control = self.extract_time_control(game)
        if time_control is None:
            return decisions
        
        initial_time, increment = time_control
        
        # Rastrear tiempo anterior para cada jugador
        prev_clock = {'white': None, 'black': None}
        
        board = game.board()
        move_num = 0
        
        for node in game.mainline():
            move_num += 1
            
            # Determinar color del movimiento
            color = 'white' if board.turn == chess.WHITE else 'black'
            
            # Ejecutar movimiento
            board.push(node.move)
            
            # Extraer tiempo de reloj del comentario
            comment = node.comment
            clk_match = re.search(r'\[%clk\s+(\d+:\d+:\d+|\d+:\d+)\]', comment)
            
            if clk_match:
                clock_str = clk_match.group(1)
                clock_time = self.parse_clock_time(clock_str)
                
                if clock_time is not None:
                    # Calcular tiempo de decisión
                    if prev_clock[color] is not None:
                        # |T(n+1) - T(n)| - increment
                        raw_decision = abs(prev_clock[color] - clock_time)
                        decision_time = max(0, raw_decision - increment)
                        
                        # Calcular porcentaje relativo al tiempo total
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


class DecisionHeatmapGenerator:
    """Genera mapas de calor visuales de tiempos de decisión por casilla de tablero."""
    
    def __init__(
        self,
        grid_size: Tuple[int, int] = (8, 8),
        output_size: int = 192,
        use_relative: bool = True,
        colormap: int = cv2.COLORMAP_JET
    ):
        """
        Parameters
        ----------
        grid_size : Tuple[int, int]
            Dimensiones del grid (filas, columnas) - debe ser 8x8 para tablero de ajedrez
        output_size : int
            Tamaño de salida en píxeles (cuadrado) - default 192x192 (tamaño del tablero)
        use_relative : bool
            True = tiempos relativos (%), False = absolutos (segundos)
        colormap : int
            Mapa de color de OpenCV (COLORMAP_JET para frío a caliente, etc.)
        """
        self.grid_rows, self.grid_cols = grid_size
        self.output_size = output_size
        self.use_relative = use_relative
        self.colormap = colormap
        self.extractor = DecisionTimeExtractor()
    
    def generate_heatmap(
        self,
        game: chess.pgn.Game,
        start_move: int = 1,
        end_move: Optional[int] = None,
        percentile_clip: float = 95.0,
        return_legend: bool = False
    ) -> Optional[np.ndarray]:
        """
        Genera mapa de calor de decisiones para una partida por casilla del tablero.
        
        El mapa representa el tiempo de decisión por casilla donde se realizó un movimiento:
        - Negro: Sin movimiento en esa casilla
        - Azul/Frío: Decisión rápida
        - Rojo/Caliente: Decisión lenta (mucho tiempo pensado)
        
        Si múltiples movimientos afectan la misma casilla, se pueden:
        - Acumular (suma de tiempos)
        - Promediar (media de tiempos)
        - Tomar el máximo (decisión más larga)
        
        Por defecto usamos ACUMULACIÓN para poder diferenciar casillas con múltiples jugadas.
        
        Parameters
        ----------
        game : chess.pgn.Game
            Partida de ajedrez
        start_move : int
            Movimiento inicial a visualizar
        end_move : int, optional
            Movimiento final (None = hasta el final)
        percentile_clip : float
            Percentil para clipear outliers (ej: 95 = top 5% se satura)
        return_legend : bool
            Si True, retorna también la barra de leyenda del gradiente
        
        Returns
        -------
        np.ndarray or None
            Imagen RGB del mapa de calor (192x192), o None si no hay datos
        """
        # Extraer decisiones
        decisions = self.extractor.extract_decision_times(game, self.use_relative)
        
        if not decisions:
            return None
        
        # Filtrar por rango de movimientos
        if end_move is None:
            end_move = max(d.move_number for d in decisions)
        
        filtered_decisions = [
            d for d in decisions 
            if start_move <= d.move_number <= end_move
        ]
        
        if not filtered_decisions:
            return None
        
        # Crear grid 8x8 para el tablero (acumulador de tiempos por casilla)
        # grid[row][col] acumula el tiempo de decisión de todos los movimientos a esa casilla
        decision_grid = np.zeros((8, 8), dtype=np.float32)
        count_grid = np.zeros((8, 8), dtype=np.int32)  # Contador de movimientos por casilla
        
        # Recorrer la partida y mapear tiempos a casillas de destino
        board = game.board()
        move_num = 0
        decision_idx = 0
        
        for node in game.mainline():
            move_num += 1
            move = node.move
            
            # Obtener casilla de destino (to_square)
            to_square = move.to_square
            
            # Convertir índice de casilla a coordenadas del grid (row, col)
            # Chess usa: a1=0, b1=1, ..., h1=7, a2=8, ..., h8=63
            # Queremos: fila 0 = fila 8 del tablero (desde arriba), col 0 = columna a
            col = to_square % 8  # 0-7 (a-h)
            row = 7 - (to_square // 8)  # Invertir para que fila 0 sea la fila 8
            
            # Si este movimiento tiene datos de decisión en el rango
            if decision_idx < len(filtered_decisions):
                decision = filtered_decisions[decision_idx]
                
                if decision.move_number == move_num:
                    value = decision.decision_pct if self.use_relative else decision.decision_time
                    
                    # Acumular tiempo en la casilla de destino
                    decision_grid[row][col] += value
                    count_grid[row][col] += 1
                    
                    decision_idx += 1
            
            board.push(move)
        
        # Normalizar valores con clipping de percentiles
        nonzero_values = decision_grid[decision_grid > 0]
        
        if len(nonzero_values) == 0:
            return None
        
        vmax = np.percentile(nonzero_values, percentile_clip)
        vmin = 0
        
        # Normalizar a rango [0, 255]
        decision_grid_norm = np.clip((decision_grid / vmax) * 255, 0, 255).astype(np.uint8)
        
        # Aplicar colormap (0=negro/azul, 255=rojo)
        heatmap_colored = cv2.applyColorMap(decision_grid_norm, self.colormap)
        
        # Forzar celdas sin datos a negro puro (transparente visualmente)
        mask_zero = decision_grid_norm == 0
        heatmap_colored[mask_zero] = [0, 0, 0]
        
        # Redimensionar a tamaño del tablero (192x192) con interpolación nearest para mantener grid
        heatmap_resized = cv2.resize(
            heatmap_colored, 
            (self.output_size, self.output_size), 
            interpolation=cv2.INTER_NEAREST  # Mantiene el efecto de grid nítido
        )
        
        # Dibujar líneas del grid (opcional pero recomendado para visualización)
        heatmap_with_grid = self._draw_grid_lines(heatmap_resized)
        
        # Convertir BGR a RGB
        heatmap_rgb = cv2.cvtColor(heatmap_with_grid, cv2.COLOR_BGR2RGB)
        
        if return_legend:
            # Generar leyenda de gradiente (barra horizontal)
            legend = self._create_colormap_legend()
            return heatmap_rgb, legend
        
        return heatmap_rgb
    
    def _draw_grid_lines(self, image: np.ndarray, line_color=(50, 50, 50), thickness=1) -> np.ndarray:
        """
        Dibuja líneas de grid sobre la imagen para separar casillas.
        
        Parameters
        ----------
        image : np.ndarray
            Imagen BGR del heatmap
        line_color : tuple
            Color BGR de las líneas de grid
        thickness : int
            Grosor de las líneas
        
        Returns
        -------
        np.ndarray
            Imagen con grid dibujado
        """
        img_copy = image.copy()
        cell_size = self.output_size // 8
        
        # Dibujar líneas verticales
        for i in range(1, 8):
            x = i * cell_size
            cv2.line(img_copy, (x, 0), (x, self.output_size), line_color, thickness)
        
        # Dibujar líneas horizontales
        for i in range(1, 8):
            y = i * cell_size
            cv2.line(img_copy, (0, y), (self.output_size, y), line_color, thickness)
        
        return img_copy
    
    def _create_colormap_legend(self, width: int = 256, height: int = 30) -> np.ndarray:
        """Crea barra de leyenda de gradiente de colores.
        
        Returns
        -------
        np.ndarray
            Imagen RGB de la leyenda (height x width x 3)
        """
        # Crear gradiente lineal de 0 a 255
        gradient = np.linspace(0, 255, width, dtype=np.uint8)
        gradient = np.repeat(gradient[np.newaxis, :], height, axis=0)
        
        # Aplicar colormap
        legend_colored = cv2.applyColorMap(gradient, self.colormap)
        
        # Convertir BGR a RGB
        legend_rgb = cv2.cvtColor(legend_colored, cv2.COLOR_BGR2RGB)
        
        return legend_rgb


def process_pgn_to_heatmaps(
    pgn_path: Path,
    output_dir: Path,
    start_move: int = 15,
    end_move: int = 23,
    grid_size: Tuple[int, int] = (8, 8),
    cell_size: int = 24,
    use_relative: bool = True,
    max_games: Optional[int] = None
) -> int:
    """
    Procesa archivo PGN y genera mapas de calor para cada partida.
    
    Parameters
    ----------
    pgn_path : Path
        Ruta al archivo PGN
    output_dir : Path
        Directorio de salida
    start_move : int
        Movimiento inicial
    end_move : int
        Movimiento final
    grid_size : Tuple[int, int]
        Dimensiones del grid (debe ser 8x8)
    cell_size : int
        Parámetro legacy, ignorado (se usa output_size=224)
    use_relative : bool
        True = tiempos relativos, False = absolutos
    max_games : int, optional
        Máximo de partidas a procesar (None = todas)
    
    Returns
    -------
    int
        Número de mapas de calor generados
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    generator = DecisionHeatmapGenerator(
        grid_size=grid_size,
        output_size=192,  # Tamaño igual al tablero
        use_relative=use_relative
    )
    
    player_name = pgn_path.stem
    games_processed = 0
    legend_saved = False
    
    print(f"\n{'='*70}")
    print(f"GENERANDO MAPAS DE CALOR DE DECISIÓN")
    print(f"{'='*70}")
    print(f"Archivo: {pgn_path.name}")
    print(f"Rango de movimientos: {start_move}-{end_move}")
    print(f"Grid: {grid_size[0]}x{grid_size[1]} (tablero estándar)")
    print(f"Tamaño salida: 192x192 píxeles (igual al tablero)")
    print(f"Tipo: {'Tiempos relativos (%)' if use_relative else 'Tiempos absolutos (s)'}")
    print(f"Gradiente: Azul (rápido) → Rojo (lento)")
    print(f"Método: Acumulación por casilla de destino")
    print(f"{'='*70}\n")
    
    with open(pgn_path, 'r', encoding='utf-8') as pgn_file:
        game_num = 0
        
        while True:
            if max_games and game_num >= max_games:
                break
            
            game = chess.pgn.read_game(pgn_file)
            if game is None:
                break
            
            game_num += 1
            
            try:
                # Generar mapa de calor con leyenda para el primero
                if not legend_saved:
                    result = generator.generate_heatmap(
                        game,
                        start_move=start_move,
                        end_move=end_move,
                        return_legend=True
                    )
                    
                    if result is not None:
                        heatmap, legend = result
                        # Guardar leyenda una sola vez
                        legend_path = output_dir / "colormap_legend.png"
                        legend_bgr = cv2.cvtColor(legend, cv2.COLOR_RGB2BGR)
                        cv2.imwrite(str(legend_path), legend_bgr)
                        legend_saved = True
                    else:
                        heatmap = None
                else:
                    heatmap = generator.generate_heatmap(
                        game,
                        start_move=start_move,
                        end_move=end_move
                    )
                
                if heatmap is None:
                    print(f"⚠ {player_name} game {game_num}: Sin datos de tiempo")
                    continue
                
                # Guardar imagen
                output_filename = f"{player_name}_game{game_num:04d}_heatmap.png"
                output_path = output_dir / output_filename
                
                # Convertir RGB a BGR para OpenCV
                heatmap_bgr = cv2.cvtColor(heatmap, cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(output_path), heatmap_bgr)
                
                games_processed += 1
                
                if games_processed % 10 == 0:
                    print(f"Procesadas: {games_processed}/{game_num}")
                
            except Exception as e:
                print(f"{player_name} game {game_num}: {str(e)}")
                continue
    
    print(f"\n{'='*70}")
    print(f"Mapas de calor generados: {games_processed}")
    print(f"Guardados en: {output_dir}")
    if legend_saved:
        print(f"Leyenda de gradiente: {output_dir / 'colormap_legend.png'}")
    print(f"{'='*70}\n")
    
    return games_processed


def main():
    parser = argparse.ArgumentParser(
        description="Genera mapas de calor de tiempo de decisión desde PGN"
    )
    
    parser.add_argument(
        "--pgn-file",
        type=Path,
        required=True,
        help="Archivo PGN con partidas"
    )
    
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/heatmaps"),
        help="Directorio de salida (default: output/heatmaps)"
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
        "--grid-rows",
        type=int,
        default=8,
        help="Filas del grid (default: 8, estándar tablero)"
    )
    
    parser.add_argument(
        "--grid-cols",
        type=int,
        default=8,
        help="Columnas del grid (default: 8, estándar tablero)"
    )
    
    parser.add_argument(
        "--output-size",
        type=int,
        default=224,
        help="Tamaño de salida en píxeles (cuadrado) (default: 224)"
    )
    
    parser.add_argument(
        "--absolute-time",
        action='store_true',
        help="Usar tiempos absolutos en vez de relativos"
    )
    
    parser.add_argument(
        "--max-games",
        type=int,
        default=None,
        help="Máximo de partidas a procesar"
    )
    
    args = parser.parse_args()
    
    if not args.pgn_file.exists():
        print(f"Error: Archivo no encontrado: {args.pgn_file}")
        return 1
    
    process_pgn_to_heatmaps(
        pgn_path=args.pgn_file,
        output_dir=args.output_dir,
        start_move=args.start_move,
        end_move=args.end_move,
        grid_size=(args.grid_rows, args.grid_cols),
        cell_size=args.output_size,  # Reusa el parámetro legacy como output_size
        use_relative=not args.absolute_time,
        max_games=args.max_games
    )
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
