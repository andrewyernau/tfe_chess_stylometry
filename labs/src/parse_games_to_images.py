#!/usr/bin/env python3
"""
Chess Game Parser - Visual Temporal Encoding

Parsea partidas de ajedrez desde archivos PGN y genera imágenes con codificación
visual temporal (transparencia decreciente para jugadas antiguas).

Basado en chess_cnn_visual_temporal.ipynb pero sin banda de metadatos.
"""

import chess
import chess.pgn
import chess.svg
import numpy as np
import cv2
from typing import List, Tuple, Optional
from pathlib import Path
from io import StringIO
import argparse
import sys
import os


def board_to_png_array(board: chess.Board, size: int = 400) -> np.ndarray:
    """Convierte un tablero de ajedrez a array numpy en escala de grises desde SVG.
    
    Parameters
    ----------
    board : chess.Board
        Tablero de ajedrez en estado específico.
    size : int, optional
        Tamaño en pixels del tablero cuadrado (default: 400).
    
    Returns
    -------
    np.ndarray
        Array en escala de grises de la imagen del tablero sin leyenda.
        Shape: (size, size), dtype: uint8
    """
    if not isinstance(board, chess.Board):
        raise TypeError("[CHESS_CNN] board debe ser una instancia de chess.Board")
    
    if size <= 0:
        raise ValueError("[CHESS_CNN] size debe ser positivo")
    
    try:
        import cairosvg
    except ImportError:
        raise ImportError(
            "[CHESS_CNN] cairosvg no disponible. Instalar con: pip install cairosvg"
        )
    
    # Generar SVG sin coordenadas para máxima limpieza visual
    svg_data = chess.svg.board(board=board, size=size, coordinates=False)
    
    # Convertir SVG a PNG en memoria
    png_data = cairosvg.svg2png(bytestring=svg_data.encode('utf-8'))
    
    # Decodificar PNG a array numpy
    png_array = np.frombuffer(png_data, dtype=np.uint8)
    img = cv2.imdecode(png_array, cv2.IMREAD_COLOR)
    
    # Convertir a escala de grises
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    return img_gray


def extract_board_sequence(
    pgn_text: str,
    start_move: int,
    end_move: int
) -> List[chess.Board]:
    """Extrae secuencia de tableros desde PGN entre movimientos especificados.
    
    Parameters
    ----------
    pgn_text : str
        Texto completo del archivo PGN.
    start_move : int
        Número del movimiento inicial (más antiguo).
    end_move : int
        Número del movimiento final (más reciente).
    
    Returns
    -------
    List[chess.Board]
        Lista de tableros ordenados del más antiguo al más reciente.
        Posición 0 = movimiento start_move
        Última posición = movimiento end_move
    """
    if start_move < 1 or end_move < start_move:
        raise ValueError(
            f"[CHESS_CNN] Rango de movimientos inválido: {start_move}-{end_move}"
        )
    
    # Parsear PGN
    game = chess.pgn.read_game(StringIO(pgn_text))
    if game is None:
        raise ValueError("[CHESS_CNN] No se pudo parsear el PGN")
    
    # Extraer secuencia de tableros
    boards = []
    board = game.board()
    move_num = 0
    
    for node in game.mainline():
        move_num += 1
        board.push(node.move)
        
        if start_move <= move_num <= end_move:
            # Copiar tablero para evitar referencias mutables
            boards.append(board.copy())
    
    if len(boards) < (end_move - start_move + 1):
        raise ValueError(
            f"[CHESS_CNN] La partida tiene solo {move_num} movimientos, "
            f"pero se solicitaron movimientos {start_move}-{end_move}"
        )
    
    return boards


def overlay_temporal_sequence(
    board_sequence: List[chess.Board],
    compression_factor: int = 2,
    board_size: int = 400,
    min_alpha: float = 0.3,
    max_alpha: float = 1.0
) -> np.ndarray:
    """Superpone secuencia de tableros con transparencia decreciente (escala de grises).
    
    Los tableros se superponen con transparencia basada en antigüedad:
    - Movimiento más reciente (última posición): opaco (alpha=1.0)
    - Movimiento más antiguo (primera posición): transparente (alpha~0.3)
    
    Parameters
    ----------
    board_sequence : List[chess.Board]
        Secuencia de tableros ordenados del más antiguo al más reciente.
    compression_factor : int, optional
        Factor de reducción de tamaño (default: 2).
        1 = sin compresión, 2 = mitad de tamaño, 4 = cuarto de tamaño, etc.
    board_size : int, optional
        Tamaño del tablero antes de compresión (default: 400).
    min_alpha : float, optional
        Transparencia mínima para movimientos antiguos (default: 0.3).
    max_alpha : float, optional
        Transparencia máxima para movimientos recientes (default: 1.0).
    
    Returns
    -------
    np.ndarray
        Imagen en escala de grises con superposición temporal.
        Shape: (height, width), dtype: uint8
    """
    if not board_sequence:
        raise ValueError("[CHESS_CNN] board_sequence no puede estar vacía")
    
    if compression_factor < 1:
        raise ValueError("[CHESS_CNN] compression_factor debe ser >= 1")
    
    if not (0.0 <= min_alpha <= max_alpha <= 1.0):
        raise ValueError(
            "[CHESS_CNN] Transparencias deben cumplir: 0 <= min <= max <= 1"
        )
    
    num_boards = len(board_sequence)
    
    # Calcular tamaño comprimido (fijo a 224x224 como estándar)
    compressed_size = 224
    
    # Inicializar imagen acumulada (float para precisión)
    accumulated = np.zeros((compressed_size, compressed_size), dtype=np.float32)
    
    # Procesar cada tablero en orden (antiguo → reciente)
    for i, board in enumerate(board_sequence):
        # Renderizar tablero a escala de grises
        img_gray = board_to_png_array(board, size=board_size)
        
        # Redimensionar a tamaño estándar
        img_resized = cv2.resize(
            img_gray,
            (compressed_size, compressed_size),
            interpolation=cv2.INTER_AREA
        )
        
        # Calcular alpha (transparencia) temporal
        # i=0 (antiguo) → min_alpha
        # i=num_boards-1 (reciente) → max_alpha
        if num_boards > 1:
            progress = i / (num_boards - 1)
        else:
            progress = 1.0
        
        alpha = min_alpha + (max_alpha - min_alpha) * progress
        
        # Aplicar alpha y acumular usando máximo (jugadas más recientes prevalecen)
        img_weighted = img_resized.astype(np.float32) * alpha
        accumulated = np.maximum(accumulated, img_weighted)
    
    # Convertir de vuelta a uint8
    result = np.clip(accumulated, 0, 255).astype(np.uint8)
    
    return result


def process_pgn_file(
    pgn_path: Path,
    output_dir: Path,
    start_move: int,
    end_move: int,
    compression_factor: int
) -> int:
    """Procesa un archivo PGN completo y genera imágenes para cada partida.
    
    Parameters
    ----------
    pgn_path : Path
        Ruta al archivo PGN.
    output_dir : Path
        Directorio de salida para las imágenes.
    start_move : int
        Movimiento inicial.
    end_move : int
        Movimiento final.
    compression_factor : int
        Factor de compresión.
    
    Returns
    -------
    int
        Número de partidas procesadas exitosamente.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    player_name = pgn_path.stem  # Nombre del archivo sin extensión
    games_processed = 0
    
    with open(pgn_path, encoding='utf-8') as pgn_file:
        game_num = 0
        while True:
            game = chess.pgn.read_game(pgn_file)
            if game is None:
                break
            
            game_num += 1
            
            try:
                # Convertir game a string PGN
                exporter = chess.pgn.StringExporter(
                    headers=True, 
                    variations=False, 
                    comments=False
                )
                pgn_text = game.accept(exporter)
                
                # Extraer secuencia de tableros
                board_sequence = extract_board_sequence(
                    pgn_text, 
                    start_move, 
                    end_move
                )
                
                # Generar imagen con superposición temporal
                img = overlay_temporal_sequence(
                    board_sequence,
                    compression_factor=compression_factor,
                    min_alpha=0.3,
                    max_alpha=1.0
                )
                
                # Guardar imagen (ya está en escala de grises)
                output_filename = f"{player_name}_game{game_num:02d}.png"
                output_path = output_dir / output_filename
                
                cv2.imwrite(str(output_path), img)
                
                games_processed += 1
                print(f"✓ {output_filename} ({img.shape[0]}x{img.shape[1]})")
                
            except Exception as e:
                print(f"✗ {player_name} game {game_num}: {str(e)}", file=sys.stderr)
                continue
    
    return games_processed


def main(
    pgn_dir: Path,
    output_dir: Path,
    start_move: int,
    end_move: int,
    compression_factor: int
):
    """Función principal que procesa todos los archivos PGN.
    
    Parameters
    ----------
    pgn_dir : Path
        Directorio con archivos .pgn (puede ser relativo o absoluto).
        Si es relativo, se busca desde el directorio actual.
    output_dir : Path
        Directorio de salida para imágenes (puede ser relativo o absoluto).
        Si es relativo, se crea desde el directorio actual.
    start_move : int
        Movimiento inicial
    end_move : int
        Movimiento final
    compression_factor : int
        Factor de compresión
    """
    # Convertir a Path si es necesario
    pgn_dir = Path(pgn_dir)
    output_dir = Path(output_dir)
    
    # Si la ruta no es absoluta, intentar resolverla desde el directorio actual
    if not pgn_dir.is_absolute():
        # Primero intentar desde el directorio actual
        pgn_dir_abs = Path.cwd() / pgn_dir
        if not pgn_dir_abs.exists():
            # Si no existe, intentar desde el directorio del script (labs/)
            script_dir = Path(__file__).parent
            pgn_dir_alt = script_dir / pgn_dir
            if pgn_dir_alt.exists():
                pgn_dir = pgn_dir_alt
            else:
                pgn_dir = pgn_dir_abs  # Usar la primera opción para el mensaje de error
        else:
            pgn_dir = pgn_dir_abs
    
    # Resolver a ruta absoluta
    pgn_dir = pgn_dir.resolve()
    output_dir = output_dir.resolve()
    
    # Validar parámetros
    if not pgn_dir.exists():
        raise ValueError(f"Directorio PGN no existe: {pgn_dir}")
    
    if start_move < 1 or end_move < start_move:
        raise ValueError(f"Rango de movimientos inválido: {start_move}-{end_move}")
    
    if compression_factor < 1:
        raise ValueError(f"Factor de compresión debe ser >= 1: {compression_factor}")
    
    # Obtener todos los archivos .pgn
    pgn_files = sorted(pgn_dir.glob("*.pgn"))
    
    if not pgn_files:
        raise ValueError(f"No se encontraron archivos .pgn en: {pgn_dir}")
    
    print(f"\n{'='*70}")
    print(f"PROCESANDO PARTIDAS DE AJEDREZ")
    print(f"{'='*70}")
    print(f"Directorio PGN: {pgn_dir}")
    print(f"Directorio salida: {output_dir}")
    print(f"Rango de movimientos: {start_move}-{end_move}")
    print(f"Factor de compresión: {compression_factor}x")
    print(f"Archivos PGN encontrados: {len(pgn_files)}")
    print(f"{'='*70}\n")
    
    total_games = 0
    
    for pgn_path in pgn_files:
        print(f"\n📁 Procesando: {pgn_path.name}")
        print(f"   {'-'*66}")
        
        games_count = process_pgn_file(
            pgn_path,
            output_dir,
            start_move,
            end_move,
            compression_factor
        )
        
        total_games += games_count
        print(f"   {'-'*66}")
        print(f"   Partidas procesadas: {games_count}")
    
    print(f"\n{'='*70}")
    print(f"RESUMEN FINAL")
    print(f"{'='*70}")
    print(f"Total de archivos PGN: {len(pgn_files)}")
    print(f"Total de partidas procesadas: {total_games}")
    print(f"Imágenes generadas en: {output_dir}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Parsea partidas de ajedrez a imágenes con codificación temporal"
    )
    
    parser.add_argument(
        "--pgn-dir",
        type=Path,
        default=Path("dataset/testpgns"),
        help="Directorio con archivos .pgn (default: dataset/testpgns)"
    )
    
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/parsed_games"),
        help="Directorio de salida para imágenes (default: output/parsed_games)"
    )
    
    parser.add_argument(
        "--start-move",
        type=int,
        required=True,
        help="Movimiento inicial (más antiguo)"
    )
    
    parser.add_argument(
        "--end-move",
        type=int,
        required=True,
        help="Movimiento final (más reciente)"
    )
    
    parser.add_argument(
        "--compression-factor",
        type=int,
        default=2,
        help="Factor de compresión (1=sin compresión, 2=mitad, 4=cuarto, etc.)"
    )
    
    args = parser.parse_args()
    
    try:
        main(
            pgn_dir=args.pgn_dir,
            output_dir=args.output_dir,
            start_move=args.start_move,
            end_move=args.end_move,
            compression_factor=args.compression_factor
        )
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}\n", file=sys.stderr)
        sys.exit(1)
