#!/usr/bin/env python3
"""
Ejemplo de uso del parser de partidas de ajedrez.

Este script muestra cómo llamar al módulo parse_games_to_images
desde otro archivo Python.
"""

from pathlib import Path
from labs.parse_games_to_images import main as parse_games


def ejemplo_basico():
    """Ejemplo básico de uso con parámetros por defecto."""
    print("Ejemplo 1: Configuración básica")
    print("-" * 50)
    
    parse_games(
        pgn_dir=Path("dataset/testpgns"),
        output_dir=Path("output/ejemplo1"),
        start_move=15,
        end_move=23,
        compression_factor=2
    )


def ejemplo_alta_calidad():
    """Ejemplo con máxima calidad (sin compresión)."""
    print("\nEjemplo 2: Máxima calidad (sin compresión)")
    print("-" * 50)
    
    parse_games(
        pgn_dir=Path("dataset/testpgns"),
        output_dir=Path("output/ejemplo2_hq"),
        start_move=10,
        end_move=20,
        compression_factor=1  # Sin compresión
    )


def ejemplo_comprimido():
    """Ejemplo con alta compresión para eficiencia."""
    print("\nEjemplo 3: Alta compresión (4x)")
    print("-" * 50)
    
    parse_games(
        pgn_dir=Path("dataset/testpgns"),
        output_dir=Path("output/ejemplo3_compressed"),
        start_move=5,
        end_move=15,
        compression_factor=4  # Compresión 4x
    )


def ejemplo_rango_amplio():
    """Ejemplo con rango amplio de movimientos."""
    print("\nEjemplo 4: Rango amplio de movimientos")
    print("-" * 50)
    
    parse_games(
        pgn_dir=Path("dataset/testpgns"),
        output_dir=Path("output/ejemplo4_wide"),
        start_move=1,
        end_move=30,
        compression_factor=2
    )


if __name__ == "__main__":
    # Descomentar el ejemplo que quieras ejecutar
    
    ejemplo_basico()
    # ejemplo_alta_calidad()
    # ejemplo_comprimido()
    # ejemplo_rango_amplio()
