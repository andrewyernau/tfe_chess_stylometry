#!/usr/bin/env python3
"""
Script para limpiar archivos temporales del directorio de descargas.
ADVERTENCIA: Este script eliminará TODOS los archivos .pgn y .zst en /tmp/chess_downloads/
"""

from pathlib import Path
import sys

def cleanup_temp_directory(directory: Path, dry_run: bool = True):
    """
    Limpia archivos temporales del directorio especificado.
    
    Parameters
    ----------
    directory : Path
        Directorio a limpiar
    dry_run : bool
        Si True, solo muestra qué se eliminaría sin hacerlo
    """
    if not directory.exists():
        print(f"❌ Directorio no existe: {directory}")
        return
    
    patterns = ['*.pgn', '*.zst', '*.pgn.pgn']
    files_to_delete = []
    
    for pattern in patterns:
        files_to_delete.extend(directory.glob(pattern))
    
    if not files_to_delete:
        print(f"✓ No hay archivos temporales para limpiar en {directory}")
        return
    
    total_size = 0
    print(f"\n{'='*70}")
    print(f"Archivos {'A ELIMINAR' if not dry_run else 'QUE SE ELIMINARÍAN'}:")
    print(f"{'='*70}\n")
    
    for file in files_to_delete:
        size_gb = file.stat().st_size / (1024**3)
        total_size += size_gb
        print(f"  - {file.name:<50} {size_gb:>8.2f} GB")
    
    print(f"\n{'='*70}")
    print(f"Total archivos: {len(files_to_delete)}")
    print(f"Espacio total: {total_size:.2f} GB")
    print(f"{'='*70}\n")
    
    if dry_run:
        print("⚠️  MODO DRY-RUN: No se eliminó nada.")
        print("   Ejecuta con --confirm para eliminar realmente los archivos.")
        return
    
    # Pedir confirmación adicional
    print("⚠️  ¿Estás seguro de que quieres eliminar estos archivos?")
    print("   Esta acción NO se puede deshacer.")
    response = input("   Escribe 'SI' en mayúsculas para confirmar: ")
    
    if response != "SI":
        print("\n❌ Cancelado. No se eliminó nada.")
        return
    
    print("\n🗑️  Eliminando archivos...")
    deleted = 0
    freed_space = 0
    
    for file in files_to_delete:
        try:
            size_gb = file.stat().st_size / (1024**3)
            file.unlink()
            deleted += 1
            freed_space += size_gb
            print(f"  ✓ Eliminado: {file.name} ({size_gb:.2f} GB)")
        except Exception as e:
            print(f"  ✗ Error eliminando {file.name}: {e}")
    
    print(f"\n{'='*70}")
    print(f"✓ Limpieza completada")
    print(f"{'='*70}")
    print(f"Archivos eliminados: {deleted}/{len(files_to_delete)}")
    print(f"Espacio liberado: {freed_space:.2f} GB")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    TEMP_DIR = Path("/tmp/chess_downloads")
    
    # Verificar si se pasó --confirm
    confirm = "--confirm" in sys.argv
    
    print(f"\n{'='*70}")
    print(f"LIMPIEZA DE ARCHIVOS TEMPORALES")
    print(f"{'='*70}")
    print(f"Directorio: {TEMP_DIR}\n")
    
    if not confirm:
        print("⚠️  Ejecutando en MODO DRY-RUN (solo simulación)")
        print("   Para eliminar realmente, ejecuta: python cleanup_temp_files.py --confirm\n")
    
    cleanup_temp_directory(TEMP_DIR, dry_run=not confirm)
