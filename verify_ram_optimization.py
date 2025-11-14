#!/usr/bin/env python3
"""
Script de verificación de optimizaciones de RAM.

Verifica que:
1. event_discovery_parallel.py usa streaming (no carga chunks completos)
2. Auto-limita workers según RAM disponible
3. Los notebooks están actualizados
"""

import sys
from pathlib import Path

def check_streaming_implementation():
    """Verifica que se use streaming en lugar de carga completa."""
    file_path = Path('labs/event_discovery_parallel.py')
    
    if not file_path.exists():
        return False, "Archivo no encontrado"
    
    content = file_path.read_text()
    
    # Buscar implementación de streaming
    if 'BUFFER_SIZE = 1024 * 1024' in content:
        if 'class LimitedReader' in content:
            if 'buffering=BUFFER_SIZE' in content:
                return True, "✓ Streaming implementado correctamente"
    
    # Verificar que NO use la versión antigua
    if 'chunk_data = f.read(chunk_size)' in content:
        return False, "✗ Todavía usa carga completa de chunks"
    
    return False, "✗ Implementación de streaming no encontrada"


def check_worker_limiting():
    """Verifica que auto-limite workers según RAM."""
    file_path = Path('labs/event_discovery_parallel.py')
    
    if not file_path.exists():
        return False, "Archivo no encontrado"
    
    content = file_path.read_text()
    
    # Buscar auto-limitación
    if 'psutil.virtual_memory()' in content:
        if 'max_workers_by_ram' in content:
            if 'available_ram_gb' in content:
                return True, "✓ Auto-limitación de workers implementada"
    
    return False, "✗ Auto-limitación no encontrada"


def check_memory_cleanup():
    """Verifica que libere memoria activamente."""
    file_path = Path('labs/event_discovery_parallel.py')
    
    if not file_path.exists():
        return False, "Archivo no encontrado"
    
    content = file_path.read_text()
    
    # Buscar liberación de memoria
    if 'gc.collect()' in content:
        if 'chunk_results[i] = None' in content or 'del chunk_results' in content:
            if 'maxtasksperchild=1' in content:
                return True, "✓ Liberación activa de memoria implementada"
    
    return False, "✗ Liberación de memoria no encontrada"


def check_notebooks():
    """Verifica que notebooks estén actualizados."""
    notebooks = [
        'labs/notebooks/0002a_single_channel_cnn.ipynb',
        'labs/notebooks/0002b_dual_channel_cnn.ipynb'
    ]
    
    results = []
    for nb_path in notebooks:
        path = Path(nb_path)
        if not path.exists():
            results.append((False, f"✗ {nb_path} no encontrado"))
            continue
        
        content = path.read_text()
        
        # Verificar que use ParallelEventDiscovery
        if 'ParallelEventDiscovery' in content:
            # Verificar que mencione optimización de RAM
            if 'RAM' in content or 'optimizado' in content.lower():
                results.append((True, f"✓ {path.name} actualizado"))
            else:
                results.append((True, f"⚠ {path.name} usa parallel pero sin mención de RAM"))
        else:
            results.append((False, f"✗ {path.name} no usa ParallelEventDiscovery"))
    
    all_ok = all(r[0] for r in results)
    msg = "\n  ".join(r[1] for r in results)
    return all_ok, msg


def check_ram_availability():
    """Verifica RAM disponible en el sistema."""
    try:
        import psutil
        ram_gb = psutil.virtual_memory().available / (1024**3)
        total_gb = psutil.virtual_memory().total / (1024**3)
        
        # Calcular workers esperados
        expected_workers = max(1, int(ram_gb / 4))
        
        return True, f"✓ RAM disponible: {ram_gb:.1f} GB / {total_gb:.1f} GB total\n  Workers esperados: {expected_workers}"
    except ImportError:
        return False, "⚠ psutil no instalado (pip install psutil)"


def main():
    print("="*70)
    print("VERIFICACIÓN DE OPTIMIZACIONES DE RAM")
    print("="*70)
    print()
    
    checks = [
        ("1. Streaming (no carga completa)", check_streaming_implementation),
        ("2. Auto-limitación de workers", check_worker_limiting),
        ("3. Liberación activa de memoria", check_memory_cleanup),
        ("4. Notebooks actualizados", check_notebooks),
        ("5. RAM del sistema", check_ram_availability),
    ]
    
    all_passed = True
    
    for name, check_func in checks:
        print(f"\n{name}")
        print("-" * 70)
        success, message = check_func()
        print(f"  {message}")
        
        if not success:
            all_passed = False
    
    print("\n" + "="*70)
    if all_passed:
        print("✅ TODAS LAS VERIFICACIONES PASARON")
        print("\nEl pipeline está optimizado para uso eficiente de RAM.")
        print("\nPróximo paso: Ejecutar test con:")
        print("  cd labs")
        print("  python3 event_discovery_parallel.py dataset/generated/lichess_db.pgn --max-games 1000")
    else:
        print("❌ ALGUNAS VERIFICACIONES FALLARON")
        print("\nRevisa los mensajes arriba para más detalles.")
        sys.exit(1)
    print("="*70)


if __name__ == '__main__':
    main()
