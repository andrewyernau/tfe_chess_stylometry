#!/usr/bin/env python3
"""
Script de prueba para verificar que el pipeline genera correctamente
1 tablero = 1 heatmap por partida.
"""

from pathlib import Path
import cv2

def verificar_emparejamiento(event_dir: Path):
    """Verifica que haya 1 tablero = 1 heatmap por partida."""
    
    board_images_dir = event_dir / "board_images"
    heatmap_images_dir = event_dir / "heatmap_images"
    
    if not board_images_dir.exists():
        print(f"❌ No existe: {board_images_dir}")
        return False
    
    if not heatmap_images_dir.exists():
        print(f"❌ No existe: {heatmap_images_dir}")
        return False
    
    print(f"\n{'='*80}")
    print(f"VERIFICANDO EMPAREJAMIENTO")
    print(f"{'='*80}")
    
    # Verificar cada jugador
    total_ok = 0
    total_fail = 0
    
    for player_dir in sorted(board_images_dir.glob("*")):
        if not player_dir.is_dir():
            continue
        
        player_name = player_dir.name
        heatmap_player_dir = heatmap_images_dir / player_name
        
        if not heatmap_player_dir.exists():
            print(f"\n⚠️  {player_name}: No existe directorio de heatmaps")
            continue
        
        # Contar archivos
        board_files = list(player_dir.glob("game_*.png"))
        heatmap_files = list(heatmap_player_dir.glob("game_*.png"))
        
        print(f"\n📂 {player_name}:")
        print(f"   Tableros: {len(board_files)}")
        print(f"   Heatmaps: {len(heatmap_files)}")
        
        # Verificar que cada tablero tiene su heatmap
        for board_file in board_files:
            # Extraer game_num del nombre
            # Antes: game_0001_move_15.png
            # Ahora: game_0001.png
            game_num = board_file.stem.split('_')[1]  # game_0001 -> 0001
            
            heatmap_file = heatmap_player_dir / f"game_{game_num}.png"
            
            if heatmap_file.exists():
                # Verificar que las imágenes se pueden leer
                board_img = cv2.imread(str(board_file))
                heatmap_img = cv2.imread(str(heatmap_file))
                
                if board_img is not None and heatmap_img is not None:
                    total_ok += 1
                else:
                    print(f"   ❌ game_{game_num}: Imagen corrupta")
                    total_fail += 1
            else:
                print(f"   ❌ game_{game_num}: Falta heatmap")
                total_fail += 1
        
        # Verificar formato del nombre de archivo
        if board_files:
            ejemplo = board_files[0].name
            if "_move_" in ejemplo:
                print(f"   ⚠️  FORMATO ANTIGUO detectado: {ejemplo}")
                print(f"   ⚠️  Regenerar datos con pipeline actualizado!")
            else:
                print(f"   ✅ Formato correcto: {ejemplo}")
    
    print(f"\n{'='*80}")
    print(f"RESUMEN")
    print(f"{'='*80}")
    print(f"Pares correctos: {total_ok}")
    print(f"Pares con error: {total_fail}")
    
    if total_fail == 0:
        print(f"\n✅ TODO CORRECTO - Emparejamiento 1:1 verificado")
        return True
    else:
        print(f"\n❌ HAY ERRORES - Revisar pipeline")
        return False


if __name__ == "__main__":
    # Verificar datos de ejemplo
    output_base = Path("/home/andrewyernau/dev/jupyter/labs/output")
    
    # Buscar todos los eventos generados
    events_dir = output_base / "events"
    
    if not events_dir.exists():
        print(f"❌ No existe directorio de eventos: {events_dir}")
        print(f"Ejecuta primero el pipeline de estilometría")
        exit(1)
    
    event_dirs = [d for d in events_dir.iterdir() if d.is_dir()]
    
    if not event_dirs:
        print(f"❌ No hay eventos generados en: {events_dir}")
        exit(1)
    
    print(f"Eventos encontrados: {len(event_dirs)}")
    
    # Verificar cada evento
    all_ok = True
    for event_dir in event_dirs:
        print(f"\n📊 Verificando evento: {event_dir.name}")
        ok = verificar_emparejamiento(event_dir)
        if not ok:
            all_ok = False
    
    if all_ok:
        print(f"\n\n{'='*80}")
        print(f"✅ TODOS LOS EVENTOS VERIFICADOS CORRECTAMENTE")
        print(f"{'='*80}")
    else:
        print(f"\n\n{'='*80}")
        print(f"❌ ALGUNOS EVENTOS TIENEN ERRORES")
        print(f"{'='*80}")
        print(f"\nSolución: Regenerar datos con el pipeline actualizado:")
        print(f"  cd /home/andrewyernau/dev/jupyter/labs")
        print(f"  python3 pipeline_stylometry_by_event.py")
