#!/usr/bin/env python3
"""
Monitor en tiempo real del pipeline de extracción.
Muestra CPU, RAM, progreso y detecta si el sistema se ha colgado.
"""

import psutil
import time
import sys
from pathlib import Path
import subprocess

def monitor_pipeline(interval=5, max_idle_time=300):
    """
    Monitorea el pipeline en tiempo real.
    
    Parameters
    ----------
    interval : int
        Segundos entre cada actualización
    max_idle_time : int
        Segundos sin cambios antes de alertar (posible cuelgue)
    """
    print("=" * 80)
    print("🔍 MONITOR DEL PIPELINE - Presiona Ctrl+C para salir")
    print("=" * 80)
    print(f"Intervalo de actualización: {interval}s")
    print(f"Alerta de inactividad: {max_idle_time}s")
    print("=" * 80)
    
    last_cpu_times = None
    last_change_time = time.time()
    last_values = {}
    iteration = 0
    
    try:
        while True:
            iteration += 1
            current_time = time.time()
            
            # CPU
            cpu_percent = psutil.cpu_percent(interval=0.5, percpu=False)
            cpu_per_core = psutil.cpu_percent(interval=0, percpu=True)
            active_cores = sum(1 for x in cpu_per_core if x > 10)
            
            # RAM
            mem = psutil.virtual_memory()
            mem_used_gb = mem.used / (1024**3)
            mem_total_gb = mem.total / (1024**3)
            
            # Procesos Python
            python_procs = []
            for proc in psutil.process_iter(['name', 'cpu_percent', 'memory_percent']):
                try:
                    if 'python' in proc.info['name'].lower():
                        python_procs.append(proc)
                except:
                    pass
            
            num_python = len(python_procs)
            
            # Disco I/O
            disk_io = psutil.disk_io_counters()
            disk_read_mb = disk_io.read_bytes / (1024**2) if disk_io else 0
            disk_write_mb = disk_io.write_bytes / (1024**2) if disk_io else 0
            
            # Detectar cambios (si hay actividad)
            current_values = {
                'cpu': cpu_percent,
                'mem': mem_used_gb,
                'procs': num_python,
                'disk_read': disk_read_mb
            }
            
            if last_values:
                # Verificar si hay cambios significativos
                has_change = False
                for key in current_values:
                    if abs(current_values[key] - last_values.get(key, 0)) > 0.1:
                        has_change = True
                        break
                
                if has_change:
                    last_change_time = current_time
            
            last_values = current_values
            
            # Calcular tiempo inactivo
            idle_time = current_time - last_change_time
            
            # Mostrar estado
            print(f"\n{'─' * 80}")
            print(f"⏱️  Actualización #{iteration} - {time.strftime('%H:%M:%S')}")
            print(f"{'─' * 80}")
            
            # CPU
            cpu_bar = "█" * int(cpu_percent / 2) + "░" * (50 - int(cpu_percent / 2))
            print(f"💻 CPU:  [{cpu_bar}] {cpu_percent:5.1f}%")
            print(f"   Cores activos (>10%): {active_cores}/{len(cpu_per_core)}")
            
            # RAM
            ram_bar = "█" * int(mem.percent / 2) + "░" * (50 - int(mem.percent / 2))
            print(f"💾 RAM:  [{ram_bar}] {mem.percent:5.1f}%")
            print(f"   Usado: {mem_used_gb:.1f} GB / {mem_total_gb:.1f} GB")
            
            # Procesos
            print(f"🐍 Python: {num_python} procesos activos")
            
            # Disco
            print(f"💿 Disco: Read {disk_read_mb/1024:.1f} GB | Write {disk_write_mb/1024:.1f} GB")
            
            # Estado
            if idle_time < 10:
                status = "🟢 ACTIVO"
                status_color = "\033[92m"
            elif idle_time < max_idle_time:
                status = "🟡 PROCESANDO"
                status_color = "\033[93m"
            else:
                status = "🔴 POSIBLE CUELGUE"
                status_color = "\033[91m"
            
            print(f"\n{status_color}📊 Estado: {status}\033[0m")
            print(f"   Última actividad hace: {idle_time:.0f}s")
            
            if idle_time > max_idle_time:
                print(f"\n⚠️  ALERTA: Sin cambios detectados en {idle_time:.0f}s")
                print(f"   El sistema podría estar colgado o procesando lentamente")
            
            # Esperar
            time.sleep(interval)
            
    except KeyboardInterrupt:
        print("\n\n" + "=" * 80)
        print("🛑 Monitor detenido por el usuario")
        print("=" * 80)


def show_current_status():
    """Muestra un snapshot rápido del estado actual."""
    cpu = psutil.cpu_percent(interval=1)
    mem = psutil.virtual_memory()
    
    print("\n" + "=" * 80)
    print("📸 SNAPSHOT DEL SISTEMA")
    print("=" * 80)
    print(f"CPU:  {cpu:.1f}%")
    print(f"RAM:  {mem.percent:.1f}% ({mem.used/(1024**3):.1f}/{mem.total/(1024**3):.1f} GB)")
    
    # Procesos Python
    python_count = 0
    for proc in psutil.process_iter(['name']):
        try:
            if 'python' in proc.info['name'].lower():
                python_count += 1
        except:
            pass
    
    print(f"Procesos Python: {python_count}")
    print("=" * 80)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Monitor del pipeline")
    parser.add_argument("--interval", type=int, default=5, help="Segundos entre actualizaciones")
    parser.add_argument("--max-idle", type=int, default=300, help="Segundos máximos sin cambios")
    parser.add_argument("--snapshot", action="store_true", help="Solo mostrar estado actual")
    
    args = parser.parse_args()
    
    if args.snapshot:
        show_current_status()
    else:
        monitor_pipeline(interval=args.interval, max_idle_time=args.max_idle)
