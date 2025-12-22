"""
Celda de emergencia para recargar el generador sin reiniciar kernel.
Agregar esta celda ANTES de importar create_train_val_generators.
"""

import sys
import importlib

# Lista de módulos a recargar
modules_to_reload = ['multi_channel_generator', 'multi_channel_model']

print("Recargando módulos actualizados...")
for module_name in modules_to_reload:
    if module_name in sys.modules:
        print(f"  - Eliminando {module_name} del cache...")
        del sys.modules[module_name]

# Ahora importar la versión nueva
from multi_channel_generator import create_train_val_generators
from multi_channel_model import build_multi_channel_model

print("✓ Módulos actualizados cargados correctamente")
print(f"✓ Generador base class: {create_train_val_generators.__module__}")
