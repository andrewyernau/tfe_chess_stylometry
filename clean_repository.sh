#!/bin/bash
# Script para limpiar archivos residuales del repositorio

echo "🧹 Limpiando repositorio..."

# Limpiar __pycache__
echo "Eliminando __pycache__..."
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null
echo "✓ __pycache__ eliminados"

# Limpiar .ipynb_checkpoints
echo "Eliminando .ipynb_checkpoints..."
find . -type d -name ".ipynb_checkpoints" -exec rm -rf {} + 2>/dev/null
echo "✓ .ipynb_checkpoints eliminados"

# Limpiar archivos .pyc
echo "Eliminando archivos .pyc..."
find . -type f -name "*.pyc" -delete 2>/dev/null
echo "✓ Archivos .pyc eliminados"

# Limpiar archivos .pyo
echo "Eliminando archivos .pyo..."
find . -type f -name "*.pyo" -delete 2>/dev/null
echo "✓ Archivos .pyo eliminados"

# Limpiar archivos temporales
echo "Eliminando archivos temporales..."
find . -type f -name "*.tmp" -delete 2>/dev/null
find . -type f -name "*.temp" -delete 2>/dev/null
find . -type f -name "*~" -delete 2>/dev/null
echo "✓ Archivos temporales eliminados"

# Limpiar archivos de swap
echo "Eliminando archivos de swap..."
find . -type f -name "*.swp" -delete 2>/dev/null
find . -type f -name "*.swo" -delete 2>/dev/null
echo "✓ Archivos de swap eliminados"

echo ""
echo "✅ Limpieza completada!"
echo ""
echo "Estructura limpia del repositorio:"
ls -lh | grep -v "^total"
