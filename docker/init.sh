#!/bin/bash
# Script de inicialización para Docker
# Crea directorios necesarios y verifica la configuración

set -e

echo "========================================="
echo "Inicializando proyecto Bank Marketing ML"
echo "========================================="

# Crear directorios necesarios
echo "Creando directorios necesarios..."
mkdir -p models data/raw data/processed logs

# Verificar que el archivo de datos existe
if [ ! -f "data/raw/bank.csv" ] && [ ! -f "data/raw/bank_100k.csv" ]; then
    echo "⚠️  ADVERTENCIA: No se encontró bank.csv ni bank_100k.csv en data/raw/"
    echo "   Asegúrate de tener los datos antes de entrenar modelos"
fi

# Verificar permisos
echo "Verificando permisos de directorios..."
chmod -R 755 models data logs 2>/dev/null || true

echo "✓ Inicialización completada"
echo ""
echo "Próximos pasos:"
echo "1. Asegúrate de tener los datos en data/raw/"
echo "2. Entrena el modelo ML: docker exec bank_marketing_api python train_model.py"
echo "3. O entrena un modelo DL desde el dashboard en la pestaña 'Entrenar DL'"
echo ""


