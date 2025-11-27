#!/bin/bash
# Script para configurar el proyecto en producción después del despliegue inicial
# Ejecutar dentro de la instancia EC2 después de clonar el repositorio

set -e

echo "========================================="
echo "Configurando proyecto para producción"
echo "========================================="

# Verificar que estamos en el directorio del proyecto
if [ ! -f "docker-compose.yml" ]; then
    echo "Error: No se encontró docker-compose.yml"
    echo "Asegúrate de estar en el directorio raíz del proyecto"
    exit 1
fi

# Generar password seguro para PostgreSQL
echo "Generando password seguro para PostgreSQL..."
DB_PASSWORD=$(openssl rand -base64 32 | tr -d "=+/" | cut -c1-25)

# Crear archivo .env
echo "Creando archivo .env..."
cat > .env << EOF
# Database Configuration
DB_HOST=db
DB_PORT=5432
DB_NAME=bank_marketing
DB_USER=postgres
DB_PASSWORD=$DB_PASSWORD

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000

# Dashboard Configuration
DASHBOARD_HOST=0.0.0.0
DASHBOARD_PORT=8050
EOF

echo "✓ Archivo .env creado"
echo ""
echo "IMPORTANTE: Guarda este password de forma segura:"
echo "DB_PASSWORD=$DB_PASSWORD"
echo ""

# Crear directorios necesarios
echo "Creando directorios necesarios..."
mkdir -p models data/raw data/processed logs backups
echo "✓ Directorios creados"

# Construir y levantar servicios
echo ""
echo "Construyendo imágenes Docker..."
docker compose build

echo ""
echo "Levantando servicios..."
docker compose up -d

# Esperar a que la base de datos esté lista
echo ""
echo "Esperando a que la base de datos esté lista..."
sleep 15

# Verificar que los servicios están corriendo
echo ""
echo "Verificando servicios..."
docker ps

# Entrenar el modelo
echo ""
echo "Entrenando el modelo (esto puede tardar varios minutos)..."
docker exec bank_marketing_api python train_model.py

echo ""
echo "========================================="
echo "Configuración completada!"
echo "========================================="
echo ""
echo "Servicios disponibles:"
echo "  API: http://localhost:8000"
echo "  Dashboard: http://localhost:8050"
echo "  API Docs: http://localhost:8000/docs"
echo ""
echo "Para ver logs:"
echo "  docker compose logs -f"
echo ""
echo "Para detener servicios:"
echo "  docker compose down"
echo ""
echo "Para reiniciar servicios:"
echo "  docker compose restart"
echo ""


