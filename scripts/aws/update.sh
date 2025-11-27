#!/bin/bash
# Script para actualizar el proyecto en producción
# Ejecutar dentro de la instancia EC2

set -e

echo "========================================="
echo "Actualizando proyecto"
echo "========================================="

# Verificar que estamos en el directorio del proyecto
if [ ! -f "docker-compose.yml" ]; then
    echo "Error: No se encontró docker-compose.yml"
    echo "Asegúrate de estar en el directorio raíz del proyecto"
    exit 1
fi

# Hacer backup antes de actualizar
echo "Haciendo backup antes de actualizar..."
if [ -f "scripts/aws/backup.sh" ]; then
    bash scripts/aws/backup.sh
else
    echo "⚠ Script de backup no encontrado, continuando sin backup..."
fi

# Actualizar código desde Git
echo ""
echo "Actualizando código desde Git..."
git pull

# Reconstruir imágenes
echo ""
echo "Reconstruyendo imágenes Docker..."
docker compose build

# Detener servicios
echo ""
echo "Deteniendo servicios..."
docker compose down

# Levantar servicios con nuevas imágenes
echo ""
echo "Levantando servicios actualizados..."
docker compose up -d

# Esperar a que los servicios estén listos
echo ""
echo "Esperando a que los servicios estén listos..."
sleep 10

# Verificar que los servicios están corriendo
echo ""
echo "Verificando servicios..."
docker ps

# Verificar salud de la API
echo ""
echo "Verificando salud de la API..."
sleep 5
if curl -f http://localhost:8000/api/v1/health &> /dev/null; then
    echo "✓ API está respondiendo correctamente"
else
    echo "⚠ API no está respondiendo, revisa los logs: docker compose logs api"
fi

echo ""
echo "========================================="
echo "Actualización completada"
echo "========================================="
echo ""
echo "Para ver logs:"
echo "  docker compose logs -f"
echo ""

