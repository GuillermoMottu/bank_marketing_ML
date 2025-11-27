#!/bin/bash
# Script de backup para base de datos y modelos
# Ejecutar manualmente o configurar en crontab

set -e

BACKUP_DIR="/home/ec2-user/backups"
DATE=$(date +%Y%m%d_%H%M%S)
RETENTION_DAYS=7

echo "========================================="
echo "Iniciando backup - $(date)"
echo "========================================="

# Crear directorio de backups si no existe
mkdir -p "$BACKUP_DIR"

# Backup de base de datos
echo "Haciendo backup de la base de datos..."
DB_BACKUP_FILE="$BACKUP_DIR/db_backup_$DATE.sql"
docker exec bank_marketing_db pg_dump -U postgres bank_marketing > "$DB_BACKUP_FILE"

if [ -f "$DB_BACKUP_FILE" ] && [ -s "$DB_BACKUP_FILE" ]; then
    echo "✓ Backup de base de datos creado: $DB_BACKUP_FILE"
    # Comprimir backup
    gzip "$DB_BACKUP_FILE"
    echo "✓ Backup comprimido: ${DB_BACKUP_FILE}.gz"
else
    echo "✗ Error al crear backup de base de datos"
    exit 1
fi

# Backup de modelos
echo ""
echo "Haciendo backup de modelos..."
MODELS_BACKUP_FILE="$BACKUP_DIR/models_backup_$DATE.tar.gz"
docker run --rm \
    -v bank_marketing_ml_models:/data \
    -v "$BACKUP_DIR":/backup \
    alpine tar czf "/backup/models_backup_$DATE.tar.gz" -C /data .

if [ -f "$MODELS_BACKUP_FILE" ]; then
    echo "✓ Backup de modelos creado: $MODELS_BACKUP_FILE"
else
    echo "✗ Error al crear backup de modelos"
    exit 1
fi

# Limpiar backups antiguos
echo ""
echo "Limpiando backups antiguos (más de $RETENTION_DAYS días)..."
find "$BACKUP_DIR" -name "*.sql.gz" -mtime +$RETENTION_DAYS -delete
find "$BACKUP_DIR" -name "models_backup_*.tar.gz" -mtime +$RETENTION_DAYS -delete
echo "✓ Limpieza completada"

# Mostrar tamaño de backups
echo ""
echo "Resumen de backups:"
du -h "$BACKUP_DIR"/*.gz 2>/dev/null | tail -1 || echo "No hay backups previos"

echo ""
echo "========================================="
echo "Backup completado exitosamente"
echo "========================================="

