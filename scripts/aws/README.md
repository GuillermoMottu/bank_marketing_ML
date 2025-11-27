# Scripts de Despliegue en AWS

Este directorio contiene scripts para facilitar el despliegue del proyecto Bank Marketing ML en AWS EC2.

## 📁 Archivos

### `init_server.sh`
Script de inicialización del servidor EC2. Instala Docker, Docker Compose y dependencias necesarias.

**Uso:**
```bash
# En la instancia EC2
curl -O https://raw.githubusercontent.com/tu-repo/bank_marketing_ML/main/scripts/aws/init_server.sh
chmod +x init_server.sh
sudo ./init_server.sh
```

### `deploy.sh`
Script de despliegue automatizado que crea la instancia EC2, configura Security Groups y prepara el servidor.

**Uso:**
```bash
# En tu máquina local (requiere AWS CLI configurado)
chmod +x scripts/aws/deploy.sh
./scripts/aws/deploy.sh
```

**Configuración:**
Edita las variables al inicio del script:
- `KEY_NAME`: Nombre de tu par de claves EC2
- `INSTANCE_TYPE`: Tipo de instancia (t3.medium recomendado)
- `AMI_ID`: ID de la AMI (ajusta según tu región)
- `REGION`: Región de AWS

### `setup_production.sh`
Configura el proyecto en producción después de clonar el repositorio.

**Uso:**
```bash
# En la instancia EC2, después de clonar el repositorio
cd bank_marketing_ML
chmod +x scripts/aws/setup_production.sh
./scripts/aws/setup_production.sh
```

Este script:
- Crea el archivo `.env` con configuración de producción
- Genera un password seguro para PostgreSQL
- Crea directorios necesarios
- Construye y levanta los servicios Docker
- Entrena el modelo

### `backup.sh`
Script de backup para base de datos y modelos.

**Uso manual:**
```bash
chmod +x scripts/aws/backup.sh
./scripts/aws/backup.sh
```

**Configurar backup automático (diario a las 2 AM):**
```bash
(crontab -l 2>/dev/null; echo "0 2 * * * /home/ec2-user/bank_marketing_ML/scripts/aws/backup.sh") | crontab -
```

### `update.sh`
Script para actualizar el proyecto en producción.

**Uso:**
```bash
# En la instancia EC2
cd bank_marketing_ML
chmod +x scripts/aws/update.sh
./scripts/aws/update.sh
```

Este script:
- Hace backup antes de actualizar
- Actualiza el código desde Git
- Reconstruye las imágenes Docker
- Reinicia los servicios
- Verifica que todo esté funcionando

## 🔒 Seguridad

**IMPORTANTE**: 
- Nunca subas el archivo `.env` al repositorio
- Usa passwords seguros generados con `openssl rand -base64 32`
- Limita el acceso SSH solo a tu IP en el Security Group
- Considera usar AWS Secrets Manager para passwords en producción

## 📝 Notas

- Todos los scripts están diseñados para ejecutarse en Linux (Amazon Linux o Ubuntu)
- Los scripts asumen que tienes permisos adecuados
- Algunos scripts requieren ejecutarse con `sudo`
- Los backups se guardan en `/home/ec2-user/backups` por defecto

