#!/bin/bash
# Script de despliegue automatizado en AWS EC2
# Requiere AWS CLI configurado

set -e

# Colores para output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuración (ajusta según tus necesidades)
KEY_NAME="tu-clave-ec2"  # Nombre de tu par de claves en AWS
INSTANCE_TYPE="t3.medium"
AMI_ID="ami-0c55b159cbfafe1f0"  # Amazon Linux 2023 (ajusta según región)
SECURITY_GROUP_NAME="bank-marketing-sg"
REGION="us-east-1"  # Ajusta según tu región preferida

echo -e "${GREEN}=========================================${NC}"
echo -e "${GREEN}Despliegue Automatizado en AWS EC2${NC}"
echo -e "${GREEN}=========================================${NC}"

# Verificar que AWS CLI está instalado
if ! command -v aws &> /dev/null; then
    echo -e "${RED}Error: AWS CLI no está instalado${NC}"
    echo "Instala AWS CLI: https://aws.amazon.com/cli/"
    exit 1
fi

# Verificar configuración de AWS CLI
if ! aws sts get-caller-identity &> /dev/null; then
    echo -e "${RED}Error: AWS CLI no está configurado${NC}"
    echo "Ejecuta: aws configure"
    exit 1
fi

echo -e "${YELLOW}Configuración detectada:${NC}"
echo "  Región: $REGION"
echo "  Tipo de instancia: $INSTANCE_TYPE"
echo "  Par de claves: $KEY_NAME"
echo ""

# Obtener IP pública del usuario
MY_IP=$(curl -s https://checkip.amazonaws.com)
echo -e "${YELLOW}Tu IP pública: $MY_IP${NC}"
echo ""

# Crear Security Group
echo -e "${YELLOW}Creando Security Group...${NC}"
SG_ID=$(aws ec2 create-security-group \
    --group-name "$SECURITY_GROUP_NAME" \
    --description "Security group para Bank Marketing ML" \
    --region "$REGION" \
    --query 'GroupId' \
    --output text 2>/dev/null || \
    aws ec2 describe-security-groups \
    --group-names "$SECURITY_GROUP_NAME" \
    --region "$REGION" \
    --query 'SecurityGroups[0].GroupId' \
    --output text)

echo -e "${GREEN}Security Group ID: $SG_ID${NC}"

# Configurar reglas del Security Group
echo -e "${YELLOW}Configurando reglas del Security Group...${NC}"

# SSH solo desde tu IP
aws ec2 authorize-security-group-ingress \
    --group-id "$SG_ID" \
    --protocol tcp \
    --port 22 \
    --cidr "$MY_IP/32" \
    --region "$REGION" 2>/dev/null || echo "Regla SSH ya existe"

# HTTP
aws ec2 authorize-security-group-ingress \
    --group-id "$SG_ID" \
    --protocol tcp \
    --port 80 \
    --cidr "0.0.0.0/0" \
    --region "$REGION" 2>/dev/null || echo "Regla HTTP ya existe"

# HTTPS
aws ec2 authorize-security-group-ingress \
    --group-id "$SG_ID" \
    --protocol tcp \
    --port 443 \
    --cidr "0.0.0.0/0" \
    --region "$REGION" 2>/dev/null || echo "Regla HTTPS ya existe"

# API (puerto 8000)
aws ec2 authorize-security-group-ingress \
    --group-id "$SG_ID" \
    --protocol tcp \
    --port 8000 \
    --cidr "0.0.0.0/0" \
    --region "$REGION" 2>/dev/null || echo "Regla API ya existe"

# Dashboard (puerto 8050)
aws ec2 authorize-security-group-ingress \
    --group-id "$SG_ID" \
    --protocol tcp \
    --port 8050 \
    --cidr "0.0.0.0/0" \
    --region "$REGION" 2>/dev/null || echo "Regla Dashboard ya existe"

# Crear script user-data
cat > /tmp/user-data.sh << 'USERDATA'
#!/bin/bash
# User data script para inicialización automática

# Actualizar sistema
yum update -y

# Instalar Docker
yum install -y docker
systemctl start docker
systemctl enable docker
usermod -aG docker ec2-user

# Instalar Docker Compose
curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
chmod +x /usr/local/bin/docker-compose

# Instalar Git
yum install -y git

# Instalar herramientas
yum install -y htop wget curl

# Crear directorio para la aplicación
mkdir -p /home/ec2-user/app
chown ec2-user:ec2-user /home/ec2-user/app
USERDATA

# Lanzar instancia EC2
echo -e "${YELLOW}Lanzando instancia EC2...${NC}"
INSTANCE_ID=$(aws ec2 run-instances \
    --image-id "$AMI_ID" \
    --instance-type "$INSTANCE_TYPE" \
    --key-name "$KEY_NAME" \
    --security-group-ids "$SG_ID" \
    --user-data file:///tmp/user-data.sh \
    --region "$REGION" \
    --query 'Instances[0].InstanceId' \
    --output text)

echo -e "${GREEN}Instancia creada: $INSTANCE_ID${NC}"

# Esperar a que la instancia esté corriendo
echo -e "${YELLOW}Esperando a que la instancia esté lista...${NC}"
aws ec2 wait instance-running --instance-ids "$INSTANCE_ID" --region "$REGION"

# Obtener IP pública
PUBLIC_IP=$(aws ec2 describe-instances \
    --instance-ids "$INSTANCE_ID" \
    --region "$REGION" \
    --query 'Reservations[0].Instances[0].PublicIpAddress' \
    --output text)

echo -e "${GREEN}IP Pública: $PUBLIC_IP${NC}"

# Esperar a que el sistema esté listo (SSH disponible)
echo -e "${YELLOW}Esperando a que SSH esté disponible...${NC}"
sleep 30

# Intentar conexión SSH (máximo 10 intentos)
MAX_ATTEMPTS=10
ATTEMPT=0
while [ $ATTEMPT -lt $MAX_ATTEMPTS ]; do
    if ssh -i ~/.ssh/${KEY_NAME}.pem -o StrictHostKeyChecking=no -o ConnectTimeout=5 ec2-user@$PUBLIC_IP "echo 'SSH ready'" &> /dev/null; then
        echo -e "${GREEN}SSH está disponible${NC}"
        break
    fi
    ATTEMPT=$((ATTEMPT + 1))
    echo "Intento $ATTEMPT/$MAX_ATTEMPTS..."
    sleep 10
done

# Limpiar archivo temporal
rm -f /tmp/user-data.sh

echo ""
echo -e "${GREEN}=========================================${NC}"
echo -e "${GREEN}Despliegue completado!${NC}"
echo -e "${GREEN}=========================================${NC}"
echo ""
echo "Detalles de la instancia:"
echo "  Instance ID: $INSTANCE_ID"
echo "  IP Pública: $PUBLIC_IP"
echo "  Security Group: $SG_ID"
echo ""
echo "Próximos pasos:"
echo "1. Conéctate a la instancia:"
echo "   ssh -i ~/.ssh/${KEY_NAME}.pem ec2-user@$PUBLIC_IP"
echo ""
echo "2. Clona el repositorio:"
echo "   git clone <tu-repo-url>"
echo ""
echo "3. Configura el archivo .env"
echo ""
echo "4. Ejecuta: docker compose up -d --build"
echo ""
echo "5. Entrena el modelo:"
echo "   docker exec bank_marketing_api python train_model.py"
echo ""



