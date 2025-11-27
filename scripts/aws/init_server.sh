#!/bin/bash
# Script de inicialización del servidor EC2 para Bank Marketing ML
# Compatible con Amazon Linux 2023 y Ubuntu 22.04

set -e  # Salir si hay algún error

echo "========================================="
echo "Inicializando servidor para Bank Marketing ML"
echo "========================================="

# Detectar sistema operativo
if [ -f /etc/os-release ]; then
    . /etc/os-release
    OS=$ID
else
    echo "No se pudo detectar el sistema operativo"
    exit 1
fi

echo "Sistema operativo detectado: $OS"

# Función para instalar en Amazon Linux
install_amazon_linux() {
    echo "Instalando dependencias en Amazon Linux..."
    
    # Actualizar sistema
    sudo yum update -y
    
    # Instalar Docker
    if ! command -v docker &> /dev/null; then
        echo "Instalando Docker..."
        sudo yum install -y docker
        sudo systemctl start docker
        sudo systemctl enable docker
    else
        echo "Docker ya está instalado"
    fi
    
    # Agregar usuario al grupo docker
    sudo usermod -aG docker $USER
    
    # Instalar Docker Compose
    if ! command -v docker-compose &> /dev/null; then
        echo "Instalando Docker Compose..."
        sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
        sudo chmod +x /usr/local/bin/docker-compose
    else
        echo "Docker Compose ya está instalado"
    fi
    
    # Instalar Git si no está
    if ! command -v git &> /dev/null; then
        sudo yum install -y git
    fi
    
    # Instalar herramientas adicionales
    sudo yum install -y htop wget curl
}

# Función para instalar en Ubuntu
install_ubuntu() {
    echo "Instalando dependencias en Ubuntu..."
    
    # Actualizar sistema
    sudo apt update && sudo apt upgrade -y
    
    # Instalar Docker
    if ! command -v docker &> /dev/null; then
        echo "Instalando Docker..."
        sudo apt install -y docker.io
        sudo systemctl start docker
        sudo systemctl enable docker
    else
        echo "Docker ya está instalado"
    fi
    
    # Agregar usuario al grupo docker
    sudo usermod -aG docker $USER
    
    # Instalar Docker Compose
    if ! command -v docker-compose &> /dev/null; then
        echo "Instalando Docker Compose..."
        sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
        sudo chmod +x /usr/local/bin/docker-compose
    else
        echo "Docker Compose ya está instalado"
    fi
    
    # Instalar Git si no está
    if ! command -v git &> /dev/null; then
        sudo apt install -y git
    fi
    
    # Instalar herramientas adicionales
    sudo apt install -y htop wget curl
}

# Instalar según el sistema operativo
case $OS in
    "amzn"|"amazon")
        install_amazon_linux
        ;;
    "ubuntu")
        install_ubuntu
        ;;
    *)
        echo "Sistema operativo no soportado: $OS"
        echo "Por favor, instala Docker y Docker Compose manualmente"
        exit 1
        ;;
esac

# Verificar instalaciones
echo ""
echo "========================================="
echo "Verificando instalaciones..."
echo "========================================="

if command -v docker &> /dev/null; then
    echo "✓ Docker instalado: $(docker --version)"
else
    echo "✗ Docker no se pudo instalar"
    exit 1
fi

if command -v docker-compose &> /dev/null; then
    echo "✓ Docker Compose instalado: $(docker-compose --version)"
else
    echo "✗ Docker Compose no se pudo instalar"
    exit 1
fi

if command -v git &> /dev/null; then
    echo "✓ Git instalado: $(git --version)"
else
    echo "✗ Git no se pudo instalar"
    exit 1
fi

echo ""
echo "========================================="
echo "Configuración completada exitosamente!"
echo "========================================="
echo ""
echo "IMPORTANTE: Necesitas desconectarte y volver a conectarte"
echo "para que los cambios de grupo de usuario surtan efecto."
echo ""
echo "Próximos pasos:"
echo "1. Desconéctate: exit"
echo "2. Vuelve a conectarte"
echo "3. Clona el repositorio: git clone <tu-repo>"
echo "4. Configura el archivo .env"
echo "5. Ejecuta: docker compose up -d --build"
echo ""


