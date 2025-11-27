#!/bin/bash
# Script para instalar y configurar Nginx como reverse proxy

set -e

echo "========================================="
echo "Instalando y configurando Nginx"
echo "========================================="

# Detectar sistema operativo
if [ -f /etc/os-release ]; then
    . /etc/os-release
    OS=$ID
else
    echo "No se pudo detectar el sistema operativo"
    exit 1
fi

# Instalar Nginx
case $OS in
    "amzn"|"amazon")
        echo "Instalando Nginx en Amazon Linux..."
        sudo yum install -y nginx
        ;;
    "ubuntu")
        echo "Instalando Nginx en Ubuntu..."
        sudo apt update
        sudo apt install -y nginx
        ;;
    *)
        echo "Sistema operativo no soportado: $OS"
        exit 1
        ;;
esac

# Verificar que Nginx se instaló correctamente
if ! command -v nginx &> /dev/null; then
    echo "Error: Nginx no se pudo instalar"
    exit 1
fi

echo "✓ Nginx instalado"

# Copiar configuración
echo ""
echo "Configurando Nginx..."
if [ -f "scripts/aws/nginx_config.conf" ]; then
    sudo cp scripts/aws/nginx_config.conf /etc/nginx/conf.d/bank_marketing.conf
    echo "✓ Configuración copiada"
    
    # Pedir dominio o IP
    echo ""
    read -p "Ingresa tu dominio o IP pública (o presiona Enter para usar IP actual): " DOMAIN_OR_IP
    
    if [ -z "$DOMAIN_OR_IP" ]; then
        DOMAIN_OR_IP=$(curl -s https://checkip.amazonaws.com)
        echo "Usando IP pública: $DOMAIN_OR_IP"
    fi
    
    # Reemplazar placeholder en configuración
    sudo sed -i "s/tu-dominio.com/$DOMAIN_OR_IP/g" /etc/nginx/conf.d/bank_marketing.conf
    
    # Verificar configuración
    echo ""
    echo "Verificando configuración de Nginx..."
    if sudo nginx -t; then
        echo "✓ Configuración válida"
    else
        echo "✗ Error en la configuración"
        exit 1
    fi
    
    # Iniciar y habilitar Nginx
    echo ""
    echo "Iniciando Nginx..."
    sudo systemctl start nginx
    sudo systemctl enable nginx
    
    echo ""
    echo "========================================="
    echo "Nginx configurado exitosamente"
    echo "========================================="
    echo ""
    echo "Tu aplicación ahora está disponible en:"
    echo "  http://$DOMAIN_OR_IP"
    echo "  http://$DOMAIN_OR_IP/api/"
    echo "  http://$DOMAIN_OR_IP/docs"
    echo ""
    echo "Para configurar SSL con Let's Encrypt:"
    echo "  sudo certbot --nginx -d $DOMAIN_OR_IP"
    echo ""
else
    echo "✗ No se encontró el archivo de configuración nginx_config.conf"
    echo "  Crea la configuración manualmente en /etc/nginx/conf.d/bank_marketing.conf"
    exit 1
fi




