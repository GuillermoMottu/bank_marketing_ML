#!/bin/bash
# Script de deployment para Railway CLI
# Uso: ./railway-cli-deploy.sh [api|dashboard|all]

set -e

SERVICE=${1:-all}

echo "========================================="
echo "Deployment con Railway CLI"
echo "========================================="

# Verificar que Railway CLI está instalado
if ! command -v railway &> /dev/null; then
    echo "❌ Railway CLI no está instalado"
    echo "Instala con: npm i -g @railway/cli"
    exit 1
fi

# Verificar que está autenticado
if ! railway whoami &> /dev/null; then
    echo "⚠️  No estás autenticado en Railway"
    echo "Ejecuta: railway login"
    exit 1
fi

echo "✅ Railway CLI está instalado y autenticado"
echo ""

# Función para desplegar API
deploy_api() {
    echo "🚀 Desplegando API..."
    
    # Vincular proyecto si no está vinculado
    if [ ! -f ".railway/project.json" ]; then
        echo "Vinculando proyecto..."
        railway link
    fi
    
    # Configurar variables de entorno para API
    echo "Configurando variables de entorno para API..."
    
    # Variables de base de datos (se referencian desde el servicio PostgreSQL)
    railway variables --set "DB_HOST=\${{Postgres.PGHOST}}" --service api 2>/dev/null || true
    railway variables --set "DB_PORT=\${{Postgres.PGPORT}}" --service api 2>/dev/null || true
    railway variables --set "DB_NAME=\${{Postgres.PGDATABASE}}" --service api 2>/dev/null || true
    railway variables --set "DB_USER=\${{Postgres.PGUSER}}" --service api 2>/dev/null || true
    railway variables --set "DB_PASSWORD=\${{Postgres.PGPASSWORD}}" --service api 2>/dev/null || true
    railway variables --set "DATABASE_URL=\${{Postgres.DATABASE_URL}}" --service api 2>/dev/null || true
    
    # Variables de API
    railway variables --set "API_HOST=0.0.0.0" --service api 2>/dev/null || true
    railway variables --set "PYTHONUNBUFFERED=1" --service api 2>/dev/null || true
    
    # Desplegar
    echo "Desplegando servicio API..."
    railway up --service api
    
    echo "✅ API desplegada"
}

# Función para desplegar Dashboard
deploy_dashboard() {
    echo "🚀 Desplegando Dashboard..."
    
    # Vincular proyecto si no está vinculado
    if [ ! -f ".railway/project.json" ]; then
        echo "Vinculando proyecto..."
        railway link
    fi
    
    # Obtener URL de la API (necesita ser configurada manualmente después del primer deploy)
    echo "⚠️  IMPORTANTE: Después del primer deploy, configura API_BASE_URL con la URL pública de la API"
    echo "   Ejemplo: railway variables --set 'API_BASE_URL=https://api-production-xxxx.up.railway.app/api/v1' --service dashboard"
    
    # Variables de base de datos
    railway variables --set "DB_HOST=\${{Postgres.PGHOST}}" --service dashboard 2>/dev/null || true
    railway variables --set "DB_PORT=\${{Postgres.PGPORT}}" --service dashboard 2>/dev/null || true
    railway variables --set "DB_NAME=\${{Postgres.PGDATABASE}}" --service dashboard 2>/dev/null || true
    railway variables --set "DB_USER=\${{Postgres.PGUSER}}" --service dashboard 2>/dev/null || true
    railway variables --set "DB_PASSWORD=\${{Postgres.PGPASSWORD}}" --service dashboard 2>/dev/null || true
    
    # Variables de Dashboard
    railway variables --set "DASHBOARD_HOST=0.0.0.0" --service dashboard 2>/dev/null || true
    railway variables --set "PYTHONUNBUFFERED=1" --service dashboard 2>/dev/null || true
    
    # Desplegar
    echo "Desplegando servicio Dashboard..."
    railway up --service dashboard
    
    echo "✅ Dashboard desplegado"
}

# Ejecutar según el servicio solicitado
case $SERVICE in
    api)
        deploy_api
        ;;
    dashboard)
        deploy_dashboard
        ;;
    all)
        deploy_api
        echo ""
        deploy_dashboard
        ;;
    *)
        echo "Uso: $0 [api|dashboard|all]"
        exit 1
        ;;
esac

echo ""
echo "========================================="
echo "✅ Deployment completado"
echo "========================================="
echo ""
echo "Próximos pasos:"
echo "1. Verifica los logs: railway logs --service [api|dashboard]"
echo "2. Obtén las URLs públicas: railway domain"
echo "3. Configura API_BASE_URL en el dashboard con la URL de la API"
echo ""

