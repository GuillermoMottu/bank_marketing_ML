# Script de deployment para Railway CLI (PowerShell)
# Uso: .\railway-cli-deploy.ps1 [api|dashboard|all]

param(
    [Parameter(Position=0)]
    [ValidateSet("api", "dashboard", "all")]
    [string]$Service = "all"
)

$ErrorActionPreference = "Stop"

Write-Host "=========================================" -ForegroundColor Cyan
Write-Host "Deployment con Railway CLI" -ForegroundColor Cyan
Write-Host "=========================================" -ForegroundColor Cyan
Write-Host ""

# Verificar que Railway CLI está instalado
try {
    $null = railway --version 2>&1
} catch {
    Write-Host "❌ Railway CLI no está instalado" -ForegroundColor Red
    Write-Host "Instala con: npm i -g @railway/cli" -ForegroundColor Yellow
    exit 1
}

# Verificar que está autenticado
try {
    $null = railway whoami 2>&1
} catch {
    Write-Host "⚠️  No estás autenticado en Railway" -ForegroundColor Yellow
    Write-Host "Ejecuta: railway login" -ForegroundColor Yellow
    exit 1
}

Write-Host "✅ Railway CLI está instalado y autenticado" -ForegroundColor Green
Write-Host ""

# Función para desplegar API
function Deploy-Api {
    Write-Host "🚀 Desplegando API..." -ForegroundColor Cyan
    
    # Vincular proyecto si no está vinculado
    if (-not (Test-Path ".railway\project.json")) {
        Write-Host "Vinculando proyecto..." -ForegroundColor Yellow
        railway link
    }
    
    # Configurar variables de entorno para API
    Write-Host "Configurando variables de entorno para API..." -ForegroundColor Yellow
    
    # Variables de base de datos (se referencian desde el servicio PostgreSQL)
    railway variables --set "DB_HOST=`${{Postgres.PGHOST}}" --service api 2>$null
    railway variables --set "DB_PORT=`${{Postgres.PGPORT}}" --service api 2>$null
    railway variables --set "DB_NAME=`${{Postgres.PGDATABASE}}" --service api 2>$null
    railway variables --set "DB_USER=`${{Postgres.PGUSER}}" --service api 2>$null
    railway variables --set "DB_PASSWORD=`${{Postgres.PGPASSWORD}}" --service api 2>$null
    railway variables --set "DATABASE_URL=`${{Postgres.DATABASE_URL}}" --service api 2>$null
    
    # Variables de API
    railway variables --set "API_HOST=0.0.0.0" --service api 2>$null
    railway variables --set "PYTHONUNBUFFERED=1" --service api 2>$null
    
    # Desplegar
    Write-Host "Desplegando servicio API..." -ForegroundColor Yellow
    railway up --service api
    
    Write-Host "✅ API desplegada" -ForegroundColor Green
}

# Función para desplegar Dashboard
function Deploy-Dashboard {
    Write-Host "🚀 Desplegando Dashboard..." -ForegroundColor Cyan
    
    # Vincular proyecto si no está vinculado
    if (-not (Test-Path ".railway\project.json")) {
        Write-Host "Vinculando proyecto..." -ForegroundColor Yellow
        railway link
    }
    
    # Obtener URL de la API (necesita ser configurada manualmente después del primer deploy)
    Write-Host "⚠️  IMPORTANTE: Después del primer deploy, configura API_BASE_URL con la URL pública de la API" -ForegroundColor Yellow
    Write-Host "   Ejemplo: railway variables --set 'API_BASE_URL=https://api-production-xxxx.up.railway.app/api/v1' --service dashboard" -ForegroundColor Yellow
    
    # Variables de base de datos
    railway variables --set "DB_HOST=`${{Postgres.PGHOST}}" --service dashboard 2>$null
    railway variables --set "DB_PORT=`${{Postgres.PGPORT}}" --service dashboard 2>$null
    railway variables --set "DB_NAME=`${{Postgres.PGDATABASE}}" --service dashboard 2>$null
    railway variables --set "DB_USER=`${{Postgres.PGUSER}}" --service dashboard 2>$null
    railway variables --set "DB_PASSWORD=`${{Postgres.PGPASSWORD}}" --service dashboard 2>$null
    
    # Variables de Dashboard
    railway variables --set "DASHBOARD_HOST=0.0.0.0" --service dashboard 2>$null
    railway variables --set "PYTHONUNBUFFERED=1" --service dashboard 2>$null
    
    # Desplegar
    Write-Host "Desplegando servicio Dashboard..." -ForegroundColor Yellow
    railway up --service dashboard
    
    Write-Host "✅ Dashboard desplegado" -ForegroundColor Green
}

# Ejecutar según el servicio solicitado
switch ($Service) {
    "api" {
        Deploy-Api
    }
    "dashboard" {
        Deploy-Dashboard
    }
    "all" {
        Deploy-Api
        Write-Host ""
        Deploy-Dashboard
    }
}

Write-Host ""
Write-Host "=========================================" -ForegroundColor Cyan
Write-Host "✅ Deployment completado" -ForegroundColor Green
Write-Host "=========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Próximos pasos:" -ForegroundColor Yellow
Write-Host "1. Verifica los logs: railway logs --service [api|dashboard]"
Write-Host "2. Obtén las URLs públicas: railway domain"
Write-Host "3. Configura API_BASE_URL en el dashboard con la URL de la API"
Write-Host ""

