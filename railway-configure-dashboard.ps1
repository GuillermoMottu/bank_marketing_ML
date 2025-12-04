# Script para configurar variables del Dashboard después de crearlo
# Uso: .\railway-configure-dashboard.ps1 [postgres-service-name]

param(
    [Parameter(Position=0)]
    [string]$PostgresServiceName = "Postgres"
)

$ErrorActionPreference = "Stop"

Write-Host "=========================================" -ForegroundColor Cyan
Write-Host "Configuracion de Variables del Dashboard" -ForegroundColor Cyan
Write-Host "=========================================" -ForegroundColor Cyan
Write-Host ""

# Verificar que Railway CLI esta instalado
try {
    $null = railway --version 2>&1
} catch {
    Write-Host "ERROR: Railway CLI no esta instalado" -ForegroundColor Red
    Write-Host "Instala con: npm i -g @railway/cli" -ForegroundColor Yellow
    exit 1
}

# Verificar que esta autenticado
try {
    $null = railway whoami 2>&1
} catch {
    Write-Host "ADVERTENCIA: No estas autenticado en Railway" -ForegroundColor Yellow
    Write-Host "Ejecuta: railway login" -ForegroundColor Yellow
    exit 1
}

Write-Host "OK: Railway CLI esta instalado y autenticado" -ForegroundColor Green
Write-Host ""

# Verificar que el servicio Dashboard existe
Write-Host "Verificando que el servicio Dashboard existe..." -ForegroundColor Yellow
try {
    $null = railway variables --service dashboard 2>&1
    Write-Host "OK: Servicio Dashboard encontrado" -ForegroundColor Green
} catch {
    Write-Host "ERROR: El servicio 'dashboard' no existe" -ForegroundColor Red
    Write-Host ""
    Write-Host "Crea el servicio Dashboard primero:" -ForegroundColor Yellow
    Write-Host "1. Ve a https://railway.app" -ForegroundColor White
    Write-Host "2. Proyecto DeepLearning -> + New -> GitHub Repo" -ForegroundColor White
    Write-Host "3. Selecciona el repositorio bank_marketing_ML" -ForegroundColor White
    Write-Host "4. Settings -> Name: dashboard" -ForegroundColor White
    Write-Host "5. Settings -> Dockerfile Path: docker/Dockerfile.dashboard" -ForegroundColor White
    Write-Host "6. Settings -> Start Command: python run_dashboard.py" -ForegroundColor White
    Write-Host ""
    Write-Host "Ver: docs/RAILWAY_MANUAL_STEPS.md para mas detalles" -ForegroundColor Yellow
    exit 1
}

Write-Host ""
Write-Host "Configurando variables del Dashboard..." -ForegroundColor Cyan
Write-Host ""

# Variables de base de datos
Write-Host "  Configurando DATABASE_URL..." -ForegroundColor Gray
railway variables --set "DATABASE_URL=`${{$PostgresServiceName.DATABASE_URL}}" --service dashboard 2>$null
if ($LASTEXITCODE -eq 0) {
    Write-Host "    OK: DATABASE_URL configurada" -ForegroundColor Green
} else {
    Write-Host "    ADVERTENCIA: Error al configurar DATABASE_URL" -ForegroundColor Yellow
}

# Variables del dashboard
Write-Host "  Configurando DASHBOARD_HOST..." -ForegroundColor Gray
railway variables --set "DASHBOARD_HOST=0.0.0.0" --service dashboard 2>$null
if ($LASTEXITCODE -eq 0) {
    Write-Host "    OK: DASHBOARD_HOST configurada" -ForegroundColor Green
} else {
    Write-Host "    ADVERTENCIA: Error al configurar DASHBOARD_HOST" -ForegroundColor Yellow
}

Write-Host "  Configurando PYTHONUNBUFFERED..." -ForegroundColor Gray
railway variables --set "PYTHONUNBUFFERED=1" --service dashboard 2>$null
if ($LASTEXITCODE -eq 0) {
    Write-Host "    OK: PYTHONUNBUFFERED configurada" -ForegroundColor Green
} else {
    Write-Host "    ADVERTENCIA: Error al configurar PYTHONUNBUFFERED" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "=========================================" -ForegroundColor Cyan
Write-Host "OK: Variables del Dashboard configuradas" -ForegroundColor Green
Write-Host "=========================================" -ForegroundColor Cyan
Write-Host ""

# Obtener URL de la API
Write-Host "Obteniendo URL de la API..." -ForegroundColor Yellow
$apiDomainOutput = railway domain --service DeepLearning 2>&1
$apiUrl = $apiDomainOutput | Select-String -Pattern "https://" | ForEach-Object { $_.Line.Trim() }

if ($apiUrl) {
    $apiBaseUrl = "$apiUrl/api/v1"
    Write-Host ""
    Write-Host "IMPORTANTE: Configura API_BASE_URL despues del deploy:" -ForegroundColor Yellow
    Write-Host "   railway variables --set API_BASE_URL=$apiBaseUrl --service dashboard" -ForegroundColor Cyan
    Write-Host ""
} else {
    Write-Host "  (No se pudo obtener la URL de la API automaticamente)" -ForegroundColor Gray
    Write-Host "  Configura manualmente con:" -ForegroundColor Yellow
    Write-Host "  railway variables --set API_BASE_URL=https://tu-api-url.up.railway.app/api/v1 --service dashboard" -ForegroundColor Cyan
}

Write-Host ""
Write-Host "Verificar variables configuradas:" -ForegroundColor Yellow
Write-Host "  railway variables --service dashboard" -ForegroundColor Cyan
Write-Host ""
