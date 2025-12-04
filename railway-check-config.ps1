# Script para verificar configuración de Railway
# Verifica que los servicios estén configurados correctamente

Write-Host "=========================================" -ForegroundColor Cyan
Write-Host "Verificacion de Configuracion Railway" -ForegroundColor Cyan
Write-Host "=========================================" -ForegroundColor Cyan
Write-Host ""

# Verificar Railway CLI
try {
    $version = railway --version 2>&1
    Write-Host "OK: Railway CLI instalado" -ForegroundColor Green
} catch {
    Write-Host "ERROR: Railway CLI no instalado" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "Verificando servicios..." -ForegroundColor Yellow
Write-Host ""

# Verificar servicio API
Write-Host "Servicio API (DeepLearning):" -ForegroundColor Cyan
try {
    $apiVars = railway variables --service DeepLearning 2>&1
    Write-Host "  OK: Servicio existe" -ForegroundColor Green
    
    # Verificar variables importantes
    if ($apiVars -match "DATABASE_URL") {
        Write-Host "  OK: DATABASE_URL configurada" -ForegroundColor Green
    } else {
        Write-Host "  ADVERTENCIA: DATABASE_URL no configurada" -ForegroundColor Yellow
    }
} catch {
    Write-Host "  ERROR: Servicio no encontrado" -ForegroundColor Red
}

Write-Host ""

# Verificar servicio Dashboard
Write-Host "Servicio Dashboard:" -ForegroundColor Cyan
try {
    $dashboardVars = railway variables --service Dashboard 2>&1
    Write-Host "  OK: Servicio existe" -ForegroundColor Green
    
    # Verificar variables importantes
    if ($dashboardVars -match "DATABASE_URL") {
        Write-Host "  OK: DATABASE_URL configurada" -ForegroundColor Green
    } else {
        Write-Host "  ADVERTENCIA: DATABASE_URL no configurada" -ForegroundColor Yellow
    }
    
    if ($dashboardVars -match "API_BASE_URL") {
        Write-Host "  OK: API_BASE_URL configurada" -ForegroundColor Green
    } else {
        Write-Host "  ADVERTENCIA: API_BASE_URL no configurada" -ForegroundColor Yellow
    }
} catch {
    Write-Host "  ERROR: Servicio no encontrado" -ForegroundColor Red
}

Write-Host ""
Write-Host "Verificando archivos locales..." -ForegroundColor Yellow
Write-Host ""

# Verificar Dockerfiles
if (Test-Path "docker/Dockerfile") {
    Write-Host "  OK: docker/Dockerfile existe" -ForegroundColor Green
} else {
    Write-Host "  ERROR: docker/Dockerfile no existe" -ForegroundColor Red
}

if (Test-Path "docker/Dockerfile.dashboard") {
    Write-Host "  OK: docker/Dockerfile.dashboard existe" -ForegroundColor Green
} else {
    Write-Host "  ERROR: docker/Dockerfile.dashboard no existe" -ForegroundColor Red
}

if (Test-Path "railway.json") {
    Write-Host "  OK: railway.json existe" -ForegroundColor Green
} else {
    Write-Host "  ERROR: railway.json no existe" -ForegroundColor Red
}

Write-Host ""
Write-Host "Verificando rama actual..." -ForegroundColor Yellow
$currentBranch = git branch --show-current
Write-Host "  Rama actual: $currentBranch" -ForegroundColor Cyan

if ($currentBranch -eq "Deep_Learning") {
    Write-Host "  OK: Estas en la rama correcta" -ForegroundColor Green
} else {
    Write-Host "  ADVERTENCIA: No estas en la rama Deep_Learning" -ForegroundColor Yellow
    Write-Host "  Cambia con: git checkout Deep_Learning" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "=========================================" -ForegroundColor Cyan
Write-Host "IMPORTANTE: Configurar rama en Railway" -ForegroundColor Yellow
Write-Host "=========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Para resolver el problema del Dockerfile:" -ForegroundColor Yellow
Write-Host "1. Ve a https://railway.app" -ForegroundColor White
Write-Host "2. Servicio DeepLearning -> Settings -> Source -> Branch: Deep_Learning" -ForegroundColor White
Write-Host "3. Servicio Dashboard -> Settings -> Source -> Branch: Deep_Learning" -ForegroundColor White
Write-Host "4. Verifica Root Directory esta vacio o es '/'" -ForegroundColor White
Write-Host "5. Verifica Dockerfile Path es correcto" -ForegroundColor White
Write-Host ""
Write-Host "Ver: docs/RAILWAY_FIX_DOCKERFILE.md para mas detalles" -ForegroundColor Cyan
Write-Host ""
