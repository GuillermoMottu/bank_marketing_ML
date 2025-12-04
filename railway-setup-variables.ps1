# Script de Configuración Automática de Variables en Railway
# Uso: .\railway-setup-variables.ps1 [api|dashboard|all] [postgres-service-name]

param(
    [Parameter(Position=0)]
    [ValidateSet("api", "dashboard", "all")]
    [string]$Service = "all",
    
    [Parameter(Position=1)]
    [string]$PostgresServiceName = "Postgres"
)

$ErrorActionPreference = "Stop"

Write-Host "=========================================" -ForegroundColor Cyan
Write-Host "Configuración de Variables en Railway" -ForegroundColor Cyan
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

# Verificar servicios disponibles
Write-Host "Verificando servicios disponibles..." -ForegroundColor Yellow
$servicesOutput = railway service 2>&1 | Out-String
Write-Host ""

# Función para configurar variables de la API
function Set-ApiVariables {
    param([string]$PostgresName)
    
    Write-Host "🔧 Configurando variables del servicio API (DeepLearning)..." -ForegroundColor Cyan
    
    $apiService = "DeepLearning"
    
    # Verificar que el servicio existe
    try {
        $null = railway variables --service $apiService 2>&1
    } catch {
        Write-Host "⚠️  Advertencia: No se pudo verificar el servicio $apiService" -ForegroundColor Yellow
        Write-Host "   Continuando de todas formas..." -ForegroundColor Yellow
    }
    
    # Variables de base de datos
    Write-Host "  Configurando DATABASE_URL..." -ForegroundColor Gray
    railway variables --set "DATABASE_URL=`${{$PostgresName.DATABASE_URL}}" --service $apiService 2>$null
    
    # Variables de API
    Write-Host "  Configurando API_HOST..." -ForegroundColor Gray
    railway variables --set "API_HOST=0.0.0.0" --service $apiService 2>$null
    
    Write-Host "  Configurando PYTHONUNBUFFERED..." -ForegroundColor Gray
    railway variables --set "PYTHONUNBUFFERED=1" --service $apiService 2>$null
    
    Write-Host "✅ Variables del servicio API configuradas" -ForegroundColor Green
}

# Función para configurar variables del Dashboard
function Set-DashboardVariables {
    param([string]$PostgresName)
    
    Write-Host "🔧 Configurando variables del servicio Dashboard..." -ForegroundColor Cyan
    
    $dashboardService = "dashboard"
    
    # Verificar que el servicio existe
    try {
        $null = railway variables --service $dashboardService 2>&1
    } catch {
        Write-Host "❌ Error: El servicio '$dashboardService' no existe" -ForegroundColor Red
        Write-Host "   Crea el servicio Dashboard primero desde el dashboard web de Railway" -ForegroundColor Yellow
        Write-Host "   Ver: docs/RAILWAY_MANUAL_STEPS.md" -ForegroundColor Yellow
        return $false
    }
    
    # Variables de base de datos
    Write-Host "  Configurando DATABASE_URL..." -ForegroundColor Gray
    railway variables --set "DATABASE_URL=`${{$PostgresName.DATABASE_URL}}" --service $dashboardService 2>$null
    
    # Variables del dashboard
    Write-Host "  Configurando DASHBOARD_HOST..." -ForegroundColor Gray
    railway variables --set "DASHBOARD_HOST=0.0.0.0" --service $dashboardService 2>$null
    
    Write-Host "  Configurando PYTHONUNBUFFERED..." -ForegroundColor Gray
    railway variables --set "PYTHONUNBUFFERED=1" --service $dashboardService 2>$null
    
    Write-Host "✅ Variables del servicio Dashboard configuradas" -ForegroundColor Green
    Write-Host ""
    Write-Host "⚠️  IMPORTANTE: Después del deploy de la API, configura API_BASE_URL:" -ForegroundColor Yellow
    Write-Host "   railway variables --set 'API_BASE_URL=https://tu-api-url.up.railway.app/api/v1' --service dashboard" -ForegroundColor Yellow
    
    return $true
}

# Función para detectar el nombre del servicio PostgreSQL
function Get-PostgresServiceName {
    param([string]$DefaultName)
    
    Write-Host "🔍 Detectando nombre del servicio PostgreSQL..." -ForegroundColor Cyan
    
    # Intentar obtener lista de servicios (puede ser interactivo, así que usamos un enfoque diferente)
    try {
        # Intentar listar servicios sin interacción
        $services = railway service --json 2>&1 | Out-String
    } catch {
        # Si falla, intentar sin json
        $services = railway service 2>&1 | Out-String
    }
    
    # Buscar nombres comunes de PostgreSQL
    $commonNames = @("Postgres", "PostgreSQL", "postgres", "postgresql", "Postgresql")
    
    foreach ($name in $commonNames) {
        if ($services -match $name) {
            Write-Host "  ✓ Servicio PostgreSQL detectado: $name" -ForegroundColor Green
            return $name
        }
    }
    
    Write-Host "  ⚠️  No se pudo detectar automáticamente el servicio PostgreSQL" -ForegroundColor Yellow
    Write-Host "  Usando nombre por defecto: $DefaultName" -ForegroundColor Yellow
    Write-Host "  Si es incorrecto, especifica el nombre con: -PostgresServiceName 'NombreServicio'" -ForegroundColor Yellow
    
    return $DefaultName
}

# Detectar nombre del servicio PostgreSQL si no se especificó
if (-not $PostgresServiceName -or $PostgresServiceName -eq "Postgres") {
    $detectedName = Get-PostgresServiceName -DefaultName $PostgresServiceName
    if ($detectedName) {
        $PostgresServiceName = $detectedName
    }
}

Write-Host ""
Write-Host "Usando servicio PostgreSQL: $PostgresServiceName" -ForegroundColor Cyan
Write-Host ""

# Ejecutar según el servicio solicitado
$success = $true

switch ($Service) {
    "api" {
        Set-ApiVariables -PostgresName $PostgresServiceName
    }
    "dashboard" {
        $result = Set-DashboardVariables -PostgresName $PostgresServiceName
        if (-not $result) {
            $success = $false
        }
    }
    "all" {
        Set-ApiVariables -PostgresName $PostgresServiceName
        Write-Host ""
        $result = Set-DashboardVariables -PostgresName $PostgresServiceName
        if (-not $result) {
            $success = $false
        }
    }
}

Write-Host ""
Write-Host "=========================================" -ForegroundColor Cyan

if ($success) {
    Write-Host "✅ Configuración completada" -ForegroundColor Green
} else {
    Write-Host "⚠️  Configuración completada con advertencias" -ForegroundColor Yellow
}

Write-Host "=========================================" -ForegroundColor Cyan
Write-Host ""

# Mostrar resumen
Write-Host "Resumen de variables configuradas:" -ForegroundColor Yellow
Write-Host ""

if ($Service -eq "api" -or $Service -eq "all") {
    Write-Host "Servicio API (DeepLearning):" -ForegroundColor Cyan
    railway variables --service DeepLearning --json | ConvertFrom-Json | Format-Table -AutoSize
    Write-Host ""
}

if ($Service -eq "dashboard" -or $Service -eq "all") {
    Write-Host "Servicio Dashboard:" -ForegroundColor Cyan
    try {
        railway variables --service dashboard --json | ConvertFrom-Json | Format-Table -AutoSize
    } catch {
        Write-Host "  (Servicio dashboard no disponible)" -ForegroundColor Gray
    }
    Write-Host ""
}

Write-Host "Próximos pasos:" -ForegroundColor Yellow
Write-Host "1. Verifica que los servicios PostgreSQL, API y Dashboard existen" -ForegroundColor White
Write-Host "2. Si falta algún servicio, créalo desde el dashboard web de Railway" -ForegroundColor White
Write-Host "3. Haz push de los modelos a GitHub: git push" -ForegroundColor White
Write-Host "4. Railway desplegará automáticamente o usa: railway up" -ForegroundColor White
Write-Host "5. Después del deploy, configura API_BASE_URL en el dashboard" -ForegroundColor White
Write-Host ""

