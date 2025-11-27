# Script para limpiar credenciales de Git y permitir nueva autenticación

Write-Host "=== Limpieza de Credenciales de Git ===" -ForegroundColor Cyan
Write-Host ""

# Mostrar configuración actual
Write-Host "Configuración actual de Git:" -ForegroundColor Yellow
git config --list | Select-String -Pattern "(user|credential|remote)" | ForEach-Object { Write-Host "  $_" }

Write-Host ""
Write-Host "Credenciales guardadas en Windows Credential Manager:" -ForegroundColor Yellow
cmdkey /list | Select-String -Pattern "github"

Write-Host ""
Write-Host "Para eliminar credenciales manualmente:" -ForegroundColor Cyan
Write-Host "1. Abre 'Administrador de credenciales de Windows'" -ForegroundColor White
Write-Host "2. Ve a 'Credenciales de Windows'" -ForegroundColor White
Write-Host "3. Busca y elimina entradas relacionadas con 'git:https://github.com'" -ForegroundColor White
Write-Host ""

# Opción para eliminar automáticamente (requiere confirmación)
$eliminar = Read-Host "¿Deseas eliminar automáticamente las credenciales de GitHub? (S/N)"
if ($eliminar -eq "S" -or $eliminar -eq "s") {
    # Buscar y eliminar credenciales de GitHub
    $credenciales = cmdkey /list | Select-String -Pattern "github" -Context 0,1
    if ($credenciales) {
        Write-Host "Credenciales encontradas. Eliminando..." -ForegroundColor Yellow
        # Nota: Necesitaríamos parsear el nombre exacto de la credencial
        Write-Host "Por favor, elimina manualmente las credenciales desde el Administrador de credenciales." -ForegroundColor Red
    } else {
        Write-Host "No se encontraron credenciales de GitHub guardadas." -ForegroundColor Green
    }
}

Write-Host ""
Write-Host "Después de eliminar las credenciales:" -ForegroundColor Cyan
Write-Host "1. Intenta hacer push: git push upstream main" -ForegroundColor White
Write-Host "2. Windows te pedirá nuevas credenciales" -ForegroundColor White
Write-Host "3. Usa tu usuario y token de GitHub para 'GuillermoMottu'" -ForegroundColor White
Write-Host ""

Write-Host "Para crear un token de acceso personal:" -ForegroundColor Cyan
Write-Host "https://github.com/settings/tokens" -ForegroundColor Blue
Write-Host "Permisos necesarios: repo (acceso completo a repositorios)" -ForegroundColor White

