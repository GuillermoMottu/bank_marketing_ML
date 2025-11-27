#!/bin/bash
# Script de verificación de salud del sistema
# Verifica que todos los servicios estén funcionando correctamente

set -e

echo "========================================="
echo "Verificación de Salud del Sistema"
echo "========================================="
echo ""

# Colores
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

ERRORS=0

# Verificar Docker
echo -n "Verificando Docker... "
if command -v docker &> /dev/null && docker ps &> /dev/null; then
    echo -e "${GREEN}✓${NC}"
else
    echo -e "${RED}✗${NC}"
    echo "  Docker no está instalado o no está corriendo"
    ERRORS=$((ERRORS + 1))
fi

# Verificar contenedores
echo -n "Verificando contenedores... "
if docker ps | grep -q bank_marketing; then
    echo -e "${GREEN}✓${NC}"
    echo "  Contenedores activos:"
    docker ps --format "table {{.Names}}\t{{.Status}}" | grep bank_marketing
else
    echo -e "${RED}✗${NC}"
    echo "  No se encontraron contenedores de bank_marketing"
    ERRORS=$((ERRORS + 1))
fi

# Verificar API
echo ""
echo -n "Verificando API (http://localhost:8000/api/v1/health)... "
if curl -f -s http://localhost:8000/api/v1/health > /dev/null; then
    echo -e "${GREEN}✓${NC}"
    API_RESPONSE=$(curl -s http://localhost:8000/api/v1/health)
    echo "  Respuesta: $API_RESPONSE"
else
    echo -e "${RED}✗${NC}"
    echo "  La API no está respondiendo"
    ERRORS=$((ERRORS + 1))
fi

# Verificar Dashboard
echo ""
echo -n "Verificando Dashboard (http://localhost:8050)... "
if curl -f -s http://localhost:8050 > /dev/null; then
    echo -e "${GREEN}✓${NC}"
else
    echo -e "${RED}✗${NC}"
    echo "  El Dashboard no está respondiendo"
    ERRORS=$((ERRORS + 1))
fi

# Verificar base de datos
echo ""
echo -n "Verificando conexión a base de datos... "
if docker exec bank_marketing_db pg_isready -U postgres > /dev/null 2>&1; then
    echo -e "${GREEN}✓${NC}"
else
    echo -e "${RED}✗${NC}"
    echo "  No se puede conectar a la base de datos"
    ERRORS=$((ERRORS + 1))
fi

# Verificar modelos
echo ""
echo -n "Verificando modelos... "
if docker exec bank_marketing_api test -f /app/models/decision_tree_model.pkl && \
   docker exec bank_marketing_api test -f /app/models/preprocessing_pipeline.pkl; then
    echo -e "${GREEN}✓${NC}"
    echo "  Modelos encontrados:"
    docker exec bank_marketing_api ls -lh /app/models/*.pkl
else
    echo -e "${YELLOW}⚠${NC}"
    echo "  Los modelos no se encontraron. Ejecuta: docker exec bank_marketing_api python train_model.py"
fi

# Verificar espacio en disco
echo ""
echo "Verificando espacio en disco..."
df -h / | tail -1 | awk '{print "  Espacio disponible: " $4 " de " $2 " (" $5 " usado)"}'

# Verificar memoria
echo ""
echo "Verificando memoria..."
free -h | grep Mem | awk '{print "  Memoria disponible: " $7 " de " $2}'

# Verificar logs recientes de errores
echo ""
echo "Verificando errores recientes en logs..."
ERROR_LOG_COUNT=$(docker compose logs --tail=100 2>&1 | grep -i error | wc -l)
if [ "$ERROR_LOG_COUNT" -gt 0 ]; then
    echo -e "${YELLOW}⚠ Se encontraron $ERROR_LOG_COUNT líneas con 'error' en los logs${NC}"
    echo "  Revisa los logs con: docker compose logs"
else
    echo -e "${GREEN}✓ No se encontraron errores recientes${NC}"
fi

# Resumen
echo ""
echo "========================================="
if [ $ERRORS -eq 0 ]; then
    echo -e "${GREEN}Sistema saludable${NC}"
    echo "========================================="
    exit 0
else
    echo -e "${RED}Se encontraron $ERRORS problema(s)${NC}"
    echo "========================================="
    echo ""
    echo "Comandos útiles para diagnóstico:"
    echo "  docker compose logs -f"
    echo "  docker compose ps"
    echo "  docker compose restart"
    exit 1
fi


