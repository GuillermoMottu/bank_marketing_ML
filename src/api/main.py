"""Aplicación principal de FastAPI."""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routes import router
from src.database.connection import init_db
from src.utils.logging_config import logger


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Gestiona el ciclo de vida de la aplicación.
    
    Inicializa la base de datos al iniciar y realiza limpieza al cerrar.
    """
    # Startup
    logger.info("Iniciando aplicación FastAPI...")
    try:
        await init_db()
        logger.info("Base de datos inicializada correctamente")
    except Exception as e:
        logger.error(f"Error al inicializar base de datos: {e}")
        # Continuar aunque falle la BD (para desarrollo)
    
    yield
    
    # Shutdown
    logger.info("Cerrando aplicación FastAPI...")


# Crear aplicación FastAPI
app = FastAPI(
    title="Bank Marketing Decision Tree API",
    description="API para predicción de campañas bancarias usando Decision Tree",
    version="1.0.0",
    lifespan=lifespan,
)

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En producción, especificar dominios permitidos
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Incluir rutas
app.include_router(router, prefix="/api/v1", tags=["predictions"])

# Ruta raíz
@app.get("/")
async def root():
    """Ruta raíz de la API."""
    return {
        "message": "Bank Marketing Decision Tree API",
        "version": "1.0.0",
        "docs": "/docs"
    }

