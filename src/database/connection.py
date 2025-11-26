"""Configuración de conexión a PostgreSQL con SQLAlchemy async."""

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import declarative_base

from src.utils.config import DATABASE_URL
from src.utils.logging_config import logger

# Base para los modelos
Base = declarative_base()

# Motor de base de datos async
engine = create_async_engine(
    DATABASE_URL,
    echo=False,  # Cambiar a True para ver las queries SQL
    future=True,
)

# Session factory
AsyncSessionLocal = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


async def init_db() -> None:
    """Inicializa la base de datos creando todas las tablas.
    
    Esta función debe ser llamada al iniciar la aplicación.
    """
    logger.info("Inicializando base de datos...")
    
    async with engine.begin() as conn:
        # Importar modelos aquí para evitar imports circulares
        from src.database.models import MetricsHistory
        
        # Crear todas las tablas
        await conn.run_sync(Base.metadata.create_all)
    
    logger.info("Base de datos inicializada correctamente")


async def close_db() -> None:
    """Cierra las conexiones de la base de datos."""
    logger.info("Cerrando conexiones de base de datos...")
    await engine.dispose()
    logger.info("Conexiones cerradas")

