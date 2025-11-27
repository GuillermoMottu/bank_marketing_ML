"""Configuración de conexión a PostgreSQL con SQLAlchemy async."""

from collections.abc import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import declarative_base

from src.utils.config import DATABASE_URL
from src.utils.logging_config import logger

# Base para los modelos
Base = declarative_base()

# Motor de base de datos async
# Configurar pool de conexiones para manejar múltiples operaciones concurrentes
engine = create_async_engine(
    DATABASE_URL,
    echo=False,  # Cambiar a True para ver las queries SQL
    future=True,
    pool_size=10,  # Tamaño del pool de conexiones
    max_overflow=20,  # Conexiones adicionales permitidas
    pool_pre_ping=True,  # Verificar que las conexiones estén vivas antes de usarlas
    pool_recycle=3600,  # Reciclar conexiones después de 1 hora
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
    import os
    from src.utils.config import DB_HOST, DB_PORT
    
    logger.info("Inicializando base de datos...")
    
    # Mostrar información de configuración (sin credenciales)
    if os.getenv("DATABASE_URL"):
        logger.info("Usando DATABASE_URL desde variable de entorno")
        # Mostrar solo el hostname (parte después de @)
        if "@" in DATABASE_URL:
            host_part = DATABASE_URL.split("@")[1].split("/")[0]
            logger.info(f"Host de base de datos: {host_part}")
    else:
        logger.info(f"Construyendo DATABASE_URL desde variables individuales")
        logger.info(f"Host: {DB_HOST}, Puerto: {DB_PORT}")
    
    try:
        async with engine.begin() as conn:
            # Importar modelos aquí para evitar imports circulares
            from src.database.models import MetricsHistory, PredictionHistory
            
            # Crear todas las tablas
            await conn.run_sync(Base.metadata.create_all)
        
        logger.info("Base de datos inicializada correctamente")
    except Exception as e:
        logger.error(f"Error al inicializar base de datos: {e}")
        logger.error(f"Tipo de error: {type(e).__name__}")
        
        # Mostrar información útil para debugging
        if "@" in DATABASE_URL:
            host_part = DATABASE_URL.split("@")[1].split("/")[0] if "@" in DATABASE_URL else "N/A"
            logger.error(f"Intentando conectar a: {host_part}")
        else:
            logger.error(f"Host configurado: {DB_HOST}, Puerto: {DB_PORT}")
        
        logger.error("Verifica que:")
        logger.error("  1. El servicio PostgreSQL está activo en Railway")
        logger.error("  2. Las variables de entorno están configuradas correctamente")
        logger.error("  3. DATABASE_URL o variables individuales (DB_HOST, etc.) están definidas")
        
        # Re-lanzar la excepción para que el caller pueda manejarla
        raise


async def close_db() -> None:
    """Cierra las conexiones de la base de datos."""
    logger.info("Cerrando conexiones de base de datos...")
    await engine.dispose()
    logger.info("Conexiones cerradas")


async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """Dependency para obtener una sesión de base de datos.
    
    Esta función es utilizada por FastAPI para inyectar sesiones de base de datos
    en los endpoints.
    
    Yields:
        AsyncSession: Sesión de base de datos asíncrona.
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()
