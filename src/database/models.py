"""Modelos SQLAlchemy para la base de datos."""

from datetime import datetime
from typing import AsyncGenerator

from sqlalchemy import JSON, Column, DateTime, Integer, String, Text
from sqlalchemy.ext.asyncio import AsyncSession

from src.database.connection import AsyncSessionLocal, Base


class MetricsHistory(Base):
    """Modelo para almacenar el historial de métricas del modelo.
    
    Cada vez que se llama al endpoint /metrics, se guarda automáticamente
    un registro en esta tabla con el JSON completo de métricas.
    """
    
    __tablename__ = "metrics_history"
    
    id = Column(Integer, primary_key=True, index=True)
    model_name = Column(String(255), nullable=False, index=True)
    metrics_json = Column(JSON, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    
    def __repr__(self) -> str:
        return f"<MetricsHistory(id={self.id}, model_name={self.model_name}, created_at={self.created_at})>"


async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """Dependencia para obtener una sesión de base de datos.
    
    Yields:
        Sesión de base de datos async.
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()

