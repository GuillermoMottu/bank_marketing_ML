"""Operaciones CRUD para la base de datos."""

from datetime import datetime
from typing import Dict, List

from sqlalchemy import desc, select
from sqlalchemy.ext.asyncio import AsyncSession

from src.database.models import MetricsHistory
from src.utils.logging_config import logger


async def create_metrics_record(
    db: AsyncSession,
    metrics_dict: Dict
) -> MetricsHistory:
    """Crea un nuevo registro en la tabla metrics_history.
    
    Args:
        db: Sesión de base de datos.
        metrics_dict: Diccionario con las métricas a guardar.
    
    Returns:
        Registro creado.
    """
    logger.info("Creando registro de métricas en base de datos...")
    
    # Extraer el nombre del modelo
    model_name = metrics_dict.get("model", "Decision Tree - Bank Marketing")
    
    # Crear registro
    metrics_record = MetricsHistory(
        model_name=model_name,
        metrics_json=metrics_dict,
        created_at=datetime.now()
    )
    
    db.add(metrics_record)
    await db.commit()
    await db.refresh(metrics_record)
    
    logger.info(f"Registro de métricas creado con ID: {metrics_record.id}")
    
    return metrics_record


async def get_all_metrics(
    db: AsyncSession,
    limit: int = 100
) -> List[MetricsHistory]:
    """Obtiene todos los registros de métricas ordenados por fecha descendente.
    
    Args:
        db: Sesión de base de datos.
        limit: Número máximo de registros a retornar.
    
    Returns:
        Lista de registros de métricas.
    """
    logger.info(f"Obteniendo últimos {limit} registros de métricas...")
    
    query = select(MetricsHistory).order_by(desc(MetricsHistory.created_at)).limit(limit)
    result = await db.execute(query)
    records = result.scalars().all()
    
    logger.info(f"Se encontraron {len(records)} registros")
    
    return list(records)


async def get_latest_metrics(db: AsyncSession) -> MetricsHistory | None:
    """Obtiene el último registro de métricas.
    
    Args:
        db: Sesión de base de datos.
    
    Returns:
        Último registro de métricas o None si no hay registros.
    """
    logger.info("Obteniendo último registro de métricas...")
    
    query = select(MetricsHistory).order_by(desc(MetricsHistory.created_at)).limit(1)
    result = await db.execute(query)
    record = result.scalar_one_or_none()
    
    return record

