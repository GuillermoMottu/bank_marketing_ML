"""Rutas de la API FastAPI."""

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.dependencies import ModelService, get_model_service
from src.api.models import MetricsOutput, PredictionInput, PredictionOutput
from src.database.crud import create_metrics_record
from src.database.connection import get_session
from src.model.evaluator import calculate_metrics
from src.preprocessing.pipeline import load_data, load_preprocessing_pipeline, transform_data
from src.utils.config import MODELS_DIR, TARGET_COLUMN
from src.utils.logging_config import logger

router = APIRouter()


@router.post("/predict", response_model=PredictionOutput)
async def predict(
    input_data: PredictionInput,
    model_service: ModelService = Depends(get_model_service)
) -> PredictionOutput:
    """Realiza una predicción de si un cliente suscribirá un depósito.
    
    Args:
        input_data: Datos del cliente en formato PredictionInput.
        model_service: Servicio del modelo inyectado.
    
    Returns:
        Predicción con probabilidad y nombre de clase.
    """
    try:
        logger.info(f"Recibida solicitud de predicción para cliente: {input_data.age} años")
        
        # Convertir Pydantic model a dict
        input_dict = input_data.model_dump()
        
        # Hacer predicción
        prediction, probability = model_service.predict(input_dict)
        
        class_name = "yes" if prediction == 1 else "no"
        
        logger.info(f"Predicción: {class_name} (probabilidad: {probability:.4f})")
        
        return PredictionOutput(
            prediction=prediction,
            probability=probability,
            class_name=class_name
        )
    
    except Exception as e:
        logger.error(f"Error en predicción: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error en la predicción: {str(e)}")


@router.get("/metrics", response_model=MetricsOutput)
async def get_metrics(
    model_service: ModelService = Depends(get_model_service),
    db: AsyncSession = Depends(get_session)
) -> MetricsOutput:
    """Obtiene las métricas actuales del modelo calculadas en el conjunto de test.
    
    Las métricas se calculan en tiempo real y se guardan automáticamente
    en la base de datos.
    
    Args:
        model_service: Servicio del modelo inyectado.
        db: Sesión de base de datos inyectada.
    
    Returns:
        Métricas del modelo en formato MetricsOutput.
    """
    try:
        logger.info("Calculando métricas del modelo...")
        
        if model_service.model is None or model_service.preprocessor is None:
            raise HTTPException(status_code=500, detail="Modelo o preprocesador no cargados")
        
        # Cargar datos de test (necesitamos guardar X_test e y_test)
        # Por ahora, cargamos todo el dataset y recreamos la división
        from src.model.trainer import split_train_test
        
        X, y = load_data()
        X_train, X_test, y_train, y_test = split_train_test(X, y)
        
        # Preprocesar datos de test
        X_test_transformed = transform_data(X_test, model_service.preprocessor)
        
        # Predecir en test set
        y_pred = model_service.model.predict(X_test_transformed)
        y_pred_proba = model_service.model.predict_proba(X_test_transformed)[:, 1]
        
        # Calcular métricas
        metrics_dict = calculate_metrics(y_test, y_pred, y_pred_proba)
        
        # Guardar en base de datos
        try:
            await create_metrics_record(db, metrics_dict)
            logger.info("Métricas guardadas en base de datos")
        except Exception as e:
            logger.warning(f"No se pudo guardar en BD: {e}")
        
        # Convertir a modelo Pydantic
        metrics_output = MetricsOutput(**metrics_dict)
        
        return metrics_output
    
    except Exception as e:
        logger.error(f"Error calculando métricas: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error calculando métricas: {str(e)}")


@router.get("/health")
async def health_check() -> dict:
    """Health check endpoint.
    
    Returns:
        Estado de salud de la API.
    """
    return {
        "status": "healthy",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

