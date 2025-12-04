"""Rutas de la API FastAPI."""

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.dependencies import ModelService, get_model_service
from src.api.models import (
    MetricsOutput,
    PredictionInput,
    PredictionOutput,
    TrainingInput,
    TrainingOutput,
)
from src.database.crud import create_metrics_record, create_prediction_record
from src.database.connection import get_session
from src.model.evaluator import calculate_metrics
from src.model.predict_dl import predict_batch_dl
from src.model.train_dl import train_dl_model_from_data
from src.preprocessing.pipeline import load_data, load_preprocessing_pipeline, transform_data
from src.utils.config import (
    DL_MODEL_CNN_FILE,
    DL_MODEL_DNN_FILE,
    MODELS_DIR,
    TARGET_COLUMN,
)
from src.utils.logging_config import logger

router = APIRouter()


@router.post("/predict", response_model=PredictionOutput)
async def predict(
    input_data: PredictionInput,
    model_service: ModelService = Depends(get_model_service),
    db: AsyncSession = Depends(get_session)
) -> PredictionOutput:
    """Realiza una predicción de si un cliente suscribirá un depósito.
    
    Args:
        input_data: Datos del cliente en formato PredictionInput.
        model_service: Servicio del modelo inyectado.
    
    Returns:
        Predicción con probabilidad y nombre de clase.
    """
    try:
        model_type = input_data.model_type
        architecture = input_data.architecture
        
        logger.info(
            f"Recibida solicitud de predicción para cliente: {input_data.age} años, "
            f"tipo: {model_type}, arquitectura: {architecture}"
        )
        
        # Validar que si es Deep Learning, se especifique la arquitectura
        if model_type == "Deep" and architecture is None:
            raise HTTPException(
                status_code=400,
                detail="La arquitectura es requerida cuando model_type='Deep'. Use 'DNN' o 'CNN'."
            )
        
        # Convertir Pydantic model a dict (sin model_type y architecture para la predicción)
        input_dict = input_data.model_dump(exclude={"model_type", "architecture"})
        
        # Hacer predicción según el tipo de modelo
        try:
            prediction, probability = model_service.predict(
                input_dict,
                model_type=model_type,
                architecture=architecture
            )
        except RuntimeError as e:
            # Error relacionado con el modelo no disponible
            logger.error(f"Error en predicción (modelo no disponible): {str(e)}", exc_info=True)
            raise HTTPException(status_code=503, detail=str(e))
        
        class_name = "yes" if prediction == 1 else "no"
        
        logger.info(f"Predicción: {class_name} (probabilidad: {probability:.4f})")
        
        # Guardar predicción en base de datos
        try:
            await create_prediction_record(
                db=db,
                input_data=input_dict,
                prediction=prediction,
                probability=probability,
                class_name=class_name
            )
            logger.info("Predicción guardada en base de datos")
        except Exception as e:
            logger.warning(f"No se pudo guardar la predicción en BD: {e}")
            # Continuar aunque falle el guardado
        
        return PredictionOutput(
            prediction=prediction,
            probability=probability,
            class_name=class_name
        )
    
    except HTTPException:
        # Re-lanzar HTTPException sin modificar
        raise
    except Exception as e:
        logger.error(f"Error en predicción: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error en la predicción: {str(e)}")


@router.get("/metrics", response_model=MetricsOutput)
async def get_metrics(
    model_type: str = "ML",
    architecture: str = None,
    model_service: ModelService = Depends(get_model_service),
    db: AsyncSession = Depends(get_session)
) -> MetricsOutput:
    """Obtiene las métricas actuales del modelo calculadas en el conjunto de test.
    
    Las métricas se calculan en tiempo real y se guardan automáticamente
    en la base de datos.
    
    Args:
        model_type: Tipo de modelo ('ML' o 'Deep'). Por defecto 'ML'.
        architecture: Arquitectura para Deep Learning ('DNN' o 'CNN'). Requerido si model_type='Deep'.
        model_service: Servicio del modelo inyectado.
        db: Sesión de base de datos inyectada.
    
    Returns:
        Métricas del modelo en formato MetricsOutput.
    """
    try:
        logger.info(f"Calculando métricas del modelo tipo: {model_type}, arquitectura: {architecture}")
        
        # Validar que si es Deep Learning, se especifique la arquitectura
        if model_type == "Deep" and architecture is None:
            raise HTTPException(
                status_code=400,
                detail="La arquitectura es requerida cuando model_type='Deep'. Use 'DNN' o 'CNN'."
            )
        
        # Cargar datos de test
        from src.model.trainer import split_train_test
        
        X, y = load_data()
        X_train, X_test, y_train, y_test = split_train_test(X, y)
        
        if model_type == "Deep":
            # Métricas para modelo DL
            if architecture.upper() == "DNN":
                model_path = MODELS_DIR / DL_MODEL_DNN_FILE
                model_display_name = "DNN - Bank Marketing"
            elif architecture.upper() == "CNN":
                model_path = MODELS_DIR / DL_MODEL_CNN_FILE
                model_display_name = "CNN - Bank Marketing"
            else:
                raise HTTPException(status_code=400, detail=f"Arquitectura no soportada: {architecture}")
            
            if not model_path.exists():
                raise HTTPException(
                    status_code=503,
                    detail=f"El modelo {architecture} no está disponible. Por favor, entrénalo primero."
                )
            
            # Cargar modelo y preprocesador DL
            from src.model.predict_dl import load_dl_model
            from src.preprocessing.dl_preprocessing import load_dl_preprocessing_pipeline
            
            model = load_dl_model(model_path)
            preprocessor = load_dl_preprocessing_pipeline()
            
            # Predecir (predict_batch_dl se encarga del preprocesamiento internamente)
            y_pred, y_pred_proba = predict_batch_dl(model, preprocessor, X_test)
        
        else:
            # Métricas para modelo ML
            if not model_service.is_loaded:
                error_detail = (
                    "El modelo ML no está disponible. "
                    f"{model_service.load_error if model_service.load_error else 'El modelo no ha sido cargado correctamente.'} "
                    "Por favor, entrena el modelo ejecutando: python train_model.py"
                )
                logger.error(error_detail)
                raise HTTPException(status_code=503, detail=error_detail)
            
            if model_service.model is None or model_service.preprocessor is None:
                error_detail = (
                    "El modelo o preprocesador ML no están disponibles. "
                    "Por favor, entrena el modelo ejecutando: python train_model.py"
                )
                raise HTTPException(status_code=503, detail=error_detail)
            
            # Preprocesar datos de test
            X_test_transformed = transform_data(X_test, model_service.preprocessor)
            
            # Predecir en test set
            y_pred = model_service.model.predict(X_test_transformed)
            y_pred_proba = model_service.model.predict_proba(X_test_transformed)[:, 1]
            model_display_name = "Decision Tree - Bank Marketing"
        
        # Calcular métricas
        metrics_dict = calculate_metrics(y_test, y_pred, y_pred_proba, model_name=model_display_name)
        
        # Guardar en base de datos
        try:
            await create_metrics_record(db, metrics_dict)
            logger.info("Métricas guardadas en base de datos")
        except Exception as e:
            logger.warning(f"No se pudo guardar en BD: {e}")
        
        # Convertir a modelo Pydantic
        metrics_output = MetricsOutput(**metrics_dict)
        
        return metrics_output
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error calculando métricas: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error calculando métricas: {str(e)}")


@router.post("/train", response_model=TrainingOutput)
async def train_model(
    training_input: TrainingInput,
    model_service: ModelService = Depends(get_model_service),
    db: AsyncSession = Depends(get_session)
) -> TrainingOutput:
    """Entrena un modelo de Deep Learning.
    
    Args:
        training_input: Parámetros de entrenamiento en formato TrainingInput.
        model_service: Servicio del modelo inyectado.
        db: Sesión de base de datos inyectada.
    
    Returns:
        Resultado del entrenamiento en formato TrainingOutput.
    """
    try:
        logger.info(f"Iniciando entrenamiento de modelo {training_input.architecture}...")
        
        # Determinar nombre del modelo
        model_name = training_input.model_name
        if model_name is None:
            if training_input.architecture.upper() == "DNN":
                model_name = "dnn_model"
            else:
                model_name = "cnn_model"
        
        # Entrenar modelo
        model, metrics_dict, preprocessor = train_dl_model_from_data(
            architecture=training_input.architecture,
            hidden_layers=training_input.hidden_layers,
            conv_filters=training_input.conv_filters,
            dropout_rate=training_input.dropout_rate,
            use_batch_norm=training_input.use_batch_norm,
            epochs=training_input.epochs,
            batch_size=training_input.batch_size,
            validation_split=training_input.validation_split,
            early_stopping=training_input.early_stopping,
            early_stopping_patience=training_input.early_stopping_patience,
            scaler_type=training_input.scaler_type,
            save_model=True,
            save_preprocessor=True,
            model_name=model_name
        )
        
        # Guardar métricas en base de datos si están disponibles
        if metrics_dict:
            try:
                await create_metrics_record(db, metrics_dict)
                logger.info("Métricas guardadas en base de datos")
            except Exception as e:
                logger.warning(f"No se pudo guardar métricas en BD: {e}")
        
        # Recargar modelos en el servicio
        model_service.load_dl_model(training_input.architecture)
        
        logger.info(f"Entrenamiento completado exitosamente: {model_name}")
        
        return TrainingOutput(
            success=True,
            message=f"Modelo {training_input.architecture} entrenado exitosamente",
            model_name=model_name,
            metrics=metrics_dict
        )
    
    except Exception as e:
        logger.error(f"Error durante el entrenamiento: {str(e)}", exc_info=True)
        return TrainingOutput(
            success=False,
            message=f"Error durante el entrenamiento: {str(e)}",
            model_name=None,
            metrics=None
        )


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

