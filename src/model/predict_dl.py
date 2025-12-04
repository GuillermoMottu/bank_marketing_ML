"""Predicción con modelos de Deep Learning."""

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from tensorflow import keras

from src.preprocessing.dl_preprocessing import (
    load_dl_preprocessing_pipeline,
    transform_data_dl,
)
from src.utils.config import MODELS_DIR
from src.utils.logging_config import logger


def load_dl_model(model_path: Path) -> keras.Model:
    """Carga un modelo de Deep Learning desde un archivo .h5.
    
    Args:
        model_path: Ruta al archivo del modelo (.h5).
    
    Returns:
        Modelo Keras cargado.
    """
    if not model_path.exists():
        raise FileNotFoundError(f"El modelo DL {model_path} no existe")
    
    logger.info(f"Cargando modelo DL desde {model_path}...")
    model = keras.models.load_model(model_path)
    logger.info("Modelo DL cargado correctamente")
    
    return model


def predict_dl(
    model: keras.Model,
    preprocessor,
    input_data: dict
) -> Tuple[int, float]:
    """Realiza una predicción con un modelo de Deep Learning.
    
    Args:
        model: Modelo Keras entrenado.
        preprocessor: Preprocesador ajustado (ColumnTransformer).
        input_data: Diccionario con las características del cliente.
    
    Returns:
        Tupla con (predicción, probabilidad).
        - predicción: 0 o 1 (clase predicha)
        - probabilidad: Probabilidad de clase positiva (0.0-1.0)
    """
    try:
        # Convertir a DataFrame
        df = pd.DataFrame([input_data])
        
        # Preprocesar
        X_transformed = transform_data_dl(df, preprocessor)
        
        # Predecir
        probabilities = model.predict(X_transformed.values, verbose=0)
        probability = float(probabilities[0][0])
        
        # Convertir probabilidad a predicción binaria
        prediction = 1 if probability >= 0.5 else 0
        
        return int(prediction), probability
    
    except Exception as e:
        logger.error(f"Error durante la predicción DL: {str(e)}", exc_info=True)
        raise RuntimeError(f"Error durante la predicción DL: {str(e)}")


def predict_batch_dl(
    model: keras.Model,
    preprocessor,
    X: pd.DataFrame
) -> Tuple[np.ndarray, np.ndarray]:
    """Realiza predicciones en batch con un modelo de Deep Learning.
    
    Args:
        model: Modelo Keras entrenado.
        preprocessor: Preprocesador ajustado (ColumnTransformer).
        X: DataFrame con múltiples muestras.
    
    Returns:
        Tupla con (predicciones, probabilidades).
        - predicciones: Array de 0s y 1s
        - probabilidades: Array de probabilidades (0.0-1.0)
    """
    try:
        # Preprocesar
        X_transformed = transform_data_dl(X, preprocessor)
        
        # Predecir
        probabilities = model.predict(X_transformed.values, verbose=0)
        probabilities = probabilities.flatten()
        
        # Convertir probabilidades a predicciones binarias
        predictions = (probabilities >= 0.5).astype(int)
        
        return predictions, probabilities
    
    except Exception as e:
        logger.error(f"Error durante la predicción batch DL: {str(e)}", exc_info=True)
        raise RuntimeError(f"Error durante la predicción batch DL: {str(e)}")




