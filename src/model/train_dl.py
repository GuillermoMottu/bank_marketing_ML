"""Entrenamiento de modelos de Deep Learning."""

from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from src.model.dl_architectures import build_cnn_tabular, build_dnn
from src.model.evaluator import calculate_metrics
from src.model.trainer import split_train_test
from src.preprocessing.dl_preprocessing import (
    fit_dl_preprocessing_pipeline,
    save_dl_preprocessing_pipeline,
    transform_data_dl,
)
from src.preprocessing.pipeline import load_data
from src.utils.config import (
    DL_MODEL_CNN_FILE,
    DL_MODEL_DNN_FILE,
    MODELS_DIR,
    RANDOM_STATE,
)
from src.utils.logging_config import logger


def train_dl_model(
    architecture: Literal["DNN", "CNN"],
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test: pd.DataFrame = None,
    y_test: np.ndarray = None,
    hidden_layers: List[int] = None,
    conv_filters: List[int] = None,
    dropout_rate: float = 0.3,
    use_batch_norm: bool = True,
    epochs: int = 100,
    batch_size: int = 32,
    validation_split: float = 0.2,
    early_stopping: bool = True,
    early_stopping_patience: int = 10,
    scaler_type: str = "standard",
    save_model: bool = True,
    model_name: str = None
) -> Tuple[keras.Model, Dict]:
    """Entrena un modelo de Deep Learning (DNN o CNN tabular).
    
    Args:
        architecture: Tipo de arquitectura ('DNN' o 'CNN').
        X_train: Datos de entrenamiento preprocesados.
        y_train: Etiquetas de entrenamiento.
        X_test: Datos de test preprocesados (opcional, para evaluación).
        y_test: Etiquetas de test (opcional, para evaluación).
        hidden_layers: Lista con neuronas por capa oculta para DNN. Por defecto [128, 64, 32].
        conv_filters: Lista con filtros por capa convolucional para CNN. Por defecto [64, 32].
        dropout_rate: Tasa de dropout (0.0-1.0). Por defecto 0.3.
        use_batch_norm: Si es True, usa Batch Normalization. Por defecto True.
        epochs: Número de épocas de entrenamiento. Por defecto 100.
        batch_size: Tamaño del batch. Por defecto 32.
        validation_split: Proporción de datos para validación. Por defecto 0.2.
        early_stopping: Si es True, usa Early Stopping. Por defecto True.
        early_stopping_patience: Paciencia para Early Stopping. Por defecto 10.
        scaler_type: Tipo de escalador ('standard' o 'minmax'). Por defecto 'standard'.
        save_model: Si es True, guarda el modelo. Por defecto True.
        model_name: Nombre del modelo para guardar. Si es None, se genera automáticamente.
    
    Returns:
        Tupla con (modelo_entrenado, métricas_dict).
    """
    logger.info(f"=== INICIANDO ENTRENAMIENTO DE MODELO {architecture} ===")
    
    input_dim = X_train.shape[1]
    logger.info(f"Dimensión de entrada: {input_dim}")
    
    # Construir modelo según arquitectura
    if architecture.upper() == "DNN":
        if hidden_layers is None:
            hidden_layers = [128, 64, 32]
        
        model = build_dnn(
            input_dim=input_dim,
            hidden_layers=hidden_layers,
            dropout_rate=dropout_rate,
            use_batch_norm=use_batch_norm
        )
        model_type_suffix = "dnn"
    elif architecture.upper() == "CNN":
        if conv_filters is None:
            conv_filters = [64, 32]
        
        model = build_cnn_tabular(
            input_dim=input_dim,
            conv_filters=conv_filters,
            dropout_rate=dropout_rate,
            use_batch_norm=use_batch_norm
        )
        model_type_suffix = "cnn"
    else:
        raise ValueError(f"Arquitectura no soportada: {architecture}. Use 'DNN' o 'CNN'")
    
    # Callbacks
    callbacks = []
    
    if early_stopping:
        early_stop = EarlyStopping(
            monitor='val_loss',
            patience=early_stopping_patience,
            restore_best_weights=True,
            verbose=1
        )
        callbacks.append(early_stop)
    
    # Model checkpoint
    if save_model:
        if model_name is None:
            # Usar nombres por defecto según la arquitectura
            if architecture.upper() == "DNN":
                model_name = DL_MODEL_DNN_FILE.replace('.h5', '')
            else:
                model_name = DL_MODEL_CNN_FILE.replace('.h5', '')
        
        # Asegurar que el nombre no tenga extensión .h5
        if model_name.endswith('.h5'):
            model_name = model_name[:-3]
        
        model_path = MODELS_DIR / f"{model_name}.h5"
        MODELS_DIR.mkdir(exist_ok=True)
        
        checkpoint = ModelCheckpoint(
            filepath=str(model_path),
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        )
        callbacks.append(checkpoint)
    
    # Entrenar modelo
    logger.info(f"Entrenando modelo por {epochs} épocas con batch_size={batch_size}...")
    
    history = model.fit(
        X_train.values,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        callbacks=callbacks,
        verbose=1
    )
    
    logger.info("Entrenamiento completado")
    
    # Cargar mejor modelo si se guardó
    if save_model and model_name:
        model_path = MODELS_DIR / f"{model_name}.h5"
        if model_path.exists():
            logger.info(f"Cargando mejor modelo desde {model_path}...")
            model = keras.models.load_model(model_path)
    
    # Evaluar en test set si está disponible
    metrics_dict = None
    if X_test is not None and y_test is not None:
        logger.info("Evaluando en conjunto de test...")
        
        # Predecir
        y_pred_proba = model.predict(X_test.values, verbose=0)
        y_pred_proba = y_pred_proba.flatten()
        y_pred = (y_pred_proba >= 0.5).astype(int)
        
        # Calcular métricas
        model_display_name = f"{architecture} - Bank Marketing"
        metrics_dict = calculate_metrics(
            y_true=y_test,
            y_pred=y_pred,
            y_pred_proba=y_pred_proba,
            model_name=model_display_name
        )
        
        logger.info(f"Accuracy en test: {metrics_dict['accuracy']:.4f}")
        logger.info(f"F1-score en test: {metrics_dict['f1_score']:.4f}")
        logger.info(f"AUC-ROC en test: {metrics_dict['auc_roc']:.4f}")
    
    logger.info(f"=== ENTRENAMIENTO DE {architecture} COMPLETADO ===")
    
    return model, metrics_dict


def train_dl_model_from_data(
    architecture: Literal["DNN", "CNN"],
    data_file: Path = None,
    hidden_layers: List[int] = None,
    conv_filters: List[int] = None,
    dropout_rate: float = 0.3,
    use_batch_norm: bool = True,
    epochs: int = 100,
    batch_size: int = 32,
    test_size: float = 0.2,
    validation_split: float = 0.2,
    early_stopping: bool = True,
    early_stopping_patience: int = 10,
    scaler_type: str = "standard",
    save_model: bool = True,
    save_preprocessor: bool = True,
    model_name: str = None
) -> Tuple[keras.Model, Dict, object]:
    """Entrena un modelo DL desde un archivo de datos CSV.
    
    Esta función carga los datos, crea y ajusta el preprocesador,
    divide en train/test, entrena el modelo y guarda todo.
    
    Args:
        architecture: Tipo de arquitectura ('DNN' o 'CNN').
        data_file: Ruta al archivo CSV. Si es None, usa el archivo por defecto.
        hidden_layers: Lista con neuronas por capa oculta para DNN.
        conv_filters: Lista con filtros por capa convolucional para CNN.
        dropout_rate: Tasa de dropout.
        use_batch_norm: Si es True, usa Batch Normalization.
        epochs: Número de épocas.
        batch_size: Tamaño del batch.
        test_size: Proporción de datos para test.
        validation_split: Proporción de datos para validación.
        early_stopping: Si es True, usa Early Stopping.
        early_stopping_patience: Paciencia para Early Stopping.
        scaler_type: Tipo de escalador.
        save_model: Si es True, guarda el modelo.
        save_preprocessor: Si es True, guarda el preprocesador.
        model_name: Nombre del modelo.
    
    Returns:
        Tupla con (modelo, métricas_dict, preprocesador).
    """
    logger.info("=== ENTRENAMIENTO DL DESDE DATOS ===")
    
    # 1. Cargar datos
    logger.info("Paso 1: Cargando datos...")
    X, y = load_data(data_file)
    
    # 2. Crear y ajustar pipeline de preprocesamiento DL
    logger.info("Paso 2: Creando pipeline de preprocesamiento DL...")
    preprocessor = fit_dl_preprocessing_pipeline(X, scaler_type=scaler_type)
    
    # 3. Transformar datos
    logger.info("Paso 3: Transformando datos...")
    X_transformed = transform_data_dl(X, preprocessor)
    
    # 4. Guardar pipeline de preprocesamiento
    if save_preprocessor:
        logger.info("Paso 4: Guardando pipeline de preprocesamiento DL...")
        save_dl_preprocessing_pipeline(preprocessor)
    
    # 5. Dividir en train/test
    logger.info("Paso 5: Dividiendo datos en train/test...")
    X_train, X_test, y_train, y_test = split_train_test(X_transformed, y, test_size=test_size)
    
    # 6. Entrenar modelo
    logger.info("Paso 6: Entrenando modelo DL...")
    model, metrics_dict = train_dl_model(
        architecture=architecture,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        hidden_layers=hidden_layers,
        conv_filters=conv_filters,
        dropout_rate=dropout_rate,
        use_batch_norm=use_batch_norm,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        early_stopping=early_stopping,
        early_stopping_patience=early_stopping_patience,
        scaler_type=scaler_type,
        save_model=save_model,
        model_name=model_name
    )
    
    logger.info("=== ENTRENAMIENTO DL COMPLETADO ===")
    
    return model, metrics_dict, preprocessor

