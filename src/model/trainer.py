"""Entrenamiento del modelo Decision Tree con GridSearchCV."""

import joblib
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit
from sklearn.tree import DecisionTreeClassifier

from src.utils.config import (
    CV_FOLDS,
    MODELS_DIR,
    MODEL_FILE,
    RANDOM_STATE,
    TEST_SIZE,
)
from src.utils.logging_config import logger
from src.model.evaluator import calculate_metrics


def train_decision_tree(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test: pd.DataFrame = None,
    y_test: np.ndarray = None,
    save_model: bool = True,
    model_path: Path = None
) -> Tuple[DecisionTreeClassifier, dict]:
    """Entrena un modelo DecisionTreeClassifier con optimización de hiperparámetros.
    
    Utiliza GridSearchCV con validación cruzada para encontrar los mejores
    hiperparámetros según el scoring 'roc_auc'.
    
    Args:
        X_train: Datos de entrenamiento preprocesados.
        y_train: Etiquetas de entrenamiento.
        X_test: Datos de test preprocesados (opcional, para evaluación).
        y_test: Etiquetas de test (opcional, para evaluación).
        save_model: Si es True, guarda el modelo entrenado.
        model_path: Ruta donde guardar el modelo. Si es None, usa la configuración por defecto.
    
    Returns:
        Tupla con (mejor_modelo, mejores_parámetros).
    """
    logger.info("Iniciando entrenamiento del DecisionTreeClassifier...")
    
    # Definir el espacio de búsqueda de hiperparámetros
    param_grid = {
        'max_depth': [5, 10, 15, 20, None],
        'min_samples_split': [2, 5, 10, 20],
        'min_samples_leaf': [1, 2, 4, 8],
        'criterion': ['gini', 'entropy'],
        'class_weight': [None, 'balanced'],
    }
    
    logger.info(f"GridSearchCV con {CV_FOLDS} folds y parámetros: {param_grid}")
    
    # Crear el modelo base
    base_model = DecisionTreeClassifier(random_state=RANDOM_STATE)
    
    # GridSearchCV con validación cruzada estratificada
    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        cv=CV_FOLDS,
        scoring='roc_auc',  # Usar ROC-AUC para datasets potencialmente desbalanceados
        n_jobs=-1,
        verbose=1,
        return_train_score=True,
    )
    
    # Entrenar
    logger.info("Ejecutando GridSearchCV...")
    grid_search.fit(X_train, y_train)
    
    # Obtener el mejor modelo
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_
    
    logger.info(f"Mejor score de validación cruzada: {best_score:.4f}")
    logger.info(f"Mejores parámetros: {best_params}")
    
    # Evaluar en test set si está disponible
    if X_test is not None and y_test is not None:
        logger.info("Evaluando en conjunto de test...")
        y_pred = best_model.predict(X_test)
        y_pred_proba = best_model.predict_proba(X_test)[:, 1]
        
        metrics = calculate_metrics(y_test, y_pred, y_pred_proba)
        logger.info(f"Accuracy en test: {metrics['accuracy']:.4f}")
        logger.info(f"F1-score en test: {metrics['f1_score']:.4f}")
    
    # Guardar el modelo si está especificado
    if save_model:
        if model_path is None:
            MODELS_DIR.mkdir(exist_ok=True)
            model_path = MODELS_DIR / MODEL_FILE
        
        logger.info(f"Guardando modelo en {model_path}...")
        joblib.dump(best_model, model_path)
        logger.info("Modelo guardado correctamente")
    
    return best_model, best_params


def split_train_test(
    X: pd.DataFrame,
    y: np.ndarray,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE
) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    """Divide los datos en conjuntos de entrenamiento y test de forma estratificada.
    
    Args:
        X: Datos de características.
        y: Variable objetivo.
        test_size: Proporción de datos para test.
        random_state: Semilla para reproducibilidad.
    
    Returns:
        Tupla con (X_train, X_test, y_train, y_test).
    """
    logger.info(f"Dividiendo datos en train/test ({1-test_size:.0%}/{test_size:.0%})...")
    
    splitter = StratifiedShuffleSplit(
        n_splits=1,
        test_size=test_size,
        random_state=random_state
    )
    
    train_idx, test_idx = next(splitter.split(X, y))
    
    X_train = X.iloc[train_idx]
    X_test = X.iloc[test_idx]
    y_train = y[train_idx]
    y_test = y[test_idx]
    
    logger.info(f"Train: {len(X_train)} muestras")
    logger.info(f"Test: {len(X_test)} muestras")
    logger.info(f"Distribución en train: {np.bincount(y_train)}")
    logger.info(f"Distribución en test: {np.bincount(y_test)}")
    
    return X_train, X_test, y_train, y_test


def load_model(model_path: Path = None) -> DecisionTreeClassifier:
    """Carga un modelo entrenado desde un archivo.
    
    Args:
        model_path: Ruta al archivo del modelo. Si es None, usa la configuración por defecto.
    
    Returns:
        Modelo cargado.
    """
    if model_path is None:
        model_path = MODELS_DIR / MODEL_FILE
    
    if not model_path.exists():
        raise FileNotFoundError(f"El modelo {model_path} no existe")
    
    logger.info(f"Cargando modelo desde {model_path}...")
    model = joblib.load(model_path)
    logger.info("Modelo cargado correctamente")
    
    return model

