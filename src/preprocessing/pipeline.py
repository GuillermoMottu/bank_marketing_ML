"""Pipeline completo de preprocesamiento para el dataset Bank Marketing."""

import joblib
import pandas as pd
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler

from src.utils.config import (
    DATA_RAW_DIR,
    MODELS_DIR,
    PREPROCESSING_PIPELINE_FILE,
    RAW_DATA_FILE,
    TARGET_COLUMN,
)
from src.utils.logging_config import logger
from src.preprocessing.transformations import UnknownValueHandler


def create_preprocessing_pipeline() -> Pipeline:
    """Crea el pipeline completo de preprocesamiento.
    
    El pipeline incluye:
    - Manejo de valores 'unknown' como categoría propia
    - One-Hot Encoding para variables categóricas
    - RobustScaler para variables numéricas
    
    Returns:
        Pipeline completo de preprocesamiento con ColumnTransformer.
    """
    logger.info("Creando pipeline de preprocesamiento...")
    
    # Definir columnas categóricas (sin incluir el target)
    categorical_features = [
        'job', 'marital', 'education', 'default', 
        'housing', 'loan', 'contact', 'month', 'poutcome'
    ]
    
    # Definir columnas numéricas
    numerical_features = [
        'age', 'balance', 'day', 'duration', 
        'campaign', 'pdays', 'previous'
    ]
    
    # Pipeline para variables categóricas
    categorical_pipeline = Pipeline([
        ('unknown_handler', UnknownValueHandler(unknown_values=['unknown'])),
        ('onehot', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')),
    ])
    
    # Pipeline para variables numéricas
    numerical_pipeline = Pipeline([
        ('scaler', RobustScaler()),
    ])
    
    # ColumnTransformer que aplica transformaciones según el tipo de columna
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_pipeline, numerical_features),
            ('cat', categorical_pipeline, categorical_features),
        ],
        remainder='drop'  # Eliminar cualquier columna no especificada
    )
    
    logger.info(f"Pipeline creado con {len(numerical_features)} numéricas y "
                f"{len(categorical_features)} categóricas")
    
    return preprocessor


def load_data(file_path: Path = None) -> tuple[pd.DataFrame, pd.Series]:
    """Carga el dataset desde el archivo CSV.
    
    Args:
        file_path: Ruta al archivo CSV. Si es None, usa la configuración por defecto.
    
    Returns:
        Tupla con (X, y) donde X es el DataFrame de características e y es la serie objetivo.
    """
    if file_path is None:
        file_path = DATA_RAW_DIR / RAW_DATA_FILE
    
    logger.info(f"Cargando datos desde {file_path}...")
    
    if not file_path.exists():
        raise FileNotFoundError(f"El archivo {file_path} no existe")
    
    df = pd.read_csv(file_path)
    logger.info(f"Datos cargados: {df.shape[0]} filas, {df.shape[1]} columnas")
    
    # Separar características y variable objetivo
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"La columna '{TARGET_COLUMN}' no existe en el dataset")
    
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN].map({'yes': 1, 'no': 0})
    
    logger.info(f"Variable objetivo: {y.value_counts().to_dict()}")
    
    return X, y


def fit_preprocessing_pipeline(
    X: pd.DataFrame,
    preprocessor: ColumnTransformer = None
) -> ColumnTransformer:
    """Ajusta el pipeline de preprocesamiento con los datos.
    
    Args:
        X: DataFrame con las características.
        preprocessor: Pipeline opcional. Si es None, se crea uno nuevo.
    
    Returns:
        Pipeline ajustado.
    """
    if preprocessor is None:
        preprocessor = create_preprocessing_pipeline()
    
    logger.info("Ajustando pipeline de preprocesamiento...")
    preprocessor.fit(X)
    logger.info("Pipeline ajustado correctamente")
    
    return preprocessor


def transform_data(
    X: pd.DataFrame,
    preprocessor: ColumnTransformer
) -> pd.DataFrame:
    """Transforma los datos usando el pipeline ajustado.
    
    Args:
        X: DataFrame con las características.
        preprocessor: Pipeline ajustado.
    
    Returns:
        DataFrame transformado (numpy array convertido a DataFrame).
    """
    logger.info(f"Transformando {len(X)} muestras...")
    X_transformed = preprocessor.transform(X)
    
    # Convertir a DataFrame para mejor manejo
    # Obtener nombres de columnas del preprocessor
    feature_names = preprocessor.get_feature_names_out()
    X_transformed_df = pd.DataFrame(
        X_transformed,
        columns=feature_names,
        index=X.index
    )
    
    logger.info(f"Datos transformados: {X_transformed_df.shape}")
    return X_transformed_df


def save_preprocessing_pipeline(
    preprocessor: ColumnTransformer,
    file_path: Path = None
) -> None:
    """Guarda el pipeline de preprocesamiento en un archivo.
    
    Args:
        preprocessor: Pipeline a guardar.
        file_path: Ruta donde guardar el pipeline. Si es None, usa la configuración por defecto.
    """
    if file_path is None:
        MODELS_DIR.mkdir(exist_ok=True)
        file_path = MODELS_DIR / PREPROCESSING_PIPELINE_FILE
    
    logger.info(f"Guardando pipeline en {file_path}...")
    joblib.dump(preprocessor, file_path)
    logger.info("Pipeline guardado correctamente")


def load_preprocessing_pipeline(file_path: Path = None) -> ColumnTransformer:
    """Carga el pipeline de preprocesamiento desde un archivo.
    
    Args:
        file_path: Ruta al archivo del pipeline. Si es None, usa la configuración por defecto.
    
    Returns:
        Pipeline cargado.
    """
    if file_path is None:
        file_path = MODELS_DIR / PREPROCESSING_PIPELINE_FILE
    
    if not file_path.exists():
        raise FileNotFoundError(f"El pipeline {file_path} no existe")
    
    logger.info(f"Cargando pipeline desde {file_path}...")
    preprocessor = joblib.load(file_path)
    logger.info("Pipeline cargado correctamente")
    
    return preprocessor

