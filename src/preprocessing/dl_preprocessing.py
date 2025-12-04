"""Preprocesador específico para modelos de Deep Learning."""

import joblib
from pathlib import Path
from typing import Optional

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler

from src.preprocessing.transformations import UnknownValueHandler
from src.utils.config import MODELS_DIR
from src.utils.logging_config import logger


def create_dl_preprocessing_pipeline(
    scaler_type: str = "standard"
) -> ColumnTransformer:
    """Crea el pipeline de preprocesamiento para Deep Learning.
    
    Similar al pipeline de ML pero con normalización adicional específica
    para redes neuronales (StandardScaler o MinMaxScaler).
    
    Args:
        scaler_type: Tipo de escalador. 'standard' (StandardScaler) o 'minmax' (MinMaxScaler).
                    Por defecto 'standard'.
    
    Returns:
        ColumnTransformer con pipeline de preprocesamiento.
    """
    logger.info(f"Creando pipeline de preprocesamiento DL con scaler: {scaler_type}")
    
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
    
    # Seleccionar escalador
    if scaler_type.lower() == "minmax":
        scaler = MinMaxScaler()
    else:
        scaler = StandardScaler()
    
    # Pipeline para variables categóricas
    categorical_pipeline = Pipeline([
        ('unknown_handler', UnknownValueHandler(unknown_values=['unknown'])),
        ('onehot', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')),
    ])
    
    # Pipeline para variables numéricas con normalización adicional
    numerical_pipeline = Pipeline([
        ('scaler', scaler),
    ])
    
    # ColumnTransformer que aplica transformaciones según el tipo de columna
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_pipeline, numerical_features),
            ('cat', categorical_pipeline, categorical_features),
        ],
        remainder='drop'  # Eliminar cualquier columna no especificada
    )
    
    logger.info(f"Pipeline DL creado con {len(numerical_features)} numéricas y "
                f"{len(categorical_features)} categóricas, scaler: {scaler_type}")
    
    return preprocessor


def fit_dl_preprocessing_pipeline(
    X: pd.DataFrame,
    preprocessor: Optional[ColumnTransformer] = None,
    scaler_type: str = "standard"
) -> ColumnTransformer:
    """Ajusta el pipeline de preprocesamiento DL con los datos.
    
    Args:
        X: DataFrame con las características.
        preprocessor: Pipeline opcional. Si es None, se crea uno nuevo.
        scaler_type: Tipo de escalador ('standard' o 'minmax').
    
    Returns:
        Pipeline ajustado.
    """
    if preprocessor is None:
        preprocessor = create_dl_preprocessing_pipeline(scaler_type=scaler_type)
    
    logger.info("Ajustando pipeline de preprocesamiento DL...")
    preprocessor.fit(X)
    logger.info("Pipeline DL ajustado correctamente")
    
    return preprocessor


def transform_data_dl(
    X: pd.DataFrame,
    preprocessor: ColumnTransformer
) -> pd.DataFrame:
    """Transforma los datos usando el pipeline DL ajustado.
    
    Args:
        X: DataFrame con las características.
        preprocessor: Pipeline ajustado.
    
    Returns:
        DataFrame transformado.
    """
    logger.info(f"Transformando {len(X)} muestras con pipeline DL...")
    X_transformed = preprocessor.transform(X)
    
    # Convertir a DataFrame para mejor manejo
    feature_names = preprocessor.get_feature_names_out()
    X_transformed_df = pd.DataFrame(
        X_transformed,
        columns=feature_names,
        index=X.index
    )
    
    logger.info(f"Datos transformados: {X_transformed_df.shape}")
    return X_transformed_df


def save_dl_preprocessing_pipeline(
    preprocessor: ColumnTransformer,
    file_path: Optional[Path] = None
) -> None:
    """Guarda el pipeline de preprocesamiento DL en un archivo.
    
    Args:
        preprocessor: Pipeline a guardar.
        file_path: Ruta donde guardar el pipeline. Si es None, usa la configuración por defecto.
    """
    if file_path is None:
        MODELS_DIR.mkdir(exist_ok=True)
        file_path = MODELS_DIR / "dl_preprocessing_pipeline.pkl"
    
    logger.info(f"Guardando pipeline DL en {file_path}...")
    joblib.dump(preprocessor, file_path)
    logger.info("Pipeline DL guardado correctamente")


def load_dl_preprocessing_pipeline(
    file_path: Optional[Path] = None
) -> ColumnTransformer:
    """Carga el pipeline de preprocesamiento DL desde un archivo.
    
    Args:
        file_path: Ruta al archivo del pipeline. Si es None, usa la configuración por defecto.
    
    Returns:
        Pipeline cargado.
    """
    if file_path is None:
        file_path = MODELS_DIR / "dl_preprocessing_pipeline.pkl"
    
    if not file_path.exists():
        raise FileNotFoundError(f"El pipeline DL {file_path} no existe")
    
    logger.info(f"Cargando pipeline DL desde {file_path}...")
    preprocessor = joblib.load(file_path)
    logger.info("Pipeline DL cargado correctamente")
    
    return preprocessor


