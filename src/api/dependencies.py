"""Dependencias inyectables para FastAPI."""

from pathlib import Path

import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.compose import ColumnTransformer

from src.model.trainer import load_model
from src.preprocessing.pipeline import load_preprocessing_pipeline
from src.utils.config import MODELS_DIR, PREPROCESSING_PIPELINE_FILE
from src.utils.logging_config import logger


class ModelService:
    """Servicio para cargar y gestionar el modelo y preprocesador."""
    
    def __init__(
        self,
        model_path: Path = None,
        preprocessor_path: Path = None
    ):
        """Inicializa el servicio con las rutas al modelo y preprocesador.
        
        Args:
            model_path: Ruta al archivo del modelo.
            preprocessor_path: Ruta al archivo del preprocesador.
        """
        self.model_path = model_path or (MODELS_DIR / "decision_tree_model.pkl")
        self.preprocessor_path = preprocessor_path or (MODELS_DIR / PREPROCESSING_PIPELINE_FILE)
        self.model: ClassifierMixin | None = None
        self.preprocessor: ColumnTransformer | None = None
        self._load_artifacts()
    
    def _load_artifacts(self) -> None:
        """Carga el modelo y el preprocesador desde los archivos."""
        try:
            logger.info("Cargando modelo y preprocesador...")
            self.model = load_model(self.model_path)
            self.preprocessor = load_preprocessing_pipeline(self.preprocessor_path)
            logger.info("Modelo y preprocesador cargados correctamente")
        except FileNotFoundError as e:
            logger.error(f"Error al cargar artefactos: {e}")
            raise
    
    def predict(self, input_data: dict) -> tuple[int, float]:
        """Realiza una predicción con los datos de entrada.
        
        Args:
            input_data: Diccionario con las características del cliente.
        
        Returns:
            Tupla con (predicción, probabilidad).
        """
        if self.model is None or self.preprocessor is None:
            raise RuntimeError("Modelo o preprocesador no cargados")
        
        # Convertir a DataFrame
        df = pd.DataFrame([input_data])
        
        # Preprocesar
        X_transformed = self.preprocessor.transform(df)
        
        # Predecir
        prediction = self.model.predict(X_transformed)[0]
        probabilities = self.model.predict_proba(X_transformed)[0]
        
        # Probabilidad de clase positiva (yes)
        probability = float(probabilities[1])
        
        return int(prediction), probability


# Instancia global del servicio (se inicializa al iniciar la app)
_model_service: ModelService | None = None


def get_model_service() -> ModelService:
    """Obtiene la instancia del servicio de modelo (singleton).
    
    Returns:
        Instancia del ModelService.
    """
    global _model_service
    if _model_service is None:
        _model_service = ModelService()
    return _model_service

