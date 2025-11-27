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
        self.is_loaded: bool = False
        self.load_error: str | None = None
        self._load_artifacts()
    
    def _load_artifacts(self) -> None:
        """Carga el modelo y el preprocesador desde los archivos."""
        try:
            logger.info("Cargando modelo y preprocesador...")
            logger.info(f"Buscando modelo en: {self.model_path}")
            logger.info(f"Buscando preprocesador en: {self.preprocessor_path}")
            
            if not self.model_path.exists():
                error_msg = (
                    f"El modelo no existe en la ruta: {self.model_path}. "
                    f"Por favor, entrena el modelo ejecutando: python train_model.py"
                )
                logger.error(error_msg)
                self.load_error = error_msg
                self.is_loaded = False
                return
            
            if not self.preprocessor_path.exists():
                error_msg = (
                    f"El preprocesador no existe en la ruta: {self.preprocessor_path}. "
                    f"Por favor, entrena el modelo ejecutando: python train_model.py"
                )
                logger.error(error_msg)
                self.load_error = error_msg
                self.is_loaded = False
                return
            
            self.model = load_model(self.model_path)
            self.preprocessor = load_preprocessing_pipeline(self.preprocessor_path)
            self.is_loaded = True
            self.load_error = None
            logger.info("Modelo y preprocesador cargados correctamente")
        except Exception as e:
            error_msg = f"Error al cargar artefactos: {str(e)}"
            logger.error(error_msg, exc_info=True)
            self.load_error = error_msg
            self.is_loaded = False
    
    def predict(self, input_data: dict) -> tuple[int, float]:
        """Realiza una predicción con los datos de entrada.
        
        Args:
            input_data: Diccionario con las características del cliente.
        
        Returns:
            Tupla con (predicción, probabilidad).
        
        Raises:
            RuntimeError: Si el modelo o preprocesador no están cargados.
        """
        if not self.is_loaded:
            error_msg = (
                "El modelo no está disponible. "
                f"{self.load_error if self.load_error else 'El modelo no ha sido cargado correctamente.'} "
                "Por favor, entrena el modelo ejecutando: python train_model.py"
            )
            raise RuntimeError(error_msg)
        
        if self.model is None or self.preprocessor is None:
            error_msg = (
                "El modelo o preprocesador no están disponibles. "
                "Por favor, entrena el modelo ejecutando: python train_model.py"
            )
            raise RuntimeError(error_msg)
        
        try:
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
        except Exception as e:
            logger.error(f"Error durante la predicción: {str(e)}", exc_info=True)
            raise RuntimeError(f"Error durante la predicción: {str(e)}")


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

