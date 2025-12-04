"""Dependencias inyectables para FastAPI."""

from pathlib import Path
from typing import Literal, Optional

import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.compose import ColumnTransformer
from tensorflow import keras

from src.model.predict_dl import load_dl_model, predict_dl
from src.model.trainer import load_model
from src.preprocessing.dl_preprocessing import load_dl_preprocessing_pipeline
from src.preprocessing.pipeline import load_preprocessing_pipeline
from src.utils.config import (
    DL_MODEL_CNN_FILE,
    DL_MODEL_DNN_FILE,
    DL_PREPROCESSING_PIPELINE_FILE,
    MODELS_DIR,
    PREPROCESSING_PIPELINE_FILE,
)
from src.utils.logging_config import logger


class ModelService:
    """Servicio para cargar y gestionar modelos ML y DL con sus preprocesadores."""
    
    def __init__(
        self,
        model_path: Path = None,
        preprocessor_path: Path = None
    ):
        """Inicializa el servicio con las rutas al modelo y preprocesador ML.
        
        Args:
            model_path: Ruta al archivo del modelo ML.
            preprocessor_path: Ruta al archivo del preprocesador ML.
        """
        self.model_path = model_path or (MODELS_DIR / "decision_tree_model.pkl")
        self.preprocessor_path = preprocessor_path or (MODELS_DIR / PREPROCESSING_PIPELINE_FILE)
        self.model: ClassifierMixin | None = None
        self.preprocessor: ColumnTransformer | None = None
        self.is_loaded: bool = False
        self.load_error: str | None = None
        
        # Modelos DL
        self.dl_model_dnn: Optional[keras.Model] = None
        self.dl_model_cnn: Optional[keras.Model] = None
        self.dl_preprocessor: Optional[ColumnTransformer] = None
        self.dl_loaded: bool = False
        
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
    
    def load_dl_model(self, architecture: Literal["DNN", "CNN"]) -> None:
        """Carga un modelo de Deep Learning según la arquitectura especificada.
        
        Args:
            architecture: Tipo de arquitectura ('DNN' o 'CNN').
        """
        try:
            if architecture.upper() == "DNN":
                model_path = MODELS_DIR / DL_MODEL_DNN_FILE
                if model_path.exists():
                    self.dl_model_dnn = load_dl_model(model_path)
                    logger.info(f"Modelo DNN cargado desde {model_path}")
                else:
                    logger.warning(f"Modelo DNN no encontrado en {model_path}")
            elif architecture.upper() == "CNN":
                model_path = MODELS_DIR / DL_MODEL_CNN_FILE
                if model_path.exists():
                    self.dl_model_cnn = load_dl_model(model_path)
                    logger.info(f"Modelo CNN cargado desde {model_path}")
                else:
                    logger.warning(f"Modelo CNN no encontrado en {model_path}")
            
            # Cargar preprocesador DL si no está cargado
            if self.dl_preprocessor is None:
                preprocessor_path = MODELS_DIR / DL_PREPROCESSING_PIPELINE_FILE
                if preprocessor_path.exists():
                    self.dl_preprocessor = load_dl_preprocessing_pipeline(preprocessor_path)
                    logger.info("Preprocesador DL cargado")
                else:
                    logger.warning(f"Preprocesador DL no encontrado en {preprocessor_path}")
            
            self.dl_loaded = True
        except Exception as e:
            logger.error(f"Error al cargar modelo DL: {str(e)}", exc_info=True)
            self.dl_loaded = False
    
    def predict(self, input_data: dict, model_type: Literal["ML", "Deep"] = "ML", 
                architecture: Optional[Literal["DNN", "CNN"]] = None) -> tuple[int, float]:
        """Realiza una predicción con los datos de entrada.
        
        Args:
            input_data: Diccionario con las características del cliente.
            model_type: Tipo de modelo a usar ('ML' o 'Deep').
            architecture: Arquitectura para Deep Learning ('DNN' o 'CNN'). Requerido si model_type='Deep'.
        
        Returns:
            Tupla con (predicción, probabilidad).
        
        Raises:
            RuntimeError: Si el modelo o preprocesador no están cargados.
        """
        if model_type == "Deep":
            return self._predict_dl(input_data, architecture)
        else:
            return self._predict_ml(input_data)
    
    def _predict_ml(self, input_data: dict) -> tuple[int, float]:
        """Predicción con modelo ML tradicional."""
        if not self.is_loaded:
            error_msg = (
                "El modelo ML no está disponible. "
                f"{self.load_error if self.load_error else 'El modelo no ha sido cargado correctamente.'} "
                "Por favor, entrena el modelo ejecutando: python train_model.py"
            )
            raise RuntimeError(error_msg)
        
        if self.model is None or self.preprocessor is None:
            error_msg = (
                "El modelo o preprocesador ML no están disponibles. "
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
            logger.error(f"Error durante la predicción ML: {str(e)}", exc_info=True)
            raise RuntimeError(f"Error durante la predicción ML: {str(e)}")
    
    def _predict_dl(self, input_data: dict, architecture: Optional[Literal["DNN", "CNN"]]) -> tuple[int, float]:
        """Predicción con modelo Deep Learning."""
        if architecture is None:
            raise ValueError("La arquitectura es requerida para modelos Deep Learning. Use 'DNN' o 'CNN'")
        
        # Cargar modelo DL si no está cargado
        if architecture.upper() == "DNN":
            if self.dl_model_dnn is None:
                self.load_dl_model("DNN")
            if self.dl_model_dnn is None:
                raise RuntimeError(
                    f"El modelo DNN no está disponible. "
                    f"Por favor, entrena el modelo ejecutando el endpoint /train o python train_model.py"
                )
            model = self.dl_model_dnn
        elif architecture.upper() == "CNN":
            if self.dl_model_cnn is None:
                self.load_dl_model("CNN")
            if self.dl_model_cnn is None:
                raise RuntimeError(
                    f"El modelo CNN no está disponible. "
                    f"Por favor, entrena el modelo ejecutando el endpoint /train o python train_model.py"
                )
            model = self.dl_model_cnn
        else:
            raise ValueError(f"Arquitectura no soportada: {architecture}")
        
        if self.dl_preprocessor is None:
            raise RuntimeError(
                "El preprocesador DL no está disponible. "
                "Por favor, entrena el modelo ejecutando el endpoint /train"
            )
        
        try:
            return predict_dl(model, self.dl_preprocessor, input_data)
        except Exception as e:
            logger.error(f"Error durante la predicción DL: {str(e)}", exc_info=True)
            raise RuntimeError(f"Error durante la predicción DL: {str(e)}")


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

