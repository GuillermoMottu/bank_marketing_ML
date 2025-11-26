"""Modelos Pydantic para validación de entrada/salida de la API."""

from typing import Literal, Optional

from pydantic import BaseModel, Field


class PredictionInput(BaseModel):
    """Modelo de entrada para la predicción.
    
    Representa todas las características necesarias para hacer una predicción.
    """
    
    age: int = Field(..., ge=0, le=120, description="Edad del cliente")
    job: str = Field(..., description="Tipo de trabajo")
    marital: Literal["single", "married", "divorced"] = Field(..., description="Estado civil")
    education: str = Field(..., description="Nivel educativo")
    default: Literal["yes", "no"] = Field(..., description="Tiene crédito en default?")
    balance: float = Field(..., description="Balance promedio anual")
    housing: Literal["yes", "no"] = Field(..., description="Tiene préstamo hipotecario?")
    loan: Literal["yes", "no"] = Field(..., description="Tiene préstamo personal?")
    contact: str = Field(..., description="Tipo de contacto de comunicación")
    day: int = Field(..., ge=1, le=31, description="Día del último contacto")
    month: str = Field(..., description="Mes del último contacto")
    duration: int = Field(..., ge=0, description="Duración del último contacto en segundos")
    campaign: int = Field(..., ge=1, description="Número de contactos durante esta campaña")
    pdays: int = Field(..., description="Número de días pasados desde el último contacto")
    previous: int = Field(..., ge=0, description="Número de contactos antes de esta campaña")
    poutcome: str = Field(..., description="Resultado de la campaña de marketing anterior")
    
    class Config:
        json_schema_extra = {
            "example": {
                "age": 59,
                "job": "admin.",
                "marital": "married",
                "education": "secondary",
                "default": "no",
                "balance": 2343,
                "housing": "yes",
                "loan": "no",
                "contact": "unknown",
                "day": 5,
                "month": "may",
                "duration": 1042,
                "campaign": 1,
                "pdays": -1,
                "previous": 0,
                "poutcome": "unknown"
            }
        }


class PredictionOutput(BaseModel):
    """Modelo de salida para la predicción."""
    
    prediction: int = Field(..., description="Predicción binaria (0=no, 1=yes)")
    probability: float = Field(..., ge=0.0, le=1.0, description="Probabilidad de que sea 'yes'")
    class_name: str = Field(..., description="Nombre de la clase predicha")


class MetricsOutput(BaseModel):
    """Modelo de salida para las métricas del modelo."""
    
    model: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    auc_roc: float
    confusion_matrix: list[list[int]]
    classification_report: dict
    timestamp: str
    roc_curve: Optional[dict] = None
    precision_recall_curve: Optional[dict] = None

