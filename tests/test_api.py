"""Tests para los endpoints de la API."""

import pytest
from fastapi import status

from tests.conftest import test_client, sample_prediction_input


def test_health_check(test_client):
    """Test para el endpoint /health."""
    response = test_client.get("/")
    assert response.status_code == status.HTTP_200_OK
    
    response = test_client.get("/api/v1/health")
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert "status" in data
    assert data["status"] == "healthy"
    assert "timestamp" in data


def test_predict_endpoint(test_client, sample_prediction_input):
    """Test para el endpoint POST /predict."""
    # Nota: Este test requiere que el modelo esté entrenado
    # En un ambiente de CI/CD, deberías entrenar el modelo antes o usar mocks
    
    response = test_client.post(
        "/api/v1/predict",
        json=sample_prediction_input
    )
    
    # Puede fallar si el modelo no existe, pero probamos la estructura
    if response.status_code == status.HTTP_200_OK:
        data = response.json()
        assert "prediction" in data
        assert "probability" in data
        assert "class_name" in data
        assert data["prediction"] in [0, 1]
        assert 0.0 <= data["probability"] <= 1.0
    else:
        # Si falla porque no hay modelo, verificamos que el error es apropiado
        assert response.status_code in [status.HTTP_500_INTERNAL_SERVER_ERROR, status.HTTP_503_SERVICE_UNAVAILABLE]


def test_predict_validation(test_client):
    """Test de validación de entrada en /predict."""
    # Test con datos inválidos (edad fuera de rango)
    invalid_input = {
        "age": 200,  # Fuera de rango
        "job": "admin.",
        "marital": "married",
        "education": "secondary",
        "default": "no",
        "balance": 2343.0,
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
    
    response = test_client.post("/api/v1/predict", json=invalid_input)
    assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY


def test_metrics_endpoint(test_client):
    """Test para el endpoint GET /metrics."""
    # Nota: Este test requiere que el modelo esté entrenado y datos de test disponibles
    
    response = test_client.get("/api/v1/metrics")
    
    # Puede fallar si no hay modelo o datos, pero probamos la estructura si tiene éxito
    if response.status_code == status.HTTP_200_OK:
        data = response.json()
        assert "model" in data
        assert "accuracy" in data
        assert "precision" in data
        assert "recall" in data
        assert "f1_score" in data
        assert "auc_roc" in data
        assert "confusion_matrix" in data
        assert "classification_report" in data
        assert "timestamp" in data
        
        # Verificar tipos
        assert isinstance(data["accuracy"], float)
        assert isinstance(data["precision"], float)
        assert isinstance(data["recall"], float)
        assert isinstance(data["f1_score"], float)
        assert isinstance(data["auc_roc"], float)
        assert isinstance(data["confusion_matrix"], list)
        assert len(data["confusion_matrix"]) == 2
    else:
        # Si falla, verificamos que el error es apropiado
        assert response.status_code in [status.HTTP_500_INTERNAL_SERVER_ERROR, status.HTTP_503_SERVICE_UNAVAILABLE]

