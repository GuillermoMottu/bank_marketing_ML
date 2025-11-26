"""Tests para el módulo de modelo."""

import pytest
import numpy as np
import pandas as pd

from src.model.evaluator import calculate_metrics


def test_calculate_metrics():
    """Test para calcular métricas."""
    y_true = np.array([0, 1, 1, 0, 1, 0, 1, 1, 0, 0])
    y_pred = np.array([0, 1, 1, 0, 1, 0, 1, 0, 0, 0])
    y_pred_proba = np.array([0.1, 0.9, 0.8, 0.2, 0.7, 0.3, 0.9, 0.4, 0.2, 0.1])
    
    metrics = calculate_metrics(y_true, y_pred, y_pred_proba)
    
    assert "accuracy" in metrics
    assert "precision" in metrics
    assert "recall" in metrics
    assert "f1_score" in metrics
    assert "auc_roc" in metrics
    assert "confusion_matrix" in metrics
    assert "classification_report" in metrics
    assert "timestamp" in metrics
    
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert 0.0 <= metrics["precision"] <= 1.0
    assert 0.0 <= metrics["recall"] <= 1.0
    assert 0.0 <= metrics["f1_score"] <= 1.0
    assert isinstance(metrics["confusion_matrix"], list)
    assert len(metrics["confusion_matrix"]) == 2

