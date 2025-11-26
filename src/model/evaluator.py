"""Evaluación de métricas del modelo."""

from datetime import datetime
from typing import Dict, List

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
)

from src.utils.logging_config import logger


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_pred_proba: np.ndarray = None,
    model_name: str = "Decision Tree - Bank Marketing"
) -> Dict:
    """Calcula todas las métricas de evaluación del modelo.
    
    Args:
        y_true: Valores reales.
        y_pred: Predicciones binarias.
        y_pred_proba: Probabilidades predichas (opcional, necesario para AUC-ROC).
        model_name: Nombre del modelo.
    
    Returns:
        Diccionario con todas las métricas calculadas.
    """
    logger.info("Calculando métricas del modelo...")
    
    # Métricas básicas
    accuracy = float(accuracy_score(y_true, y_pred))
    precision = float(precision_score(y_true, y_pred, zero_division=0))
    recall = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    
    # Matriz de confusión (formato: [[tn, fp], [fn, tp]])
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    confusion_matrix_formatted = [[int(tn), int(fp)], [int(fn), int(tp)]]
    
    # AUC-ROC (si tenemos probabilidades)
    auc_roc = None
    if y_pred_proba is not None:
        try:
            auc_roc = float(roc_auc_score(y_true, y_pred_proba))
        except ValueError:
            logger.warning("No se pudo calcular AUC-ROC (probablemente solo una clase presente)")
            auc_roc = 0.0
    
    # Classification report
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    
    # Curvas ROC y Precision-Recall
    roc_curve_data = None
    pr_curve_data = None
    if y_pred_proba is not None:
        try:
            fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
            roc_curve_data = {
                'fpr': fpr.tolist(),
                'tpr': tpr.tolist(),
            }
            
            precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_pred_proba)
            pr_curve_data = {
                'precision': precision_curve.tolist(),
                'recall': recall_curve.tolist(),
            }
        except ValueError:
            logger.warning("No se pudieron calcular las curvas ROC/PR")
    
    metrics = {
        "model": model_name,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "auc_roc": auc_roc if auc_roc is not None else 0.0,
        "confusion_matrix": confusion_matrix_formatted,
        "classification_report": report,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "roc_curve": roc_curve_data,
        "precision_recall_curve": pr_curve_data,
    }
    
    logger.info(f"Métricas calculadas - Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
    
    return metrics

