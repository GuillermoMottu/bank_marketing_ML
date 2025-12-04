"""Aplicación principal del Dashboard Dash."""

import json
from datetime import datetime
from typing import Dict, List

import dash
import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import dcc, html, Input, Output, State
from dash.exceptions import PreventUpdate

from src.dashboard.components import create_input_field, create_metric_card
from src.database.connection import AsyncSessionLocal
from src.database.crud import get_all_metrics, get_latest_metrics
from src.utils.config import API_HOST, API_PORT
from src.utils.logging_config import logger

import httpx
import os

# URL base de la API - usar variable de entorno si está disponible (Railway)
# Si no, usar configuración local para desarrollo
API_BASE_URL = os.getenv(
    "API_BASE_URL",
    f"http://{API_HOST}:{API_PORT}/api/v1"
)

# Inicializar la app Dash
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    suppress_callback_exceptions=True
)

app.title = "Bank Marketing - Model Dashboard"

# Layout principal
app.layout = dbc.Container([
    dcc.Interval(
        id='interval-component',
        interval=30*1000,  # Actualizar cada 30 segundos
        n_intervals=0
    ),
    
    # Header
    html.H1("Bank Marketing Decision Tree Dashboard", className="text-center my-4"),
    
    # Tabs
    dbc.Tabs([
        dbc.Tab(label="Métricas", tab_id="metrics-tab"),
        dbc.Tab(label="Predicción", tab_id="prediction-tab"),
        dbc.Tab(label="Entrenar DL", tab_id="training-tab"),
    ], id="tabs", active_tab="metrics-tab"),
    
    html.Div(id="tab-content", className="mt-4"),
    
    # Store para datos
    dcc.Store(id="metrics-store"),
    dcc.Store(id="prediction-store"),
    
], fluid=True)


def create_metrics_tab() -> html.Div:
    """Crea el contenido de la pestaña de métricas."""
    
    return html.Div([
        # Tarjetas de métricas actuales
        html.H3("Métricas Actuales", className="mb-3"),
        dbc.Row([
            dbc.Col(id="accuracy-card", md=2),
            dbc.Col(id="precision-card", md=2),
            dbc.Col(id="recall-card", md=2),
            dbc.Col(id="f1-card", md=2),
            dbc.Col(id="auc-card", md=2),
            dbc.Col(id="hypothesis-card", md=2),
        ], className="mb-4"),
        
        # Gráficos históricos
        html.H3("Evolución Histórica", className="mb-3"),
        dbc.Row([
            dbc.Col([
                dcc.Graph(id="historical-metrics-chart")
            ], md=12),
        ], className="mb-4"),
        
        # Matriz de confusión
        html.H3("Matriz de Confusión", className="mb-3"),
        dbc.Row([
            dbc.Col([
                dcc.Graph(id="confusion-matrix-chart")
            ], md=6),
            dbc.Col([
                dcc.Graph(id="roc-curve-chart")
            ], md=6),
        ], className="mb-4"),
        
        # Curva Precision-Recall
        html.H3("Curva Precision-Recall", className="mb-3"),
        dbc.Row([
            dbc.Col([
                dcc.Graph(id="pr-curve-chart")
            ], md=12),
        ], className="mb-4"),
        
        # Tabla histórica
        html.H3("Historial Completo", className="mb-3"),
        dbc.Row([
            dbc.Col([
                html.Div(id="metrics-table")
            ], md=12),
        ]),
    ])


def create_prediction_tab() -> html.Div:
    """Crea el contenido de la pestaña de predicción."""
    
    # Opciones para los campos select
    job_options = [
        {"label": "Admin", "value": "admin."},
        {"label": "Blue-collar", "value": "blue-collar"},
        {"label": "Entrepreneur", "value": "entrepreneur"},
        {"label": "Housemaid", "value": "housemaid"},
        {"label": "Management", "value": "management"},
        {"label": "Retired", "value": "retired"},
        {"label": "Self-employed", "value": "self-employed"},
        {"label": "Services", "value": "services"},
        {"label": "Student", "value": "student"},
        {"label": "Technician", "value": "technician"},
        {"label": "Unemployed", "value": "unemployed"},
        {"label": "Unknown", "value": "unknown"},
    ]
    
    marital_options = [
        {"label": "Single", "value": "single"},
        {"label": "Married", "value": "married"},
        {"label": "Divorced", "value": "divorced"},
    ]
    
    education_options = [
        {"label": "Primary", "value": "primary"},
        {"label": "Secondary", "value": "secondary"},
        {"label": "Tertiary", "value": "tertiary"},
        {"label": "Unknown", "value": "unknown"},
    ]
    
    month_options = [
        {"label": m.capitalize(), "value": m.lower()}
        for m in ["jan", "feb", "mar", "apr", "may", "jun",
                  "jul", "aug", "sep", "oct", "nov", "dec"]
    ]
    
    contact_options = [
        {"label": "Cellular", "value": "cellular"},
        {"label": "Telephone", "value": "telephone"},
        {"label": "Unknown", "value": "unknown"},
    ]
    
    poutcome_options = [
        {"label": "Success", "value": "success"},
        {"label": "Failure", "value": "failure"},
        {"label": "Other", "value": "other"},
        {"label": "Unknown", "value": "unknown"},
    ]
    
    return html.Div([
        html.H3("Formulario de Predicción", className="mb-3"),
        
        # Selectores de modelo
        dbc.Row([
            dbc.Col([
                create_input_field("input-model-type", "Tipo de Modelo", "select", options=[
                    {"label": "Machine Learning (ML)", "value": "ML"},
                    {"label": "Deep Learning (Deep)", "value": "Deep"},
                ], value="ML"),
            ], md=6),
            dbc.Col([
                # Siempre crear el componente para que exista en el DOM
                html.Div([
                    create_input_field("input-architecture", "Arquitectura", "select", options=[
                        {"label": "DNN (Red Neuronal Densa)", "value": "DNN"},
                        {"label": "CNN (Convolucional Tabular)", "value": "CNN"},
                    ], value="DNN")
                ], id="architecture-selector"),
            ], md=6),
        ], className="mb-3"),
        
        dbc.Row([
            dbc.Col([
                create_input_field("input-age", "Edad", "number", value=30, min=0, max=120),
                create_input_field("input-job", "Trabajo", "select", options=job_options, value="admin."),
                create_input_field("input-marital", "Estado Civil", "select", options=marital_options, value="married"),
                create_input_field("input-education", "Educación", "select", options=education_options, value="secondary"),
                create_input_field("input-default", "Default", "select", options=[
                    {"label": "No", "value": "no"},
                    {"label": "Yes", "value": "yes"},
                ], value="no"),
            ], md=6),
            dbc.Col([
                create_input_field("input-balance", "Balance", "number", value=0, step=0.01),
                create_input_field("input-housing", "Housing", "select", options=[
                    {"label": "Yes", "value": "yes"},
                    {"label": "No", "value": "no"},
                ], value="no"),
                create_input_field("input-loan", "Loan", "select", options=[
                    {"label": "Yes", "value": "yes"},
                    {"label": "No", "value": "no"},
                ], value="no"),
                create_input_field("input-contact", "Contact", "select", options=contact_options, value="unknown"),
                create_input_field("input-day", "Día", "number", value=1, min=1, max=31),
            ], md=6),
        ], className="mb-3"),
        
        dbc.Row([
            dbc.Col([
                create_input_field("input-month", "Mes", "select", options=month_options, value="may"),
                create_input_field("input-duration", "Duración (segundos)", "number", value=0, min=0),
                create_input_field("input-campaign", "Campaign", "number", value=1, min=1),
                create_input_field("input-pdays", "PDays", "number", value=-1),
                create_input_field("input-previous", "Previous", "number", value=0, min=0),
                create_input_field("input-poutcome", "Poutcome", "select", options=poutcome_options, value="unknown"),
            ], md=12),
        ], className="mb-3"),
        
        dbc.Button(
            "Predecir",
            id="predict-button",
            color="primary",
            size="lg",
            className="mb-3"
        ),
        
        html.Div(id="prediction-result"),
    ])


def create_training_tab() -> html.Div:
    """Crea el contenido de la pestaña de entrenamiento."""
    
    return html.Div([
        html.H3("Entrenar Modelo de Deep Learning", className="mb-3"),
        html.P("Configura los parámetros y entrena un modelo de Deep Learning (DNN o CNN).", className="mb-4"),
        
        dbc.Row([
            dbc.Col([
                create_input_field("train-architecture", "Arquitectura", "select", options=[
                    {"label": "DNN (Red Neuronal Densa)", "value": "DNN"},
                    {"label": "CNN (Convolucional Tabular)", "value": "CNN"},
                ], value="DNN"),
                create_input_field("train-epochs", "Épocas", "number", value=100, min=1),
                create_input_field("train-batch-size", "Batch Size", "number", value=32, min=1),
                create_input_field("train-dropout", "Dropout Rate", "number", value=0.3, min=0.0, max=1.0, step=0.1),
            ], md=6),
            dbc.Col([
                create_input_field("train-scaler", "Tipo de Escalador", "select", options=[
                    {"label": "Standard Scaler", "value": "standard"},
                    {"label": "MinMax Scaler", "value": "minmax"},
                ], value="standard"),
                create_input_field("train-validation-split", "Validation Split", "number", value=0.2, min=0.0, max=0.5, step=0.1),
                create_input_field("train-early-stopping-patience", "Early Stopping Patience", "number", value=10, min=1),
                dbc.Checklist(
                    options=[{"label": "Usar Batch Normalization", "value": "use_batch_norm"}],
                    value=["use_batch_norm"],
                    id="train-batch-norm",
                    className="mt-4"
                ),
            ], md=6),
        ], className="mb-3"),
        
        dbc.Button(
            "Entrenar Modelo",
            id="train-button",
            color="success",
            size="lg",
            className="mb-3"
        ),
        
        html.Div(id="training-result"),
    ])


@app.callback(
    Output("tab-content", "children"),
    Input("tabs", "active_tab")
)
def update_tab_content(active_tab: str):
    """Actualiza el contenido según la pestaña activa."""
    if active_tab == "metrics-tab":
        return create_metrics_tab()
    elif active_tab == "prediction-tab":
        return create_prediction_tab()
    elif active_tab == "training-tab":
        return create_training_tab()
    return html.Div()


@app.callback(
    Output("architecture-selector", "style"),
    Input("input-model-type", "value")
)
def update_architecture_selector(model_type: str):
    """Muestra u oculta el selector de arquitectura según el tipo de modelo."""
    if model_type == "Deep":
        return {"display": "block"}
    else:
        return {"display": "none"}


@app.callback(
    Output("metrics-store", "data"),
    Input("interval-component", "n_intervals")
)
def update_metrics_store(n_intervals: int):
    """Actualiza los datos de métricas desde la base de datos."""
    import asyncio
    try:
        async def _fetch():
            async with AsyncSessionLocal() as session:
                try:
                    metrics_records = await get_all_metrics(session, limit=100)
                    return [
                        {
                            "id": record.id,
                            "model_name": record.model_name,
                            "metrics": record.metrics_json,
                            "created_at": record.created_at.isoformat()
                        }
                        for record in metrics_records
                    ]
                finally:
                    await session.close()
        return asyncio.run(_fetch())
    except Exception as e:
        logger.error(f"Error obteniendo métricas: {e}")
        return []


@app.callback(
    [
        Output("accuracy-card", "children"),
        Output("precision-card", "children"),
        Output("recall-card", "children"),
        Output("f1-card", "children"),
        Output("auc-card", "children"),
        Output("hypothesis-card", "children"),
    ],
    Input("metrics-store", "data")
)
def update_metric_cards(metrics_data: List[Dict]):
    """Actualiza las tarjetas de métricas con los últimos datos."""
    if not metrics_data:
        return [html.Div("Sin datos")] * 6
    
    latest = metrics_data[0]["metrics"]
    
    cards = [
        create_metric_card("Accuracy", latest.get("accuracy", 0), "primary"),
        create_metric_card("Precision", latest.get("precision", 0), "success"),
        create_metric_card("Recall", latest.get("recall", 0), "info"),
        create_metric_card("F1-Score", latest.get("f1_score", 0), "warning"),
        create_metric_card("AUC-ROC", latest.get("auc_roc", 0), "danger"),
    ]
    
    # Tarjeta de hipótesis (calcular p-value)
    accuracy = latest.get("accuracy", 0)
    # Aquí calcularíamos el p-value, por ahora mostramos accuracy
    cards.append(
        create_metric_card("Accuracy (H0: ≤0.5)", accuracy, "secondary")
    )
    
    return cards


@app.callback(
    Output("historical-metrics-chart", "figure"),
    Input("metrics-store", "data")
)
def update_historical_chart(metrics_data: List[Dict]):
    """Actualiza el gráfico histórico de métricas."""
    if not metrics_data:
        return go.Figure()
    
    df = pd.DataFrame([
        {
            "timestamp": pd.to_datetime(m["created_at"]),
            "accuracy": m["metrics"].get("accuracy", 0),
            "precision": m["metrics"].get("precision", 0),
            "recall": m["metrics"].get("recall", 0),
            "f1_score": m["metrics"].get("f1_score", 0),
            "auc_roc": m["metrics"].get("auc_roc", 0),
        }
        for m in metrics_data
    ])
    
    fig = go.Figure()
    
    for metric in ["accuracy", "precision", "recall", "f1_score", "auc_roc"]:
        fig.add_trace(go.Scatter(
            x=df["timestamp"],
            y=df[metric],
            name=metric.replace("_", " ").title(),
            mode="lines+markers"
        ))
    
    fig.update_layout(
        title="Evolución de Métricas",
        xaxis_title="Fecha",
        yaxis_title="Valor",
        hovermode="x unified"
    )
    
    return fig


@app.callback(
    Output("confusion-matrix-chart", "figure"),
    Input("metrics-store", "data")
)
def update_confusion_matrix(metrics_data: List[Dict]):
    """Actualiza la matriz de confusión."""
    if not metrics_data:
        return go.Figure()
    
    cm = metrics_data[0]["metrics"].get("confusion_matrix", [[0, 0], [0, 0]])
    
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=["Predicted: No", "Predicted: Yes"],
        y=["Actual: No", "Actual: Yes"],
        colorscale="Blues",
        text=cm,
        texttemplate="%{text}",
        textfont={"size": 16},
        showscale=True
    ))
    
    fig.update_layout(
        title="Matriz de Confusión",
        xaxis_title="Predicción",
        yaxis_title="Real"
    )
    
    return fig


@app.callback(
    [
        Output("roc-curve-chart", "figure"),
        Output("pr-curve-chart", "figure"),
    ],
    Input("metrics-store", "data")
)
def update_curves(metrics_data: List[Dict]):
    """Actualiza las curvas ROC y Precision-Recall."""
    roc_fig = go.Figure()
    pr_fig = go.Figure()
    
    if metrics_data:
        latest = metrics_data[0]["metrics"]
        
        # ROC Curve
        roc_curve = latest.get("roc_curve")
        if roc_curve:
            roc_fig.add_trace(go.Scatter(
                x=roc_curve["fpr"],
                y=roc_curve["tpr"],
                name="ROC Curve",
                mode="lines"
            ))
            roc_fig.add_trace(go.Scatter(
                x=[0, 1],
                y=[0, 1],
                name="Random",
                mode="lines",
                line=dict(dash="dash")
            ))
            roc_fig.update_layout(
                title="Curva ROC",
                xaxis_title="False Positive Rate",
                yaxis_title="True Positive Rate"
            )
        
        # Precision-Recall Curve
        pr_curve = latest.get("precision_recall_curve")
        if pr_curve:
            pr_fig.add_trace(go.Scatter(
                x=pr_curve["recall"],
                y=pr_curve["precision"],
                name="PR Curve",
                mode="lines"
            ))
            pr_fig.update_layout(
                title="Curva Precision-Recall",
                xaxis_title="Recall",
                yaxis_title="Precision"
            )
    
    return roc_fig, pr_fig


@app.callback(
    Output("metrics-table", "children"),
    Input("metrics-store", "data")
)
def update_metrics_table(metrics_data: List[Dict]):
    """Actualiza la tabla de historial de métricas."""
    if not metrics_data:
        return html.Div("Sin datos")
    
    df = pd.DataFrame([
        {
            "Fecha": pd.to_datetime(m["created_at"]).strftime("%Y-%m-%d %H:%M:%S"),
            "Accuracy": f"{m['metrics'].get('accuracy', 0):.4f}",
            "Precision": f"{m['metrics'].get('precision', 0):.4f}",
            "Recall": f"{m['metrics'].get('recall', 0):.4f}",
            "F1-Score": f"{m['metrics'].get('f1_score', 0):.4f}",
            "AUC-ROC": f"{m['metrics'].get('auc_roc', 0):.4f}",
        }
        for m in metrics_data
    ])
    
    return dbc.Table.from_dataframe(df, striped=True, bordered=True, hover=True, size="sm")


@app.callback(
    Output("prediction-result", "children"),
    Input("predict-button", "n_clicks"),
    State("input-model-type", "value"),
    State("input-architecture", "value"),
    State("input-age", "value"),
    State("input-job", "value"),
    State("input-marital", "value"),
    State("input-education", "value"),
    State("input-default", "value"),
    State("input-balance", "value"),
    State("input-housing", "value"),
    State("input-loan", "value"),
    State("input-contact", "value"),
    State("input-day", "value"),
    State("input-month", "value"),
    State("input-duration", "value"),
    State("input-campaign", "value"),
    State("input-pdays", "value"),
    State("input-previous", "value"),
    State("input-poutcome", "value"),
)
def make_prediction(n_clicks, model_type, architecture, *args):
    """Realiza una predicción usando la API."""
    if n_clicks is None:
        raise PreventUpdate
    
    # Validar que si es Deep Learning, se especifique la arquitectura
    if model_type == "Deep" and architecture is None:
        return dbc.Alert("Por favor selecciona una arquitectura para Deep Learning.", color="warning")
    
    # Construir payload
    input_data = {
        "age": args[0],
        "job": args[1],
        "marital": args[2],
        "education": args[3],
        "default": args[4],
        "balance": args[5],
        "housing": args[6],
        "loan": args[7],
        "contact": args[8],
        "day": args[9],
        "month": args[10],
        "duration": args[11],
        "campaign": args[12],
        "pdays": args[13],
        "previous": args[14],
        "poutcome": args[15],
        "model_type": model_type or "ML",
        "architecture": architecture if model_type == "Deep" else None,
    }
    
    try:
        with httpx.Client() as client:
            response = client.post(
                f"{API_BASE_URL}/predict",
                json=input_data,
                timeout=10.0
            )
            response.raise_for_status()
            result = response.json()
            
            color = "success" if result["prediction"] == 1 else "danger"
            result_text = "SÍ" if result["prediction"] == 1 else "NO"
            
            return dbc.Alert([
                html.H4(f"Predicción: {result_text}"),
                html.P(f"Probabilidad: {result['probability']:.4f} ({result['probability']*100:.2f}%)"),
                html.P(f"El cliente {'probablemente SÍ' if result['prediction'] == 1 else 'probablemente NO'} suscribirá un depósito."),
            ], color=color, className="mt-3")
    
    except Exception as e:
        logger.error(f"Error en predicción: {e}")
        return dbc.Alert(f"Error al realizar la predicción: {str(e)}", color="danger")


@app.callback(
    Output("training-result", "children"),
    Input("train-button", "n_clicks"),
    State("train-architecture", "value"),
    State("train-epochs", "value"),
    State("train-batch-size", "value"),
    State("train-dropout", "value"),
    State("train-scaler", "value"),
    State("train-validation-split", "value"),
    State("train-early-stopping-patience", "value"),
    State("train-batch-norm", "value"),
)
def train_model(n_clicks, architecture, epochs, batch_size, dropout, scaler, validation_split, patience, batch_norm):
    """Entrena un modelo de Deep Learning usando la API."""
    if n_clicks is None:
        raise PreventUpdate
    
    # Construir payload
    training_data = {
        "architecture": architecture,
        "epochs": epochs or 100,
        "batch_size": batch_size or 32,
        "dropout_rate": dropout or 0.3,
        "scaler_type": scaler or "standard",
        "validation_split": validation_split or 0.2,
        "early_stopping_patience": patience or 10,
        "use_batch_norm": "use_batch_norm" in (batch_norm or []),
        "early_stopping": True,
    }
    
    try:
        with httpx.Client(timeout=300.0) as client:  # Timeout largo para entrenamiento
            response = client.post(
                f"{API_BASE_URL}/train",
                json=training_data,
            )
            response.raise_for_status()
            result = response.json()
            
            if result.get("success"):
                metrics_info = ""
                if result.get("metrics"):
                    metrics = result["metrics"]
                    metrics_info = html.Div([
                        html.H5("Métricas del Modelo Entrenado:", className="mt-3"),
                        html.P(f"Accuracy: {metrics.get('accuracy', 0):.4f}"),
                        html.P(f"Precision: {metrics.get('precision', 0):.4f}"),
                        html.P(f"Recall: {metrics.get('recall', 0):.4f}"),
                        html.P(f"F1-Score: {metrics.get('f1_score', 0):.4f}"),
                        html.P(f"AUC-ROC: {metrics.get('auc_roc', 0):.4f}"),
                    ])
                
                return dbc.Alert([
                    html.H4("Entrenamiento Completado Exitosamente"),
                    html.P(result.get("message", "")),
                    html.P(f"Modelo: {result.get('model_name', 'N/A')}"),
                    metrics_info,
                ], color="success", className="mt-3")
            else:
                return dbc.Alert([
                    html.H4("Error en el Entrenamiento"),
                    html.P(result.get("message", "Error desconocido")),
                ], color="danger", className="mt-3")
    
    except httpx.TimeoutException:
        return dbc.Alert("El entrenamiento está tomando más tiempo del esperado. Por favor verifica los logs del servidor.", color="warning")
    except Exception as e:
        logger.error(f"Error en entrenamiento: {e}")
        return dbc.Alert(f"Error al entrenar el modelo: {str(e)}", color="danger")
