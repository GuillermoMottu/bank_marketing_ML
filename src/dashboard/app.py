"""Aplicación principal del Dashboard Dash."""

import json
import threading
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

import httpx

from src.dashboard.components import create_input_field, create_metric_card
from src.database.connection import AsyncSessionLocal
from src.database.crud import get_all_metrics, get_latest_metrics
from src.utils.config import API_HOST, API_PORT
from src.utils.logging_config import logger

# URL base de la API
API_BASE_URL = f"http://api:8000/api/v1"

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
    return html.Div()


# Lock para evitar operaciones concurrentes en la base de datos
_db_lock = threading.Lock()

def fetch_metrics_sync() -> list:
    """Función helper para obtener métricas de forma síncrona desde el dashboard.
    
    Esta función maneja correctamente las operaciones asíncronas en un contexto
    síncrono, creando un engine completamente nuevo para cada operación en un thread
    separado con su propio event loop aislado.
    
    Returns:
        Lista de diccionarios con las métricas o lista vacía en caso de error.
    """
    import asyncio
    import concurrent.futures
    from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
    from src.utils.config import DATABASE_URL
    
    async def _fetch_metrics() -> list:
        """Función asíncrona para obtener métricas de la base de datos."""
        # Crear un engine completamente nuevo para este event loop
        engine = create_async_engine(
            DATABASE_URL,
            echo=False,
            pool_pre_ping=True,
            pool_size=1,
            max_overflow=0
        )
        
        # Crear un sessionmaker nuevo para este engine
        AsyncSessionLocal = async_sessionmaker(
            engine,
            class_=AsyncSession,
            expire_on_commit=False,
        )
        
        try:
            async with AsyncSessionLocal() as session:
                metrics_records = await get_all_metrics(session, limit=100)
                result = [
                    {
                        "id": record.id,
                        "model_name": record.model_name,
                        "metrics": record.metrics_json,
                        "created_at": record.created_at.isoformat()
                    }
                    for record in metrics_records
                ]
                return result
        except Exception as e:
            logger.error(f"Error obteniendo métricas: {e}", exc_info=True)
            return []
        finally:
            # Cerrar el engine completamente antes de salir
            await engine.dispose()
    
    def _run_in_new_loop():
        """Ejecuta la función asíncrona en un nuevo event loop completamente aislado."""
        # Crear un nuevo event loop para este thread
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(_fetch_metrics())
        except Exception as e:
            logger.error(f"Error en event loop: {e}", exc_info=True)
            return []
        finally:
            # Cerrar todas las tareas pendientes
            try:
                pending = asyncio.all_tasks(loop)
                for task in pending:
                    task.cancel()
                if pending:
                    loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
            except Exception:
                pass
            finally:
                loop.close()
    
    # Usar lock para evitar operaciones concurrentes y ejecutar en thread separado
    with _db_lock:
        try:
            # Siempre ejecutar en un thread separado para evitar conflictos con event loops
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(_run_in_new_loop)
                return future.result(timeout=15)
        except concurrent.futures.TimeoutError:
            logger.error("Timeout obteniendo métricas")
            return []
        except Exception as e:
            logger.error(f"Error en fetch_metrics_sync: {e}", exc_info=True)
            return []


@app.callback(
    Output("metrics-store", "data"),
    Input("interval-component", "n_intervals")
)
def update_metrics_store(n_intervals: int):
    """Actualiza los datos de métricas desde la base de datos."""
    return fetch_metrics_sync()


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
def make_prediction(n_clicks, *args):
    """Realiza una predicción usando la API."""
    if n_clicks is None:
        raise PreventUpdate
    
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

