"""Componentes reutilizables para el dashboard Dash."""

import dash_bootstrap_components as dbc
from dash import dcc, html


def create_metric_card(title: str, value: float, color: str = "primary") -> dbc.Card:
    """Crea una tarjeta con una métrica.
    
    Args:
        title: Título de la métrica.
        value: Valor numérico de la métrica.
        color: Color de la tarjeta (Bootstrap color).
    
    Returns:
        Componente Card de Dash Bootstrap.
    """
    formatted_value = f"{value:.4f}" if isinstance(value, float) else str(value)
    
    return dbc.Card(
        dbc.CardBody([
            html.H5(title, className="card-title"),
            html.H2(formatted_value, className="text-center"),
        ]),
        color=color,
        inverse=True,
        className="mb-3"
    )


def create_input_field(
    field_id: str,
    label: str,
    input_type: str = "text",
    placeholder: str = "",
    value: any = None,
    **kwargs
) -> dbc.FormGroup:
    """Crea un campo de entrada para el formulario.
    
    Args:
        field_id: ID del campo.
        label: Etiqueta del campo.
        input_type: Tipo de input (text, number, select, etc.).
        placeholder: Texto de placeholder.
        value: Valor por defecto.
        **kwargs: Argumentos adicionales para el componente.
    
    Returns:
        FormGroup con el campo de entrada.
    """
    if input_type == "number":
        component = dbc.Input(
            id=field_id,
            type="number",
            placeholder=placeholder,
            value=value,
            **kwargs
        )
    elif input_type == "select":
        component = dbc.Select(
            id=field_id,
            options=kwargs.get("options", []),
            value=value,
            **{k: v for k, v in kwargs.items() if k != "options"}
        )
    else:
        component = dbc.Input(
            id=field_id,
            type=input_type,
            placeholder=placeholder,
            value=value,
            **kwargs
        )
    
    return dbc.FormGroup([
        dbc.Label(label, html_for=field_id),
        component
    ])

