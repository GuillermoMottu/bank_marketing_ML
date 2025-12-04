"""Arquitecturas de redes neuronales para Deep Learning."""

from typing import List, Optional

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models

from src.utils.logging_config import logger


def build_dnn(
    input_dim: int,
    hidden_layers: List[int] = [128, 64, 32],
    dropout_rate: float = 0.3,
    use_batch_norm: bool = True,
    activation: str = "relu",
    output_activation: str = "sigmoid"
) -> keras.Model:
    """Construye una red neuronal densa (DNN) para clasificación binaria.
    
    Args:
        input_dim: Dimensión de entrada (número de características).
        hidden_layers: Lista con el número de neuronas por capa oculta. Por defecto [128, 64, 32].
        dropout_rate: Tasa de dropout para regularización (0.0-1.0).
        use_batch_norm: Si es True, agrega Batch Normalization después de cada capa densa.
        activation: Función de activación para capas ocultas. Por defecto 'relu'.
        output_activation: Función de activación para capa de salida. Por defecto 'sigmoid'.
    
    Returns:
        Modelo Keras compilado.
    """
    logger.info(f"Construyendo DNN con {len(hidden_layers)} capas ocultas: {hidden_layers}")
    logger.info(f"Input dim: {input_dim}, Dropout: {dropout_rate}, BatchNorm: {use_batch_norm}")
    
    model = models.Sequential()
    
    # Capa de entrada
    model.add(layers.Input(shape=(input_dim,)))
    
    # Capas ocultas
    for i, units in enumerate(hidden_layers):
        model.add(layers.Dense(units, activation=activation, name=f"dense_{i+1}"))
        
        if use_batch_norm:
            model.add(layers.BatchNormalization(name=f"batch_norm_{i+1}"))
        
        if dropout_rate > 0:
            model.add(layers.Dropout(dropout_rate, name=f"dropout_{i+1}"))
    
    # Capa de salida (clasificación binaria)
    model.add(layers.Dense(1, activation=output_activation, name="output"))
    
    # Compilar modelo
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=["accuracy", "precision", "recall"]
    )
    
    logger.info(f"Modelo DNN construido con {model.count_params()} parámetros")
    
    return model


def build_cnn_tabular(
    input_dim: int,
    conv_filters: List[int] = [64, 32],
    conv_kernel_size: int = 3,
    pool_size: int = 2,
    dense_units: List[int] = [64, 32],
    dropout_rate: float = 0.3,
    use_batch_norm: bool = True,
    activation: str = "relu",
    output_activation: str = "sigmoid"
) -> keras.Model:
    """Construye una CNN 1D para datos tabulares.
    
    La CNN tabular funciona reshapeando los datos tabulares a formato 1D
    y aplicando convoluciones 1D.
    
    Args:
        input_dim: Dimensión de entrada (número de características).
        conv_filters: Lista con el número de filtros por capa convolucional. Por defecto [64, 32].
        conv_kernel_size: Tamaño del kernel para las capas convolucionales. Por defecto 3.
        pool_size: Tamaño del pooling. Por defecto 2.
        dense_units: Lista con el número de neuronas en capas densas finales. Por defecto [64, 32].
        dropout_rate: Tasa de dropout para regularización (0.0-1.0).
        use_batch_norm: Si es True, agrega Batch Normalization.
        activation: Función de activación para capas ocultas. Por defecto 'relu'.
        output_activation: Función de activación para capa de salida. Por defecto 'sigmoid'.
    
    Returns:
        Modelo Keras compilado.
    """
    logger.info(f"Construyendo CNN Tabular con {len(conv_filters)} capas convolucionales: {conv_filters}")
    logger.info(f"Input dim: {input_dim}, Dense units: {dense_units}, Dropout: {dropout_rate}")
    
    model = models.Sequential()
    
    # Reshape: convertir datos tabulares a formato 1D para convolución
    # input_dim características -> (input_dim, 1) para convolución 1D
    model.add(layers.Reshape((input_dim, 1), input_shape=(input_dim,), name="reshape"))
    
    # Capas convolucionales 1D
    for i, filters in enumerate(conv_filters):
        model.add(layers.Conv1D(
            filters=filters,
            kernel_size=conv_kernel_size,
            activation=activation,
            padding="same",
            name=f"conv1d_{i+1}"
        ))
        
        if use_batch_norm:
            model.add(layers.BatchNormalization(name=f"batch_norm_conv_{i+1}"))
        
        model.add(layers.MaxPooling1D(pool_size=pool_size, name=f"pooling_{i+1}"))
        
        if dropout_rate > 0:
            model.add(layers.Dropout(dropout_rate, name=f"dropout_conv_{i+1}"))
    
    # Flatten para conectar con capas densas
    model.add(layers.Flatten(name="flatten"))
    
    # Capas densas finales
    for i, units in enumerate(dense_units):
        model.add(layers.Dense(units, activation=activation, name=f"dense_{i+1}"))
        
        if use_batch_norm:
            model.add(layers.BatchNormalization(name=f"batch_norm_dense_{i+1}"))
        
        if dropout_rate > 0:
            model.add(layers.Dropout(dropout_rate, name=f"dropout_dense_{i+1}"))
    
    # Capa de salida (clasificación binaria)
    model.add(layers.Dense(1, activation=output_activation, name="output"))
    
    # Compilar modelo
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=["accuracy", "precision", "recall"]
    )
    
    logger.info(f"Modelo CNN Tabular construido con {model.count_params()} parámetros")
    
    return model


