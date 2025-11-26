"""Tests para el módulo de preprocesamiento."""

import pytest
import pandas as pd
import numpy as np

from src.preprocessing.pipeline import (
    create_preprocessing_pipeline,
    fit_preprocessing_pipeline,
    transform_data,
)
from src.preprocessing.transformations import UnknownValueHandler


def test_unknown_value_handler():
    """Test para UnknownValueHandler."""
    handler = UnknownValueHandler(unknown_values=['unknown'])
    
    df = pd.DataFrame({
        'col1': ['a', 'unknown', 'b'],
        'col2': ['x', 'y', 'unknown'],
        'col3': [1, 2, 3]  # Numérica
    })
    
    transformed = handler.transform(df)
    
    assert 'unknown_col1' in transformed['col1'].values
    assert 'unknown_col2' in transformed['col2'].values
    assert transformed['col3'].equals(df['col3'])


def test_preprocessing_pipeline():
    """Test para crear el pipeline de preprocesamiento."""
    preprocessor = create_preprocessing_pipeline()
    assert preprocessor is not None


def test_fit_and_transform_pipeline(sample_dataframe):
    """Test para ajustar y transformar datos con el pipeline."""
    # Separar target
    X = sample_dataframe.drop(columns=['deposit'])
    
    # Crear y ajustar pipeline
    preprocessor = fit_preprocessing_pipeline(X)
    
    # Transformar
    X_transformed = transform_data(X, preprocessor)
    
    assert X_transformed is not None
    assert len(X_transformed) == len(X)
    assert isinstance(X_transformed, pd.DataFrame)

