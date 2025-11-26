"""Transformaciones personalizadas para el preprocesamiento."""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class UnknownValueHandler(BaseEstimator, TransformerMixin):
    """Manejador de valores 'unknown' en variables categóricas.
    
    Convierte los valores 'unknown' en una categoría propia en lugar de
    imputarlos, preservando la información de que el valor era desconocido.
    """
    
    def __init__(self, unknown_values: list[str] = None):
        """Inicializa el transformador.
        
        Args:
            unknown_values: Lista de valores a tratar como desconocidos.
                           Por defecto ['unknown'].
        """
        self.unknown_values = unknown_values or ['unknown']
    
    def fit(self, X: pd.DataFrame, y: np.ndarray = None) -> 'UnknownValueHandler':
        """Ajusta el transformador (no hace nada en este caso).
        
        Args:
            X: DataFrame con los datos.
            y: Variable objetivo (opcional).
        
        Returns:
            self
        """
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transforma los valores desconocidos en una categoría propia.
        
        Args:
            X: DataFrame con los datos a transformar.
        
        Returns:
            DataFrame con los valores 'unknown' convertidos a categoría propia.
        """
        X = X.copy()
        for col in X.columns:
            if X[col].dtype == 'object':
                for unknown_val in self.unknown_values:
                    # Reemplazar 'unknown' con 'unknown_' + nombre_columna
                    X[col] = X[col].replace(
                        unknown_val,
                        f'unknown_{col}'
                    )
        return X

