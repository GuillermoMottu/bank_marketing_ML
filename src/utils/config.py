"""Configuración centralizada del proyecto."""

import os
from pathlib import Path
from typing import Optional

# Rutas del proyecto
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

# Nombres de archivos
RAW_DATA_FILE = "bank.csv"
MODEL_FILE = "decision_tree_model.pkl"
PREPROCESSING_PIPELINE_FILE = "preprocessing_pipeline.pkl"

# Configuración de base de datos
DB_HOST: str = os.getenv("DB_HOST", "db")
DB_PORT: int = int(os.getenv("DB_PORT", "5432"))
DB_NAME: str = os.getenv("DB_NAME", "bank_marketing")
DB_USER: str = os.getenv("DB_USER", "postgres")
DB_PASSWORD: str = os.getenv("DB_PASSWORD", "postgres")

# URL de conexión a la base de datos
DATABASE_URL: str = (
    f"postgresql+asyncpg://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
)

# Configuración de API
API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
API_PORT: int = int(os.getenv("API_PORT", "8000"))

# Configuración de Dashboard
DASHBOARD_HOST: str = os.getenv("DASHBOARD_HOST", "0.0.0.0")
DASHBOARD_PORT: int = int(os.getenv("DASHBOARD_PORT", "8050"))

# Configuración de modelo
RANDOM_STATE: int = 42
TEST_SIZE: float = 0.2
CV_FOLDS: int = 10

# Variable objetivo
TARGET_COLUMN: str = "deposit"

