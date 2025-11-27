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
# Siempre definir variables individuales (pueden usarse en otros lugares)
DB_HOST: str = os.getenv("DB_HOST", "db")
DB_PORT: int = int(os.getenv("DB_PORT", "5432"))
DB_NAME: str = os.getenv("DB_NAME", "bank_marketing")
DB_USER: str = os.getenv("DB_USER", "postgres")
DB_PASSWORD: str = os.getenv("DB_PASSWORD", "postgres")

# Railway y otras plataformas pueden proporcionar DATABASE_URL directamente
# Si está disponible, usarla; si no, construirla desde variables individuales
if os.getenv("DATABASE_URL"):
    # Si DATABASE_URL está disponible, usarla directamente
    # Asegurar que use asyncpg si no lo especifica
    db_url = os.getenv("DATABASE_URL")
    if db_url.startswith("postgresql://"):
        # Convertir postgresql:// a postgresql+asyncpg://
        DATABASE_URL = db_url.replace("postgresql://", "postgresql+asyncpg://", 1)
    elif db_url.startswith("postgresql+asyncpg://"):
        DATABASE_URL = db_url
    else:
        # Si tiene otro formato, construir desde variables individuales
        DATABASE_URL = (
            f"postgresql+asyncpg://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
        )
else:
    # Construir desde variables individuales
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

