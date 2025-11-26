"""Configuración y fixtures para pytest."""

import asyncio
from pathlib import Path
from typing import AsyncGenerator

import pytest
import numpy as np
import pandas as pd
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import declarative_base

from src.api.main import app
from src.database.connection import Base, get_session
from src.utils.config import MODELS_DIR

# Base de datos de prueba (en memoria)
TEST_DATABASE_URL = "sqlite+aiosqlite:///:memory:"

# Motor de prueba
test_engine = create_async_engine(TEST_DATABASE_URL, echo=False)
TestSessionLocal = async_sessionmaker(
    test_engine, class_=AsyncSession, expire_on_commit=False
)


@pytest.fixture(scope="session")
def event_loop():
    """Crea un event loop para las pruebas async."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="function", autouse=True)
async def setup_db() -> AsyncGenerator[AsyncSession, None]:
    """Configura la base de datos de prueba antes de cada test."""
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    
    async with TestSessionLocal() as session:
        yield session
        await session.rollback()
    
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)


@pytest.fixture
def test_client(setup_db):
    """Crea un cliente de prueba para FastAPI."""
    # Sobrescribir la dependencia de la sesión de BD
    async def override_get_session():
        async with TestSessionLocal() as session:
            yield session
    
    app.dependency_overrides[get_session] = override_get_session
    
    with TestClient(app) as client:
        yield client
    
    app.dependency_overrides.clear()


@pytest.fixture
def sample_prediction_input():
    """Datos de ejemplo para una predicción."""
    return {
        "age": 59,
        "job": "admin.",
        "marital": "married",
        "education": "secondary",
        "default": "no",
        "balance": 2343.0,
        "housing": "yes",
        "loan": "no",
        "contact": "unknown",
        "day": 5,
        "month": "may",
        "duration": 1042,
        "campaign": 1,
        "pdays": -1,
        "previous": 0,
        "poutcome": "unknown"
    }


@pytest.fixture
def sample_dataframe():
    """DataFrame de ejemplo para pruebas."""
    return pd.DataFrame({
        'age': [30, 40, 50],
        'job': ['admin.', 'technician', 'management'],
        'marital': ['single', 'married', 'divorced'],
        'education': ['secondary', 'tertiary', 'primary'],
        'default': ['no', 'no', 'yes'],
        'balance': [1000, 2000, 3000],
        'housing': ['yes', 'no', 'yes'],
        'loan': ['no', 'yes', 'no'],
        'contact': ['cellular', 'unknown', 'telephone'],
        'day': [1, 15, 30],
        'month': ['may', 'jun', 'jul'],
        'duration': [100, 200, 300],
        'campaign': [1, 2, 3],
        'pdays': [-1, 5, 10],
        'previous': [0, 1, 2],
        'poutcome': ['unknown', 'success', 'failure'],
        'deposit': ['yes', 'no', 'yes']
    })

