"""Script para ejecutar la API FastAPI."""

import uvicorn
from src.utils.config import API_HOST, API_PORT

if __name__ == "__main__":
    uvicorn.run(
        "src.api.main:app",
        host=API_HOST,
        port=API_PORT,
        reload=True
    )

