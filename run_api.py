"""Script para ejecutar la API FastAPI."""

import os
import uvicorn
from src.utils.config import API_HOST, API_PORT

if __name__ == "__main__":
    # Railway asigna el puerto dinámicamente a través de $PORT
    # En desarrollo local, usar API_PORT por defecto
    port = int(os.getenv("PORT", API_PORT))
    
    # Desactivar reload en producción (cuando PORT está definido por Railway)
    # Activar reload solo en desarrollo local
    reload_mode = os.getenv("PORT") is None and os.getenv("DEBUG", "false").lower() == "true"
    
    uvicorn.run(
        "src.api.main:app",
        host=API_HOST,
        port=port,
        reload=reload_mode
    )

