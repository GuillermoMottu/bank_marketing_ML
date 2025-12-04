#!/usr/bin/env python3
"""
Script de entrada que detecta qué servicio ejecutar (API o Dashboard)
basándose en la variable de entorno SERVICE_TYPE.
"""

import os
import sys

def main():
    """Ejecuta el servicio apropiado basándose en SERVICE_TYPE."""
    service_type = os.getenv("SERVICE_TYPE", "api").lower()
    
    if service_type == "dashboard":
        print("🚀 Iniciando Dashboard Dash...")
        # Ejecutar dashboard
        from run_dashboard import app
        from src.utils.config import DASHBOARD_HOST, DASHBOARD_PORT
        
        port = int(os.getenv("PORT", DASHBOARD_PORT))
        debug_mode = os.getenv("PORT") is None and os.getenv("DEBUG", "false").lower() == "true"
        
        app.run_server(
            host=DASHBOARD_HOST,
            port=port,
            debug=debug_mode
        )
    else:
        # Por defecto, ejecutar API
        print("🚀 Iniciando API FastAPI...")
        import uvicorn
        from src.utils.config import API_HOST, API_PORT
        
        port = int(os.getenv("PORT", API_PORT))
        reload_mode = os.getenv("PORT") is None and os.getenv("DEBUG", "false").lower() == "true"
        
        uvicorn.run(
            "src.api.main:app",
            host=API_HOST,
            port=port,
            reload=reload_mode
        )

if __name__ == "__main__":
    main()

