"""Script para ejecutar el dashboard Dash."""

import os
from src.dashboard.app import app
from src.utils.config import DASHBOARD_HOST, DASHBOARD_PORT

if __name__ == "__main__":
    # Railway asigna el puerto dinámicamente a través de $PORT
    # En desarrollo local, usar DASHBOARD_PORT por defecto
    port = int(os.getenv("PORT", DASHBOARD_PORT))
    
    # Desactivar debug en producción (cuando PORT está definido por Railway)
    # Activar debug solo en desarrollo local
    debug_mode = os.getenv("PORT") is None and os.getenv("DEBUG", "false").lower() == "true"
    
    app.run_server(
        host=DASHBOARD_HOST,
        port=port,
        debug=debug_mode
    )

