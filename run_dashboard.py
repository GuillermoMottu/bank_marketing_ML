"""Script para ejecutar el dashboard Dash."""

from src.dashboard.app import app
from src.utils.config import DASHBOARD_HOST, DASHBOARD_PORT

if __name__ == "__main__":
    app.run_server(
        host=DASHBOARD_HOST,
        port=DASHBOARD_PORT,
        debug=True
    )

