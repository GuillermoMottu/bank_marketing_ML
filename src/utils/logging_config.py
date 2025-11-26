"""Configuración de logging centralizada."""

import logging
import sys
from pathlib import Path

# Crear directorio de logs si no existe
LOG_DIR = Path(__file__).parent.parent.parent / "logs"
LOG_DIR.mkdir(exist_ok=True)


def setup_logging(log_level: str = "INFO", log_to_file: bool = True) -> logging.Logger:
    """Configura el sistema de logging del proyecto.
    
    Args:
        log_level: Nivel de logging (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_to_file: Si es True, guarda los logs en un archivo
    
    Returns:
        Logger configurado
    """
    logger = logging.getLogger("bank_marketing")
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Evitar duplicados
    if logger.handlers:
        return logger
    
    # Formato de los logs
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # Handler para consola
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Handler para archivo (si está habilitado)
    if log_to_file:
        file_handler = logging.FileHandler(LOG_DIR / "bank_marketing.log")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


# Logger global
logger = setup_logging()

