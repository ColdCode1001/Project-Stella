"""
Logger compartido para todos los scripts de entrenamiento de Stella WM.
Escribe a consola Y a archivo de log automáticamente.
Uso: from worldmodel.training_logger import setup_logger, log
"""

import logging
import sys
from datetime import datetime
from pathlib import Path

_logger: logging.Logger | None = None
_log_path: Path | None = None


def setup_logger(name: str) -> logging.Logger:
    """
    Configura el logger para un script de entrenamiento.
    Crea logs/<name>_YYYYMMDD_HHMMSS.log automáticamente.
    Llama una vez al inicio del script.
    """
    global _logger, _log_path

    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    _log_path = logs_dir / f"{name}_{timestamp}.log"

    _logger = logging.getLogger(name)
    _logger.setLevel(logging.DEBUG)
    _logger.handlers.clear()

    # Consola
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(logging.Formatter("%(message)s"))
    _logger.addHandler(ch)

    # Archivo
    fh = logging.FileHandler(_log_path, encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s  %(message)s", datefmt="%H:%M:%S"))
    _logger.addHandler(fh)

    _logger.info(f"Log: {_log_path}")
    return _logger


def get_logger() -> logging.Logger:
    global _logger
    if _logger is None:
        _logger = setup_logger("stella_wm")
    return _logger


def log(msg: str):
    get_logger().info(msg)
