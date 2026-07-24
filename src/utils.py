import logging
from pathlib import Path
from typing import Any

import yaml


def get_logger(name: str) -> logging.Logger:
    """
    Sets up and returns a logger with standard formatting.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        
        # Console handler
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        
    return logger

def load_config(config_path: str = "configs/config.yaml") -> dict[str, Any]:
    """
    Loads configuration from a YAML file.
    """
    logger = get_logger(__name__)
    try:
        with open(config_path, "r") as file:
            config = yaml.safe_load(file)
            logger.info(f"Loaded config from {config_path}")
            return config
    except Exception as e:
        logger.error(f"Failed to load config from {config_path}: {e}")
        raise

def ensure_dirs(dirs: list):
    """
    Ensures that given directories exist.
    """
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)
