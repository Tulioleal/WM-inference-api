"""
Configuration - Variables de entorno y configuración del sistema
"""

import os
from typing import Optional
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """
    Configuración de la aplicación cargada desde variables de entorno.
    """
    
    # Aplicación
    app_name: str = "Waste Detection API"
    app_version: str = "1.0.0"
    debug: bool = False
    log_level: str = "INFO"
    
    # Servidor
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1
    
    # Base de datos
    database_url: str
    db_pool_min: int = 2
    db_pool_max: int = 10
    
    # Google Cloud Storage
    gcs_models_bucket: str
    gcs_images_bucket: str
    
    # Modelo
    model_version: str = "latest"
    confidence_threshold: float = 0.5
    iou_threshold: float = 0.45
    model_cache_dir: str = "/app/models"
    
    # Límites
    max_image_size_mb: int = 10
    max_batch_size: int = 32
    request_timeout_seconds: int = 30
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


# Instancia global de configuración
settings = Settings()
