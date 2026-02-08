"""
Storage Manager - Gestión de Google Cloud Storage
"""

import logging
import asyncio
from typing import Optional
from pathlib import Path
from datetime import datetime

from google.cloud import storage
from google.cloud.exceptions import NotFound

logger = logging.getLogger(__name__)


class StorageManager:
    """
    Gestiona las operaciones con Google Cloud Storage.
    """
    
    def __init__(
        self,
        models_bucket: str,
        images_bucket: str,
        datasets_bucket: Optional[str] = None
    ):
        self.models_bucket_name = models_bucket
        self.images_bucket_name = images_bucket
        self.datasets_bucket_name = datasets_bucket
        
        # Inicializar cliente de GCS
        self.client = storage.Client()
        
        # Referencias a buckets
        self.models_bucket = self.client.bucket(models_bucket)
        self.images_bucket = self.client.bucket(images_bucket)
        self.datasets_bucket = (
            self.client.bucket(datasets_bucket) if datasets_bucket else None
        )
    
    async def health_check(self) -> bool:
        """Verifica conexión con GCS"""
        try:
            loop = asyncio.get_event_loop()
            client = storage.Client()
            bucket = client.bucket(self.models_bucket_name)
            exists = await loop.run_in_executor(
                None,
                bucket.exists
            )
            return exists
        except Exception:
            return False
    
    # ========================================================================
    # Operaciones de Modelos
    # ========================================================================
    
    async def download_file(self, gcs_path: str, local_path: str) -> None:
        """
        Descarga un archivo desde GCS.
        
        Args:
            gcs_path: Ruta en GCS (relativa al bucket de modelos)
            local_path: Ruta local donde guardar el archivo
        """
        loop = asyncio.get_event_loop()
        
        blob = self.models_bucket.blob(gcs_path)
        
        # Verificar existencia
        exists = await loop.run_in_executor(None, blob.exists)
        if not exists:
            raise FileNotFoundError(f"Archivo no encontrado en GCS: {gcs_path}")
        
        # Crear directorio local si no existe
        Path(local_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Descargar
        await loop.run_in_executor(
            None,
            blob.download_to_filename,
            local_path
        )
        
        logger.info(f"Archivo descargado: gs://{self.models_bucket_name}/{gcs_path}")
    
    async def upload_model(
        self,
        local_path: str,
        version: str,
        metadata: Optional[dict] = None
    ) -> str:
        """
        Sube un modelo entrenado a GCS.
        
        Args:
            local_path: Ruta local del modelo
            version: Versión del modelo (ej: v20250129_143000)
            metadata: Metadata adicional
            
        Returns:
            Ruta GCS del modelo subido
        """
        loop = asyncio.get_event_loop()
        
        gcs_path = f"models/{version}/yolov8n_waste.pt"
        blob = self.models_bucket.blob(gcs_path)
        
        # Agregar metadata
        if metadata:
            blob.metadata = metadata
        
        # Subir archivo
        await loop.run_in_executor(
            None,
            blob.upload_from_filename,
            local_path
        )
        
        logger.info(f"Modelo subido: gs://{self.models_bucket_name}/{gcs_path}")
        
        return f"gs://{self.models_bucket_name}/{gcs_path}"
    
    async def upload_model_metadata(
        self,
        version: str,
        metadata: dict
    ) -> str:
        """
        Sube metadata del modelo como JSON.
        
        Args:
            version: Versión del modelo
            metadata: Diccionario con metadata
            
        Returns:
            Ruta GCS del archivo de metadata
        """
        import json
        
        loop = asyncio.get_event_loop()
        
        gcs_path = f"models/{version}/metadata.json"
        blob = self.models_bucket.blob(gcs_path)
        
        # Subir JSON
        await loop.run_in_executor(
            None,
            blob.upload_from_string,
            json.dumps(metadata, indent=2, default=str),
            "application/json"
        )
        
        return f"gs://{self.models_bucket_name}/{gcs_path}"
    
    async def list_model_versions(self) -> list:
        """Lista todas las versiones de modelos disponibles"""
        loop = asyncio.get_event_loop()
        
        blobs = await loop.run_in_executor(
            None,
            lambda: list(self.client.list_blobs(
                self.models_bucket_name,
                prefix="models/",
                delimiter="/"
            ))
        )
        
        versions = set()
        for blob in blobs:
            parts = blob.name.split("/")
            if len(parts) >= 2 and parts[1]:
                versions.add(parts[1])
        
        return sorted(versions, reverse=True)
    
    async def get_latest_model_version(self) -> Optional[str]:
        """Obtiene la versión más reciente del modelo"""
        versions = await self.list_model_versions()
        return versions[0] if versions else None
    
    # ========================================================================
    # Operaciones de Imágenes
    # ========================================================================
    
    async def upload_image(
        self,
        request_id: str,
        image_bytes: bytes,
        content_type: str = "image/jpeg"
    ) -> str:
        """
        Sube una imagen de inferencia a GCS.
        
        Args:
            request_id: ID de la petición
            image_bytes: Bytes de la imagen
            content_type: Tipo MIME de la imagen
            
        Returns:
            URL de la imagen en GCS
        """
        loop = asyncio.get_event_loop()
        
        # Organizar por fecha
        date_prefix = datetime.utcnow().strftime("%Y/%m/%d")
        extension = content_type.split("/")[-1]
        gcs_path = f"inferences/{date_prefix}/{request_id}.{extension}"
        
        blob = self.images_bucket.blob(gcs_path)
        blob.content_type = content_type
        
        await loop.run_in_executor(
            None,
            blob.upload_from_string,
            image_bytes,
            content_type
        )
        
        logger.debug(f"Imagen subida: gs://{self.images_bucket_name}/{gcs_path}")
        
        return f"gs://{self.images_bucket_name}/{gcs_path}"
    
    async def upload_annotations(
        self,
        request_id: str,
        annotations: str
    ) -> str:
        """
        Sube las anotaciones YOLO a GCS.
        
        Args:
            request_id: ID de la petición (mismo que la imagen)
            annotations: String con las anotaciones en formato YOLO
            
        Returns:
            URL del archivo de anotaciones en GCS
        """
        loop = asyncio.get_event_loop()
        
        # Organizar por fecha (mismo path que la imagen)
        date_prefix = datetime.utcnow().strftime("%Y/%m/%d")
        gcs_path = f"inferences/{date_prefix}/{request_id}.txt"
        
        blob = self.images_bucket.blob(gcs_path)
        blob.content_type = "text/plain"
        
        await loop.run_in_executor(
            None,
            blob.upload_from_string,
            annotations,
            "text/plain"
        )
        
        logger.debug(f"Anotaciones subidas: gs://{self.images_bucket_name}/{gcs_path}")
        
        return f"gs://{self.images_bucket_name}/{gcs_path}"
    
    async def download_image(self, gcs_url: str) -> bytes:
        """Descarga una imagen desde GCS"""
        loop = asyncio.get_event_loop()
        
        # Parsear URL
        if gcs_url.startswith("gs://"):
            path = gcs_url[5:]  # Remover "gs://"
            bucket_name, blob_path = path.split("/", 1)
            bucket = self.client.bucket(bucket_name)
        else:
            bucket = self.images_bucket
            blob_path = gcs_url
        
        blob = bucket.blob(blob_path)
        
        return await loop.run_in_executor(
            None,
            blob.download_as_bytes
        )
    
    # ========================================================================
    # Operaciones de Datasets
    # ========================================================================
    
    async def download_dataset(self, dataset_name: str, local_dir: str) -> str:
        """
        Descarga un dataset completo desde GCS.
        
        Args:
            dataset_name: Nombre del dataset
            local_dir: Directorio local de destino
            
        Returns:
            Ruta local del dataset descargado
        """
        if not self.datasets_bucket:
            raise ValueError("Bucket de datasets no configurado")
        
        loop = asyncio.get_event_loop()
        local_path = Path(local_dir) / dataset_name
        local_path.mkdir(parents=True, exist_ok=True)
        
        prefix = f"datasets/{dataset_name}/"
        blobs = await loop.run_in_executor(
            None,
            lambda: list(self.client.list_blobs(
                self.datasets_bucket_name,
                prefix=prefix
            ))
        )
        
        for blob in blobs:
            relative_path = blob.name[len(prefix):]
            if not relative_path:
                continue
                
            local_file = local_path / relative_path
            local_file.parent.mkdir(parents=True, exist_ok=True)
            
            await loop.run_in_executor(
                None,
                blob.download_to_filename,
                str(local_file)
            )
        
        logger.info(f"Dataset descargado: {dataset_name} -> {local_path}")
        return str(local_path)
    
    async def upload_training_data(
        self,
        local_path: str,
        dataset_name: str
    ) -> str:
        """
        Sube datos de entrenamiento a GCS.
        
        Args:
            local_path: Ruta local de los datos
            dataset_name: Nombre del dataset
            
        Returns:
            Ruta GCS del dataset
        """
        if not self.datasets_bucket:
            raise ValueError("Bucket de datasets no configurado")
        
        loop = asyncio.get_event_loop()
        local_path = Path(local_path)
        
        if local_path.is_file():
            # Subir archivo único
            gcs_path = f"datasets/{dataset_name}/{local_path.name}"
            blob = self.datasets_bucket.blob(gcs_path)
            await loop.run_in_executor(
                None,
                blob.upload_from_filename,
                str(local_path)
            )
        else:
            # Subir directorio completo
            for file_path in local_path.rglob("*"):
                if file_path.is_file():
                    relative = file_path.relative_to(local_path)
                    gcs_path = f"datasets/{dataset_name}/{relative}"
                    blob = self.datasets_bucket.blob(gcs_path)
                    await loop.run_in_executor(
                        None,
                        blob.upload_from_filename,
                        str(file_path)
                    )
        
        return f"gs://{self.datasets_bucket_name}/datasets/{dataset_name}"