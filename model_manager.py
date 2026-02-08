"""
Model Manager - Gestión del modelo YOLOv8 para detección de desechos
"""

import os
import logging
import asyncio
from typing import Optional, List, Dict, Any
from pathlib import Path
from datetime import datetime

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# Clases de desechos que el modelo puede detectar
WASTE_CLASSES = {
    0: "biodegradable",
    1: "cardboard",
    2: "glass",
    3: "metal",
    4: "paper",
    5: "plastic",
}


class ModelManager:
    """
    Gestiona la carga, versionado y predicción del modelo YOLOv8.
    """
    
    def __init__(
        self,
        storage_manager,
        model_version: str = "latest",
        confidence_threshold: float = 0.5,
        iou_threshold: float = 0.45,
        local_cache_dir: str = "/app/models"
    ):
        self.storage_manager = storage_manager
        self.requested_version = model_version
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.local_cache_dir = Path(local_cache_dir)
        
        self.model = None
        self.current_version: str = "unknown"
        self.model_path: Optional[Path] = None
        
        # Crear directorio de cache
        self.local_cache_dir.mkdir(parents=True, exist_ok=True)
    
    async def load_model(self, version: Optional[str] = None) -> None:
        """
        Carga el modelo desde GCS o cache local.
        
        Args:
            version: Versión específica a cargar. Si es None, usa la configurada.
        """
        target_version = version or self.requested_version
        
        try:
            # Intentar descargar desde GCS
            if self.storage_manager:
                model_path = await self._download_model(target_version)
            else:
                # Fallback a modelo local o preentrenado
                model_path = await self._get_local_model(target_version)
            
            # Cargar modelo con ultralytics
            await self._load_yolo_model(model_path)
            
            self.current_version = target_version
            self.model_path = model_path
            
            logger.info(f"Modelo cargado exitosamente: {self.current_version}")
            
        except Exception as e:
            logger.error(f"Error cargando modelo {target_version}: {e}")
            # Intentar cargar modelo por defecto
            await self._load_default_model()
    
    async def _download_model(self, version: str) -> Path:
        """Descarga el modelo desde GCS"""
        local_path = self.local_cache_dir / f"yolov8n_waste_{version}.pt"
        
        # Verificar cache local
        if local_path.exists():
            logger.info(f"Modelo encontrado en cache: {local_path}")
            return local_path
        
        # Descargar desde GCS
        gcs_path = f"models/{version}/yolov8n_waste.pt"
        
        try:
            await self.storage_manager.download_file(
                gcs_path, 
                str(local_path)
            )
            logger.info(f"Modelo descargado: {gcs_path} -> {local_path}")
            return local_path
            
        except Exception as e:
            logger.warning(f"No se pudo descargar modelo {version}: {e}")
            raise
    
    async def _get_local_model(self, version: str) -> Path:
        """Busca modelo en directorio local"""
        patterns = [
            self.local_cache_dir / f"yolov8n_waste_{version}.pt",
            self.local_cache_dir / f"yolov8n_waste.pt",
            self.local_cache_dir / "best.pt",
        ]
        
        for path in patterns:
            if path.exists():
                return path
        
        # Si no hay modelo local, usar preentrenado
        return Path("yolov8n.pt")
    
    async def _load_yolo_model(self, model_path: Path) -> None:
        """Carga el modelo YOLO"""
        from ultralytics import YOLO
        
        # Cargar en thread separado para no bloquear
        loop = asyncio.get_event_loop()
        self.model = await loop.run_in_executor(
            None,
            lambda: YOLO(str(model_path))
        )
        
        # Warmup del modelo
        logger.info("Realizando warmup del modelo...")
        dummy_image = np.zeros((640, 640, 3), dtype=np.uint8)
        await loop.run_in_executor(
            None,
            lambda: self.model(dummy_image, verbose=False)
        )
    
    async def _load_default_model(self) -> None:
        """Carga modelo preentrenado por defecto"""
        logger.warning("Cargando modelo YOLOv8n preentrenado por defecto")
        
        from ultralytics import YOLO
        
        loop = asyncio.get_event_loop()
        self.model = await loop.run_in_executor(
            None,
            lambda: YOLO("yolov8n.pt")
        )
        self.current_version = "yolov8n-pretrained"
    
    async def predict(self, image: Image.Image) -> List[Dict[str, Any]]:
        """
        Realiza predicción sobre una imagen.
        
        Args:
            image: Imagen PIL para analizar
            
        Returns:
            Lista de detecciones con clase, confianza y bbox
        """
        if self.model is None:
            raise RuntimeError("Modelo no cargado")
        
        # Convertir PIL a numpy
        image_np = np.array(image)
        
        # Ejecutar inferencia en thread pool
        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(
            None,
            lambda: self.model(
                image_np,
                conf=self.confidence_threshold,
                iou=self.iou_threshold,
                verbose=False
            )
        )
        
        # Procesar resultados
        detections = []
        
        for result in results:
            if result.boxes is None:
                continue
                
            boxes = result.boxes
            
            for i in range(len(boxes)):
                box = boxes[i]
                
                # Obtener clase y confianza
                class_id = int(box.cls[0].item())
                confidence = float(box.conf[0].item())
                
                # Obtener bbox en formato [x1, y1, x2, y2]
                bbox = box.xyxy[0].tolist()
                bbox = [round(coord, 2) for coord in bbox]
                
                # Mapear clase
                if class_id in WASTE_CLASSES:
                    class_name = WASTE_CLASSES[class_id]
                elif hasattr(self.model, 'names') and class_id in self.model.names:
                    class_name = self.model.names[class_id]
                else:
                    class_name = f"class_{class_id}"
                
                detections.append({
                    "class_name": class_name,
                    "class_id": class_id,
                    "confidence": round(confidence, 4),
                    "bbox": bbox
                })
        
        # Ordenar por confianza descendente
        detections.sort(key=lambda x: x["confidence"], reverse=True)
        
        return detections
    
    def get_model_info(self) -> Dict[str, Any]:
        """Retorna información del modelo cargado"""
        info = {
            "version": self.current_version,
            "loaded": self.model is not None,
            "confidence_threshold": self.confidence_threshold,
            "iou_threshold": self.iou_threshold,
        }
        
        if self.model and hasattr(self.model, 'names'):
            info["classes"] = list(self.model.names.values())
            info["num_classes"] = len(self.model.names)
        
        if self.model_path:
            info["model_path"] = str(self.model_path)
        
        return info
