"""
API de Inferencia para Detección y Clasificación de Desechos
Framework: FastAPI
Modelo: YOLOv8n
"""

import asyncio
import os
import time
import uuid
import logging
from datetime import datetime
from typing import Optional, List
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import numpy as np
from PIL import Image
import io

from model_manager import ModelManager
from database import DatabaseManager, InferenceRecord, ModelMetadata
from storage import StorageManager
from config import Settings

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuración global
settings = Settings()

# Managers globales
model_manager: Optional[ModelManager] = None
db_manager: Optional[DatabaseManager] = None
storage_manager: Optional[StorageManager] = None


# ============================================================================
# Schemas Pydantic 
# ============================================================================

class Detection(BaseModel):
    """Representa una detección individual"""
    class_name: str = Field(..., description="Nombre de la clase detectada")
    class_id: int = Field(..., description="ID de la clase")
    confidence: float = Field(..., ge=0, le=1, description="Confianza de la detección")
    bbox: List[float] = Field(..., description="Bounding box [x1, y1, x2, y2]")


class InferenceResponse(BaseModel):
    """Respuesta de inferencia"""
    request_id: str = Field(..., description="ID único de la petición")
    timestamp: str = Field(..., description="Timestamp de la inferencia")
    model_version: str = Field(..., description="Versión del modelo utilizado")
    inference_time_ms: float = Field(..., description="Tiempo de inferencia en milisegundos")
    image_size: List[int] = Field(..., description="Tamaño de la imagen [width, height]")
    detections: List[Detection] = Field(default=[], description="Lista de detecciones")
    detection_count: int = Field(..., description="Número total de detecciones")


class HealthResponse(BaseModel):
    """Respuesta del health check"""
    status: str
    timestamp: str
    model_loaded: bool
    model_version: str
    database_connected: bool
    storage_connected: bool


class ModelVersionInfo(BaseModel):
    """Información de versión del modelo"""
    version: str
    created_at: str
    accuracy: Optional[float]
    map50: Optional[float]
    is_active: bool


class TrainingRequest(BaseModel):
    """Request para iniciar entrenamiento"""
    epochs: int = Field(default=50, ge=1, le=500)
    batch_size: int = Field(default=16, ge=1, le=64)
    learning_rate: float = Field(default=0.01, ge=0.0001, le=0.1)
    dataset_path: Optional[str] = None


class TrainingResponse(BaseModel):
    """Respuesta de inicio de entrenamiento"""
    job_id: str
    status: str
    message: str


class MetricsResponse(BaseModel):
    """Métricas del servicio"""
    total_inferences: int
    avg_inference_time_ms: float
    requests_last_hour: int
    model_version: str
    uptime_seconds: float


# ============================================================================
# Lifespan Context Manager
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Gestiona el ciclo de vida de la aplicación"""
    global model_manager, db_manager, storage_manager
    
    logger.info("Iniciando aplicación...")
    
    # Inicializar Storage Manager (opcional)
    for attempt in range(5):
        try:
            if settings.gcs_models_bucket:
                storage_manager = StorageManager(
                    models_bucket=settings.gcs_models_bucket,
                    images_bucket=settings.gcs_images_bucket
                )
                logger.info("Storage Manager inicializado")
                break
        except Exception as e:
            logger.warning(f"Storage Manager intento {attempt + 1}/5 falló: {e}")
            storage_manager = None
        await asyncio.sleep(5)
    
    # Inicializar Database Manager (opcional)
    try:
        if settings.database_url and not settings.database_url.startswith("postgresql://user:pass@"):
            db_manager = DatabaseManager(settings.database_url)
            await db_manager.connect()
            logger.info("Database Manager conectado")
        else:
            logger.warning("Database URL no configurada - BD deshabilitada")
    except Exception as e:
        logger.warning(f"Database Manager no disponible: {e}")
        db_manager = None
    
    # Inicializar Model Manager (requerido)
    try:
        model_manager = ModelManager(
            storage_manager=storage_manager,
            model_version=settings.model_version,
            confidence_threshold=settings.confidence_threshold
        )
        await model_manager.load_model()
        logger.info(f"Modelo cargado: versión {model_manager.current_version}")
    except Exception as e:
        logger.error(f"Error cargando modelo: {e}")
        # Crear model manager sin modelo - funcionará en modo degradado
        model_manager = ModelManager(
            storage_manager=None,
            model_version="none",
            confidence_threshold=settings.confidence_threshold
        )
        logger.warning("API iniciada en modo degradado (sin modelo)")
    
    app.state.start_time = time.time()
    
    yield
    
    # Cleanup
    logger.info("Cerrando aplicación...")
    if db_manager:
        await db_manager.disconnect()


# ============================================================================
# Aplicación FastAPI
# ============================================================================

app = FastAPI(
    title="API de Detección de Desechos",
    description="Sistema distribuido para detección y clasificación de desechos usando YOLOv8",
    version="1.0.0",
    lifespan=lifespan
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/debug/storage-init", tags=["Debug"])
async def debug_storage_init():
    from config import Settings
    s = Settings()
    try:
        from storage import StorageManager
        sm = StorageManager(
            models_bucket=s.gcs_models_bucket,
            images_bucket=s.gcs_images_bucket
        )
        return {"success": True, "models_bucket": sm.models_bucket_name}
    except Exception as e:
        return {"error": f"{type(e).__name__}: {str(e)}"}

@app.get("/debug/storage", tags=["Debug"])
async def debug_storage():
    results = {}
    
    results["storage_manager_exists"] = storage_manager is not None
    
    if storage_manager:
        results["bucket_name"] = storage_manager.models_bucket_name
        try:
            from google.cloud import storage as gcs
            client = gcs.Client()
            bucket = client.bucket(storage_manager.models_bucket_name)
            exists = bucket.exists()
            results["bucket_exists"] = exists
        except Exception as e:
            results["error"] = f"{type(e).__name__}: {str(e)}"
    
    return results

@app.get("/debug/network", tags=["Debug"])
async def debug_network():
    import socket
    results = {}
    
    # ¿Puede conectar al DNS de Kubernetes?
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(3)
        s.connect(("10.2.0.10", 53))
        s.close()
        results["kube_dns_tcp"] = "reachable"
    except Exception as e:
        results["kube_dns_tcp"] = str(e)
    
    # ¿Qué tiene /etc/resolv.conf?
    try:
        with open("/etc/resolv.conf") as f:
            results["resolv_conf"] = f.read()
    except Exception as e:
        results["resolv_conf_error"] = str(e)
    
    # ¿Puede alcanzar storage.googleapis.com por IP?
    try:
        import urllib.request
        req = urllib.request.Request("https://142.250.80.128", headers={"Host": "storage.googleapis.com"})
        resp = urllib.request.urlopen(req, timeout=3)
        results["storage_api"] = "reachable"
    except Exception as e:
        results["storage_api_error"] = str(e)
    
    return results

@app.get("/debug/dns", tags=["Debug"])
async def debug_dns():
    import socket
    results = {}
    
    # ¿Resuelve DNS en general?
    try:
        results["google_dns"] = socket.getaddrinfo("google.com", 443)[0][4][0]
    except Exception as e:
        results["google_dns_error"] = str(e)
    
    # ¿Resuelve el metadata server?
    try:
        results["metadata_dns"] = socket.getaddrinfo("metadata.google.internal", 80)[0][4][0]
    except Exception as e:
        results["metadata_dns_error"] = str(e)
    
    # ¿Resuelve por IP directa?
    try:
        import urllib.request
        req = urllib.request.Request(
            "http://169.254.169.254/computeMetadata/v1/instance/service-accounts/default/email",
            headers={"Metadata-Flavor": "Google"}
        )
        resp = urllib.request.urlopen(req, timeout=3)
        results["metadata_by_ip"] = resp.read().decode()
    except Exception as e:
        results["metadata_by_ip_error"] = str(e)
    
    return results

@app.get("/debug/workload-identity", tags=["Debug"])
async def debug_workload_identity():
    """Diagnóstico temporal - BORRAR después de resolver"""
    import subprocess
    results = {}
    
    # 1. ¿Hay credenciales de GCP?
    try:
        from google.auth import default
        credentials, project = default()
        results["gcp_project"] = project
        results["credentials_type"] = type(credentials).__name__
    except Exception as e:
        results["credentials_error"] = str(e)
    
    # 2. ¿Puede alcanzar el metadata server?
    try:
        import urllib.request
        req = urllib.request.Request(
            "http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/email",
            headers={"Metadata-Flavor": "Google"}
        )
        resp = urllib.request.urlopen(req, timeout=3)
        results["service_account_email"] = resp.read().decode()
    except Exception as e:
        results["metadata_error"] = str(e)
    
    # 3. ¿Puede listar el bucket?
    try:
        from google.cloud import storage
        client = storage.Client()
        bucket = client.bucket("waste-detection-001-waste-detection-models-dev")
        blobs = list(client.list_blobs(bucket, prefix="models/", max_results=5))
        results["bucket_files"] = [b.name for b in blobs]
    except Exception as e:
        results["storage_error"] = str(e)
    
    return results


# ============================================================================
# Endpoints
# ============================================================================

@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """
    Health check del servicio.
    Verifica el estado del modelo, base de datos y storage.
    """
    db_connected = False
    storage_connected = False
    
    try:
        if db_manager:
            db_connected = await db_manager.health_check()
    except Exception:
        pass
    
    try:
        if storage_manager:
            storage_connected = await storage_manager.health_check()
    except Exception:
        pass
    
    return HealthResponse(
        status="healthy" if model_manager and model_manager.model else "degraded",
        timestamp=datetime.utcnow().isoformat(),
        model_loaded=model_manager is not None and model_manager.model is not None,
        model_version=model_manager.current_version if model_manager else "unknown",
        database_connected=db_connected,
        storage_connected=storage_connected
    )


@app.post("/predict", response_model=InferenceResponse, tags=["Inference"])
async def predict(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="Imagen para analizar"),
    save_image: bool = Query(True, description="Guardar imagen en GCS para reentrenamiento")
):
    """
    Realiza inferencia sobre una imagen.
    
    Acepta formatos: JPEG, PNG, WEBP
    Retorna las detecciones con clase, confianza y bounding boxes.
    """
    request_id = str(uuid.uuid4())
    start_time = time.time()
    
    # Validar tipo de archivo
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="El archivo debe ser una imagen (JPEG, PNG, WEBP)"
        )
    
    try:
        # Leer y procesar imagen
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        image_size = [image.width, image.height]
        
        # Convertir a RGB si es necesario
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        # Realizar inferencia
        detections = await model_manager.predict(image)
        
        inference_time = (time.time() - start_time) * 1000  # ms
        
        # Crear respuesta
        detection_list = [
            Detection(
                class_name=d["class_name"],
                class_id=d["class_id"],
                confidence=d["confidence"],
                bbox=d["bbox"]
            )
            for d in detections
        ]
        
        response = InferenceResponse(
            request_id=request_id,
            timestamp=datetime.utcnow().isoformat(),
            model_version=model_manager.current_version,
            inference_time_ms=round(inference_time, 2),
            image_size=image_size,
            detections=detection_list,
            detection_count=len(detection_list)
        )
        
        # Guardar en background
        background_tasks.add_task(
            save_inference_record,
            request_id=request_id,
            response=response,
            image_bytes=contents if save_image else None
        )
        
        logger.info(f"Inferencia {request_id}: {len(detections)} detecciones en {inference_time:.2f}ms")
        
        return response
        
    except Exception as e:
        logger.error(f"Error en inferencia {request_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/models", response_model=List[ModelVersionInfo], tags=["Models"])
async def list_models():
    """
    Lista todas las versiones de modelos disponibles.
    """
    if not db_manager:
        raise HTTPException(status_code=503, detail="Database not available")
    
    try:
        models = await db_manager.get_model_versions()
        return [
            ModelVersionInfo(
                version=m.version,
                created_at=m.created_at.isoformat(),
                accuracy=m.accuracy,
                map50=m.map50,
                is_active=m.version == model_manager.current_version
            )
            for m in models
        ]
    except Exception as e:
        logger.error(f"Error listando modelos: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/models/{version}/activate", tags=["Models"])
async def activate_model(version: str):
    """
    Activa una versión específica del modelo.
    """
    if not model_manager:
        raise HTTPException(status_code=503, detail="Model manager not available")
    
    try:
        await model_manager.load_model(version=version)
        logger.info(f"Modelo activado: {version}")
        return {"status": "success", "active_version": version}
    except Exception as e:
        logger.error(f"Error activando modelo {version}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/training/start", response_model=TrainingResponse, tags=["Training"])
async def start_training(request: TrainingRequest):
    """
    Inicia un job de entrenamiento.
    
    Este endpoint dispara un Job de Kubernetes para entrenar el modelo.
    """
    job_id = f"training-{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}"
    
    try:
        # Aquí se integraría con la API de Kubernetes para crear el Job
        # Por ahora retornamos un placeholder
        logger.info(f"Entrenamiento solicitado: {job_id}")
        
        return TrainingResponse(
            job_id=job_id,
            status="queued",
            message=f"Entrenamiento iniciado con {request.epochs} epochs"
        )
    except Exception as e:
        logger.error(f"Error iniciando entrenamiento: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics", response_model=MetricsResponse, tags=["Monitoring"])
async def get_metrics():
    """
    Retorna métricas del servicio.
    """
    if not db_manager:
        raise HTTPException(status_code=503, detail="Database not available")
    
    try:
        stats = await db_manager.get_inference_stats()
        uptime = time.time() - app.state.start_time
        
        return MetricsResponse(
            total_inferences=stats.get("total", 0),
            avg_inference_time_ms=stats.get("avg_time_ms", 0),
            requests_last_hour=stats.get("last_hour", 0),
            model_version=model_manager.current_version if model_manager else "unknown",
            uptime_seconds=round(uptime, 2)
        )
    except Exception as e:
        logger.error(f"Error obteniendo métricas: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/", tags=["Root"])
async def root():
    """Endpoint raíz con información del servicio"""
    return {
        "service": "Waste Detection API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/training/export", tags=["Training"])
async def export_training_data(
    limit: int = Query(1000, description="Número máximo de registros"),
    min_detections: int = Query(1, description="Mínimo de detecciones por imagen")
):
    """
    Exporta datos de inferencias para reentrenamiento.
    
    Retorna las inferencias con sus URLs de imagen y anotaciones,
    filtradas por número mínimo de detecciones.
    
    Los archivos están organizados en GCS:
    - Imágenes: gs://bucket/inferences/YYYY/MM/DD/{request_id}.jpeg
    - Anotaciones: gs://bucket/inferences/YYYY/MM/DD/{request_id}.txt
    
    El formato de las anotaciones es YOLO:
    class_id x_center y_center width height (normalizado 0-1)
    """
    if not db_manager:
        raise HTTPException(status_code=503, detail="Database not available")
    
    try:
        # Obtener inferencias con detecciones
        inferences = await db_manager.get_recent_inferences(limit=limit)
        
        # Filtrar por número mínimo de detecciones
        filtered = [
            {
                "request_id": inf.request_id,
                "timestamp": inf.timestamp.isoformat(),
                "image_url": inf.image_url,
                "annotations_url": inf.image_url.replace(".jpeg", ".txt").replace(".jpg", ".txt").replace(".png", ".txt") if inf.image_url else None,
                "detection_count": inf.detection_count,
                "detections": inf.detections,
                "model_version": inf.model_version
            }
            for inf in inferences
            if inf.detection_count >= min_detections and inf.image_url
        ]
        
        return {
            "total_records": len(filtered),
            "min_detections_filter": min_detections,
            "data": filtered
        }
        
    except Exception as e:
        logger.error(f"Error exportando datos de entrenamiento: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/inferences/{request_id}", tags=["Inference"])
async def get_inference(request_id: str):
    """
    Obtiene los detalles de una inferencia específica por su ID.
    """
    if not db_manager:
        raise HTTPException(status_code=503, detail="Database not available")
    
    try:
        record = await db_manager.get_inference(request_id)
        
        if not record:
            raise HTTPException(status_code=404, detail="Inferencia no encontrada")
        
        return {
            "request_id": record.request_id,
            "timestamp": record.timestamp.isoformat(),
            "model_version": record.model_version,
            "inference_time_ms": record.inference_time_ms,
            "detection_count": record.detection_count,
            "detections": record.detections,
            "image_url": record.image_url,
            "annotations_url": record.image_url.replace(".jpeg", ".txt").replace(".jpg", ".txt").replace(".png", ".txt") if record.image_url else None
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error obteniendo inferencia {request_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Funciones auxiliares
# ============================================================================

def generate_yolo_annotations(detections: List[Detection], img_width: int, img_height: int) -> str:
    """
    Convierte detecciones al formato de anotación YOLO.
    
    Formato YOLO: class_id x_center y_center width height
    (todos los valores normalizados entre 0 y 1)
    
    Args:
        detections: Lista de detecciones
        img_width: Ancho de la imagen
        img_height: Alto de la imagen
        
    Returns:
        String con las anotaciones en formato YOLO (una línea por detección)
    """
    lines = []
    
    for det in detections:
        # bbox está en formato [x1, y1, x2, y2]
        x1, y1, x2, y2 = det.bbox
        
        # Calcular centro y dimensiones
        x_center = (x1 + x2) / 2 / img_width
        y_center = (y1 + y2) / 2 / img_height
        width = (x2 - x1) / img_width
        height = (y2 - y1) / img_height
        
        # Asegurar que los valores estén entre 0 y 1
        x_center = max(0, min(1, x_center))
        y_center = max(0, min(1, y_center))
        width = max(0, min(1, width))
        height = max(0, min(1, height))
        
        # Formato: class_id x_center y_center width height
        line = f"{det.class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
        lines.append(line)
    
    return "\n".join(lines)


async def save_inference_record(
    request_id: str,
    response: InferenceResponse,
    image_bytes: Optional[bytes] = None
):
    """Guarda el registro de inferencia en background"""
    try:
        image_url = None
        annotations_url = None
        
        # Guardar imagen y anotaciones si se proporciona la imagen
        if image_bytes and storage_manager:
            # Guardar imagen
            image_url = await storage_manager.upload_image(
                request_id, 
                image_bytes
            )
            
            # Generar y guardar anotaciones en formato YOLO
            if response.detections:
                yolo_annotations = generate_yolo_annotations(
                    response.detections,
                    response.image_size[0],  # width
                    response.image_size[1]   # height
                )
                annotations_url = await storage_manager.upload_annotations(
                    request_id,
                    yolo_annotations
                )
                logger.info(f"Anotaciones guardadas: {annotations_url}")
        
        # Guardar registro en BD
        if db_manager:
            record = InferenceRecord(
                request_id=request_id,
                timestamp=datetime.fromisoformat(response.timestamp),
                model_version=response.model_version,
                inference_time_ms=response.inference_time_ms,
                detection_count=response.detection_count,
                detections=[d.model_dump() for d in response.detections],
                image_url=image_url
            )
            await db_manager.save_inference(record)
            logger.info(f"Registro guardado en BD: {request_id}")
            
            
    except Exception as e:
        logger.error(f"Error guardando registro {request_id}: {e}")


# ============================================================================
# Ejecución
# ============================================================================

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.debug,
        workers=1
    )
