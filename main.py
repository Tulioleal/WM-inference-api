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
from typing import Optional, List, Dict
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel, Field
import numpy as np
from PIL import Image
import io

from model_manager import ModelManager, WASTE_CLASSES
from database import DatabaseManager, InferenceRecord, ModelMetadata, VerificationStatus
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


# --- Schemas de Verificación ---

class VerificationUpdate(BaseModel):
    """Request para actualizar verificación"""
    status: str = Field(
        ..., 
        description="Estado: verified, corrected, needs_review, discarded"
    )
    verified_detections: Optional[List[Detection]] = Field(
        None,
        description="Detecciones corregidas (requerido si status='corrected')"
    )
    verified_by: str = Field(
        default="anonymous",
        description="Identificador del revisor"
    )


class UserFeedback(BaseModel):
    """Feedback del usuario sobre una inferencia"""
    is_correct: bool = Field(..., description="¿La inferencia es correcta?")
    user_id: str = Field(default="anonymous", description="ID del usuario")


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


# ============================================================================
# Endpoints principales
# ============================================================================

@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check del servicio."""
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
        
        inference_time = (time.time() - start_time) * 1000
        
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
    """Lista todas las versiones de modelos disponibles."""
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
    """Activa una versión específica del modelo."""
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
    """Inicia un job de entrenamiento."""
    job_id = f"training-{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}"
    
    try:
        # TODO
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
    """Retorna métricas del servicio."""
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
    """Endpoint raíz"""
    return {
        "service": "Waste Detection API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/inferences/{request_id}", tags=["Inference"])
async def get_inference(request_id: str):
    """Obtiene los detalles de una inferencia específica."""
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
            "annotations_url": record.image_url.replace(".jpeg", ".txt").replace(".jpg", ".txt").replace(".png", ".txt") if record.image_url else None,
            "verification_status": record.verification_status,
            "verified_detections": record.verified_detections,
            "verified_by": record.verified_by,
            "verified_at": record.verified_at.isoformat() if record.verified_at else None,
            "min_confidence": record.min_confidence,
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error obteniendo inferencia {request_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/images/{request_id}", tags=["Images"])
async def get_image(request_id: str):
    """Sirve la imagen de una inferencia desde GCS."""
    if not db_manager or not storage_manager:
        raise HTTPException(status_code=503, detail="Service not available")
    
    record = await db_manager.get_inference(request_id)
    if not record or not record.image_url:
        raise HTTPException(status_code=404, detail="Image not found")
    
    try:
        image_bytes = await storage_manager.download_image(record.image_url)
        
        # Detectar content type por extensión
        content_type = "image/jpeg"
        if record.image_url.endswith(".png"):
            content_type = "image/png"
        elif record.image_url.endswith(".webp"):
            content_type = "image/webp"
        
        return Response(content=image_bytes, media_type=content_type)
    except Exception as e:
        logger.error(f"Error sirviendo imagen {request_id}: {e}")
        raise HTTPException(status_code=500, detail="Error retrieving image")

# ============================================================================
# Endpoints de Verificación / Review
# ============================================================================

@app.get("/review/queue", tags=["Verification"])
async def get_review_queue(
    status: Optional[str] = Query(None, description="Filtrar por estado: pending, needs_review, verified, corrected, discarded"),
    limit: int = Query(20, ge=1, le=100, description="Cantidad de registros"),
    offset: int = Query(0, ge=0, description="Offset para paginación")
):
    """
    Obtiene la cola de imágenes pendientes de verificación.
    
    Prioriza imágenes con needs_review (baja confianza o reportadas),
    seguidas de pending (sin revisar).
    """
    if not db_manager:
        raise HTTPException(status_code=503, detail="Database not available")
    
    try:
        result = await db_manager.get_review_queue(
            status_filter=status,
            limit=limit,
            offset=offset
        )
        
        # Serializar records
        records_data = []
        for r in result["records"]:
            records_data.append({
                "request_id": r.request_id,
                "timestamp": r.timestamp.isoformat(),
                "model_version": r.model_version,
                "inference_time_ms": r.inference_time_ms,
                "detection_count": r.detection_count,
                "detections": r.detections,
                "image_url": r.image_url,
                "verification_status": r.verification_status,
                "verified_detections": r.verified_detections,
                "verified_by": r.verified_by,
                "verified_at": r.verified_at.isoformat() if r.verified_at else None,
                "min_confidence": r.min_confidence,
            })
        
        return {
            "records": records_data,
            "status_counts": result["status_counts"],
            "total_pending": result["total_pending"],
            "limit": result["limit"],
            "offset": result["offset"],
        }
    except Exception as e:
        logger.error(f"Error obteniendo cola de revisión: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.put("/review/{request_id}", tags=["Verification"])
async def update_verification(request_id: str, update: VerificationUpdate):
    """
    Actualiza el estado de verificación de una inferencia.
    
    Estados posibles:
    - **verified**: La inferencia del modelo es correcta
    - **corrected**: Se corrigieron las etiquetas (enviar verified_detections)
    - **needs_review**: Marcar para revisión posterior
    - **discarded**: Imagen no útil para entrenamiento
    """
    if not db_manager:
        raise HTTPException(status_code=503, detail="Database not available")
    
    try:
        # Validar que si es corrección, tenga detecciones
        if update.status == VerificationStatus.CORRECTED and not update.verified_detections:
            raise HTTPException(
                status_code=400,
                detail="verified_detections es requerido cuando status='corrected'"
            )
        
        verified_dets = None
        if update.verified_detections:
            verified_dets = [d.model_dump() for d in update.verified_detections]
        
        record = await db_manager.update_verification(
            request_id=request_id,
            status=update.status,
            verified_detections=verified_dets,
            verified_by=update.verified_by
        )
        
        if not record:
            raise HTTPException(status_code=404, detail="Inferencia no encontrada")
        
        # Si fue corregida, actualizar anotaciones YOLO en GCS
        if update.status == VerificationStatus.CORRECTED and verified_dets and storage_manager:
            try:
                # Necesitamos el tamaño de imagen para generar anotaciones YOLO
                # Lo podemos obtener de las detecciones originales o de la imagen
                if record.image_url:
                    corrected_detections = [
                        Detection(**d) for d in verified_dets
                    ]
                    # Generar nuevas anotaciones YOLO con las correcciones
                    # Nota: Necesitamos el tamaño de imagen. Lo guardamos como
                    # anotaciones con sufijo _verified
                    yolo_annotations = generate_yolo_annotations_from_dicts(
                        verified_dets
                    )
                    await storage_manager.upload_annotations(
                        f"{request_id}_verified",
                        yolo_annotations
                    )
            except Exception as e:
                logger.warning(f"No se pudieron guardar anotaciones corregidas: {e}")
        
        return {
            "status": "success",
            "request_id": request_id,
            "verification_status": record.verification_status,
            "verified_at": record.verified_at.isoformat() if record.verified_at else None,
        }
        
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error actualizando verificación {request_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/review/{request_id}/feedback", tags=["Verification"])
async def submit_user_feedback(request_id: str, feedback: UserFeedback):
    """
    Endpoint simplificado para que el usuario dé feedback rápido
    después de una inferencia: ¿es correcta o no?
    
    - Si dice que es correcta → verified
    - Si dice que NO es correcta → needs_review
    - Si la confianza mínima < 50% → needs_review sin importar el feedback
    """
    if not db_manager:
        raise HTTPException(status_code=503, detail="Database not available")
    
    try:
        record = await db_manager.get_inference(request_id)
        if not record:
            raise HTTPException(status_code=404, detail="Inferencia no encontrada")
        
        # Regla: si la confianza mínima es < 50%, siempre needs_review
        confidence_threshold = settings.confidence_threshold
        force_review = (
            record.min_confidence is not None 
            and record.min_confidence < confidence_threshold
        )
        
        if force_review:
            new_status = VerificationStatus.NEEDS_REVIEW
        elif feedback.is_correct:
            new_status = VerificationStatus.VERIFIED
        else:
            new_status = VerificationStatus.NEEDS_REVIEW
        
        updated = await db_manager.update_verification(
            request_id=request_id,
            status=new_status,
            verified_by=feedback.user_id
        )
        
        return {
            "status": "success",
            "request_id": request_id,
            "verification_status": new_status,
            "forced_review": force_review,
            "message": (
                "Marcada para revisión manual (confianza baja)" 
                if force_review 
                else ("Verificada como correcta" if feedback.is_correct else "Marcada para revisión")
            )
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error procesando feedback {request_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/review/stats", tags=["Verification"])
async def get_verification_stats():
    """Obtiene estadísticas del proceso de verificación."""
    if not db_manager:
        raise HTTPException(status_code=503, detail="Database not available")
    
    try:
        return await db_manager.get_verification_stats()
    except Exception as e:
        logger.error(f"Error obteniendo stats de verificación: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/review/classes", tags=["Verification"])
async def get_available_classes():
    """Retorna las clases disponibles para re-etiquetar."""
    return {
        "classes": [
            {"id": k, "name": v} for k, v in WASTE_CLASSES.items()
        ]
    }


@app.get("/training/export", tags=["Training"])
async def export_training_data(
    limit: int = Query(1000, description="Número máximo de registros"),
    min_detections: int = Query(1, description="Mínimo de detecciones por imagen"),
    only_verified: bool = Query(False, description="Solo exportar datos verificados/corregidos")
):
    """
    Exporta datos de inferencias para reentrenamiento.
    
    Si only_verified=True, exporta solo datos que pasaron por verificación humana,
    usando las detecciones corregidas cuando estén disponibles.
    """
    if not db_manager:
        raise HTTPException(status_code=503, detail="Database not available")
    
    try:
        if only_verified:
            inferences = await db_manager.get_verified_training_data(limit=limit)
        else:
            inferences = await db_manager.get_recent_inferences(limit=limit)
        
        filtered = []
        for inf in inferences:
            if inf.detection_count < min_detections or not inf.image_url:
                continue
            
            # Si fue corregida, usar las detecciones verificadas
            effective_detections = inf.detections
            annotations_suffix = ""
            if inf.verification_status == VerificationStatus.CORRECTED and inf.verified_detections:
                effective_detections = inf.verified_detections
                annotations_suffix = "_verified"
            
            filtered.append({
                "request_id": inf.request_id,
                "timestamp": inf.timestamp.isoformat(),
                "image_url": inf.image_url,
                "annotations_url": (
                    inf.image_url
                    .replace(".jpeg", f"{annotations_suffix}.txt")
                    .replace(".jpg", f"{annotations_suffix}.txt")
                    .replace(".png", f"{annotations_suffix}.txt")
                    if inf.image_url else None
                ),
                "detection_count": len(effective_detections),
                "detections": effective_detections,
                "model_version": inf.model_version,
                "verification_status": inf.verification_status,
            })
        
        return {
            "total_records": len(filtered),
            "min_detections_filter": min_detections,
            "only_verified": only_verified,
            "data": filtered
        }
        
    except Exception as e:
        logger.error(f"Error exportando datos de entrenamiento: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Funciones auxiliares
# ============================================================================

def generate_yolo_annotations(detections: List[Detection], img_width: int, img_height: int) -> str:
    """
    Convierte detecciones al formato de anotación YOLO.
    Formato: class_id x_center y_center width height (normalizado 0-1)
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


def generate_yolo_annotations_from_dicts(detections: List[Dict]) -> str:
    """
    Genera anotaciones YOLO desde diccionarios de detecciones.
    Nota: Las bbox se guardan tal cual si no tenemos tamaño de imagen.
    En ese caso se asume que ya están en coordenadas absolutas y se 
    guardan como referencia para post-procesamiento.
    """
    lines = []
    for det in detections:
        class_id = det.get("class_id", 0)
        bbox = det.get("bbox", [0, 0, 0, 0])
        x1, y1, x2, y2 = bbox
        
        # Guardar en formato YOLO relativo (estimación)
        # El post-procesamiento real debería usar el tamaño de imagen
        line = f"{class_id} {x1:.2f} {y1:.2f} {x2:.2f} {y2:.2f}"
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
                    response.image_size[0],
                    response.image_size[1]
                )
                annotations_url = await storage_manager.upload_annotations(
                    request_id,
                    yolo_annotations
                )
                logger.info(f"Anotaciones guardadas: {annotations_url}")
        
        # Calcular confianza mínima de las detecciones
        min_confidence = None
        if response.detections:
            min_confidence = min(d.confidence for d in response.detections)
        
        # Determinar estado inicial de verificación
        # Si la confianza mínima < umbral → needs_review automáticamente
        initial_status = VerificationStatus.PENDING
        if min_confidence is not None and min_confidence < settings.confidence_threshold:
            initial_status = VerificationStatus.NEEDS_REVIEW
        
        if db_manager:
            record = InferenceRecord(
                request_id=request_id,
                timestamp=datetime.fromisoformat(response.timestamp),
                model_version=response.model_version,
                inference_time_ms=response.inference_time_ms,
                detection_count=response.detection_count,
                detections=[d.model_dump() for d in response.detections],
                image_url=image_url,
                verification_status=initial_status,
                min_confidence=min_confidence
            )
            await db_manager.save_inference(record)
            logger.info(
                f"Registro guardado: {request_id} "
                f"(status={initial_status}, min_conf={min_confidence})"
            )
            
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
