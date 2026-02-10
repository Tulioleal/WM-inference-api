"""
Database Manager - Gestión de PostgreSQL para el sistema de detección
"""

import logging
import json
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field

import asyncpg
from asyncpg import Pool

logger = logging.getLogger(__name__)


# ============================================================================
# Estados de verificación
# ============================================================================

class VerificationStatus:
    PENDING = "pending"           # Recién inferida, sin revisar
    NEEDS_REVIEW = "needs_review" # Marcada para revisión (baja confianza o usuario reportó)
    VERIFIED = "verified"         # Usuario confirmó que es correcta
    CORRECTED = "corrected"       # Usuario corrigió las etiquetas
    DISCARDED = "discarded"       # Imagen descartada (no útil para entrenamiento)


@dataclass
class InferenceRecord:
    """Registro de una inferencia"""
    request_id: str
    timestamp: datetime
    model_version: str
    inference_time_ms: float
    detection_count: int
    detections: List[Dict[str, Any]] = field(default_factory=list)
    image_url: Optional[str] = None
    # Campos de verificación
    verification_status: str = VerificationStatus.PENDING
    verified_detections: Optional[List[Dict[str, Any]]] = None
    verified_by: Optional[str] = None
    verified_at: Optional[datetime] = None
    min_confidence: Optional[float] = None


@dataclass
class ModelMetadata:
    """Metadata de un modelo"""
    version: str
    created_at: datetime
    accuracy: Optional[float] = None
    map50: Optional[float] = None
    map50_95: Optional[float] = None
    training_epochs: Optional[int] = None
    training_time_seconds: Optional[float] = None
    dataset_size: Optional[int] = None
    notes: Optional[str] = None


@dataclass
class TrainingRecord:
    """Registro de un entrenamiento"""
    job_id: str
    started_at: datetime
    finished_at: Optional[datetime] = None
    status: str = "running"
    model_version: Optional[str] = None
    epochs: int = 50
    batch_size: int = 16
    learning_rate: float = 0.01
    final_accuracy: Optional[float] = None
    final_map50: Optional[float] = None
    error_message: Optional[str] = None


class DatabaseManager:
    """
    Gestiona las operaciones con PostgreSQL.
    """
    
    def __init__(self, database_url: str):
        self.database_url = database_url
        self.pool: Optional[Pool] = None
    
    async def connect(self) -> None:
        """Establece conexión con la base de datos"""
        try:
            self.pool = await asyncpg.create_pool(
                self.database_url,
                min_size=2,
                max_size=10,
                command_timeout=60
            )
            
            # Crear tablas si no existen
            await self._create_tables()
            
            logger.info("Conexión a base de datos establecida")
            
        except Exception as e:
            logger.error(f"Error conectando a base de datos: {e}")
            raise
    
    async def disconnect(self) -> None:
        """Cierra la conexión"""
        if self.pool:
            await self.pool.close()
            logger.info("Conexión a base de datos cerrada")
    
    async def health_check(self) -> bool:
        """Verifica estado de la conexión"""
        if not self.pool:
            return False
        
        try:
            async with self.pool.acquire() as conn:
                await conn.fetchval("SELECT 1")
            return True
        except Exception:
            return False
    
    async def _create_tables(self) -> None:
        """Crea las tablas necesarias"""
        async with self.pool.acquire() as conn:
            # Tabla de inferencias (con campos de verificación)
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS inferences (
                    id SERIAL PRIMARY KEY,
                    request_id VARCHAR(36) UNIQUE NOT NULL,
                    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
                    model_version VARCHAR(50) NOT NULL,
                    inference_time_ms FLOAT NOT NULL,
                    detection_count INTEGER NOT NULL,
                    detections JSONB,
                    image_url TEXT,
                    verification_status VARCHAR(20) NOT NULL DEFAULT 'pending',
                    verified_detections JSONB,
                    verified_by VARCHAR(100),
                    verified_at TIMESTAMP WITH TIME ZONE,
                    min_confidence FLOAT,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                )
            """)
            
            # Índices para inferencias
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_inferences_timestamp 
                ON inferences(timestamp DESC)
            """)
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_inferences_model 
                ON inferences(model_version)
            """)
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_inferences_verification 
                ON inferences(verification_status)
            """)
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_inferences_min_confidence 
                ON inferences(min_confidence)
            """)
            
            # Migración: agregar columnas si no existen (para DBs existentes)
            for col, col_type, col_default in [
                ("verification_status", "VARCHAR(20)", "'pending'"),
                ("verified_detections", "JSONB", "NULL"),
                ("verified_by", "VARCHAR(100)", "NULL"),
                ("verified_at", "TIMESTAMP WITH TIME ZONE", "NULL"),
                ("min_confidence", "FLOAT", "NULL"),
            ]:
                try:
                    await conn.execute(f"""
                        ALTER TABLE inferences 
                        ADD COLUMN IF NOT EXISTS {col} {col_type} DEFAULT {col_default}
                    """)
                except Exception:
                    pass  # La columna ya existe
            
            # Tabla de modelos
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS models (
                    id SERIAL PRIMARY KEY,
                    version VARCHAR(50) UNIQUE NOT NULL,
                    created_at TIMESTAMP WITH TIME ZONE NOT NULL,
                    accuracy FLOAT,
                    map50 FLOAT,
                    map50_95 FLOAT,
                    training_epochs INTEGER,
                    training_time_seconds FLOAT,
                    dataset_size INTEGER,
                    notes TEXT,
                    gcs_path TEXT
                )
            """)
            
            # Tabla de entrenamientos
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS trainings (
                    id SERIAL PRIMARY KEY,
                    job_id VARCHAR(100) UNIQUE NOT NULL,
                    started_at TIMESTAMP WITH TIME ZONE NOT NULL,
                    finished_at TIMESTAMP WITH TIME ZONE,
                    status VARCHAR(20) NOT NULL DEFAULT 'running',
                    model_version VARCHAR(50),
                    epochs INTEGER,
                    batch_size INTEGER,
                    learning_rate FLOAT,
                    final_accuracy FLOAT,
                    final_map50 FLOAT,
                    error_message TEXT
                )
            """)
            
            logger.info("Tablas de base de datos verificadas/creadas")
    
    # ========================================================================
    # Operaciones de Inferencias
    # ========================================================================
    
    async def save_inference(self, record: InferenceRecord) -> None:
        """Guarda un registro de inferencia"""
        async with self.pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO inferences 
                (request_id, timestamp, model_version, inference_time_ms, 
                 detection_count, detections, image_url,
                 verification_status, min_confidence)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                ON CONFLICT (request_id) DO NOTHING
            """,
                record.request_id,
                record.timestamp,
                record.model_version,
                record.inference_time_ms,
                record.detection_count,
                json.dumps(record.detections),  # asyncpg maneja JSON automáticamente
                record.image_url,
                record.verification_status,
                record.min_confidence
            )
    
    async def get_inference(self, request_id: str) -> Optional[InferenceRecord]:
        """Obtiene un registro de inferencia por ID"""
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow("""
                SELECT request_id, timestamp, model_version, inference_time_ms,
                       detection_count, detections, image_url,
                       verification_status, verified_detections, verified_by,
                       verified_at, min_confidence
                FROM inferences WHERE request_id = $1
            """, request_id)
            
            if row:
                return self._row_to_inference(row)
            return None
    
    async def get_inference_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas de inferencias"""
        async with self.pool.acquire() as conn:
            # Total de inferencias
            total = await conn.fetchval(
                "SELECT COUNT(*) FROM inferences"
            )
            
            # Promedio de tiempo de inferencia
            avg_time = await conn.fetchval("""
                SELECT AVG(inference_time_ms) FROM inferences
            """) or 0
            
            # Inferencias en la última hora
            last_hour = await conn.fetchval("""
                SELECT COUNT(*) FROM inferences 
                WHERE timestamp > NOW() - INTERVAL '1 hour'
            """)
            
            # Distribución por clase (top 5)
            class_dist = await conn.fetch("""
                SELECT 
                    d->>'class_name' as class_name,
                    COUNT(*) as count
                FROM inferences, 
                     jsonb_array_elements(detections) as d
                GROUP BY d->>'class_name'
                ORDER BY count DESC
                LIMIT 5
            """)
            
            return {
                "total": total,
                "avg_time_ms": round(avg_time, 2),
                "last_hour": last_hour,
                "top_classes": [
                    {"class": r['class_name'], "count": r['count']} 
                    for r in class_dist
                ]
            }
    
    async def get_recent_inferences(
        self, 
        limit: int = 100,
        model_version: Optional[str] = None
    ) -> List[InferenceRecord]:
        """Obtiene inferencias recientes"""
        async with self.pool.acquire() as conn:
            if model_version:
                rows = await conn.fetch("""
                    SELECT request_id, timestamp, model_version, inference_time_ms,
                           detection_count, detections, image_url,
                           verification_status, verified_detections, verified_by,
                           verified_at, min_confidence
                    FROM inferences 
                    WHERE model_version = $1
                    ORDER BY timestamp DESC
                    LIMIT $2
                """, model_version, limit)
            else:
                rows = await conn.fetch("""
                    SELECT request_id, timestamp, model_version, inference_time_ms,
                           detection_count, detections, image_url,
                           verification_status, verified_detections, verified_by,
                           verified_at, min_confidence
                    FROM inferences 
                    ORDER BY timestamp DESC
                    LIMIT $1
                """, limit)
            
            return [self._row_to_inference(row) for row in rows]
    
    # ========================================================================
    # Operaciones de Verificación
    # ========================================================================
    
    async def get_review_queue(
        self,
        status_filter: Optional[str] = None,
        limit: int = 50,
        offset: int = 0
    ) -> Dict[str, Any]:
        """
        Obtiene la cola de imágenes pendientes de revisión.
        
        Prioridad:
        1. needs_review (baja confianza o reportadas por usuario)
        2. pending (no revisadas aún)
        
        Args:
            status_filter: Filtrar por estado específico
            limit: Máximo de resultados
            offset: Offset para paginación
        """
        async with self.pool.acquire() as conn:
            # Contar totales por estado
            counts = await conn.fetch("""
                SELECT verification_status, COUNT(*) as count
                FROM inferences
                WHERE image_url IS NOT NULL
                GROUP BY verification_status
            """)
            status_counts = {r['verification_status']: r['count'] for r in counts}
            
            # Obtener registros
            if status_filter:
                rows = await conn.fetch("""
                    SELECT request_id, timestamp, model_version, inference_time_ms,
                           detection_count, detections, image_url,
                           verification_status, verified_detections, verified_by,
                           verified_at, min_confidence
                    FROM inferences
                    WHERE image_url IS NOT NULL
                      AND verification_status = $1
                    ORDER BY 
                        CASE WHEN min_confidence IS NOT NULL THEN min_confidence ELSE 1.0 END ASC,
                        timestamp DESC
                    LIMIT $2 OFFSET $3
                """, status_filter, limit, offset)
            else:
                # Por defecto: needs_review primero, luego pending
                rows = await conn.fetch("""
                    SELECT request_id, timestamp, model_version, inference_time_ms,
                           detection_count, detections, image_url,
                           verification_status, verified_detections, verified_by,
                           verified_at, min_confidence
                    FROM inferences
                    WHERE image_url IS NOT NULL
                      AND verification_status IN ('needs_review', 'pending')
                    ORDER BY 
                        CASE verification_status 
                            WHEN 'needs_review' THEN 0 
                            WHEN 'pending' THEN 1 
                        END,
                        CASE WHEN min_confidence IS NOT NULL THEN min_confidence ELSE 1.0 END ASC,
                        timestamp DESC
                    LIMIT $1 OFFSET $2
                """, limit, offset)
            
            records = [self._row_to_inference(row) for row in rows]
            
            return {
                "records": records,
                "status_counts": status_counts,
                "total_pending": status_counts.get(VerificationStatus.PENDING, 0)
                    + status_counts.get(VerificationStatus.NEEDS_REVIEW, 0),
                "limit": limit,
                "offset": offset,
            }
    
    async def update_verification(
        self,
        request_id: str,
        status: str,
        verified_detections: Optional[List[Dict[str, Any]]] = None,
        verified_by: str = "anonymous"
    ) -> Optional[InferenceRecord]:
        """
        Actualiza el estado de verificación de una inferencia.
        
        Args:
            request_id: ID de la inferencia
            status: Nuevo estado de verificación
            verified_detections: Detecciones corregidas (si se corrigió)
            verified_by: Identificador del revisor
        """
        valid_statuses = [
            VerificationStatus.VERIFIED,
            VerificationStatus.CORRECTED,
            VerificationStatus.NEEDS_REVIEW,
            VerificationStatus.DISCARDED,
        ]
        if status not in valid_statuses:
            raise ValueError(f"Estado inválido: {status}. Válidos: {valid_statuses}")
        
        async with self.pool.acquire() as conn:
            # Si es corrección, guardar las detecciones verificadas
            if status == VerificationStatus.CORRECTED and verified_detections is not None:
                await conn.execute("""
                    UPDATE inferences
                    SET verification_status = $2,
                        verified_detections = $3,
                        verified_by = $4,
                        verified_at = NOW()
                    WHERE request_id = $1
                """,
                    request_id,
                    status,
                    json.dumps(verified_detections),
                    verified_by
                )
            else:
                await conn.execute("""
                    UPDATE inferences
                    SET verification_status = $2,
                        verified_by = $3,
                        verified_at = NOW()
                    WHERE request_id = $1
                """,
                    request_id,
                    status,
                    verified_by
                )
        
        return await self.get_inference(request_id)
    
    async def flag_low_confidence_inferences(
        self,
        threshold: float = 0.5
    ) -> int:
        """
        Marca como needs_review todas las inferencias con confianza 
        mínima por debajo del umbral que aún estén en estado pending.
        
        Returns:
            Número de registros actualizados
        """
        async with self.pool.acquire() as conn:
            result = await conn.execute("""
                UPDATE inferences
                SET verification_status = 'needs_review'
                WHERE verification_status = 'pending'
                  AND min_confidence IS NOT NULL
                  AND min_confidence < $1
            """, threshold)
            
            # asyncpg retorna "UPDATE N"
            count = int(result.split()[-1])
            logger.info(f"Marcados {count} registros como needs_review (confianza < {threshold})")
            return count
    
    async def get_verification_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas de verificación"""
        async with self.pool.acquire() as conn:
            counts = await conn.fetch("""
                SELECT verification_status, COUNT(*) as count
                FROM inferences
                WHERE image_url IS NOT NULL
                GROUP BY verification_status
            """)
            
            avg_confidence_by_status = await conn.fetch("""
                SELECT verification_status, 
                       AVG(min_confidence) as avg_min_confidence,
                       COUNT(*) as count
                FROM inferences
                WHERE image_url IS NOT NULL 
                  AND min_confidence IS NOT NULL
                GROUP BY verification_status
            """)
            
            return {
                "counts": {r['verification_status']: r['count'] for r in counts},
                "confidence_by_status": {
                    r['verification_status']: {
                        "avg_min_confidence": round(r['avg_min_confidence'], 4) if r['avg_min_confidence'] else None,
                        "count": r['count']
                    }
                    for r in avg_confidence_by_status
                }
            }
    
    async def get_verified_training_data(
        self,
        limit: int = 1000
    ) -> List[InferenceRecord]:
        """
        Obtiene datos verificados listos para reentrenamiento.
        Incluye tanto las verificadas (etiqueta original correcta)
        como las corregidas (usa verified_detections).
        """
        async with self.pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT request_id, timestamp, model_version, inference_time_ms,
                       detection_count, detections, image_url,
                       verification_status, verified_detections, verified_by,
                       verified_at, min_confidence
                FROM inferences
                WHERE image_url IS NOT NULL
                  AND verification_status IN ('verified', 'corrected')
                ORDER BY verified_at DESC
                LIMIT $1
            """, limit)
            
            return [self._row_to_inference(row) for row in rows]
    
    # ========================================================================
    # Operaciones de Modelos
    # ========================================================================
    
    async def save_model(self, model: ModelMetadata, gcs_path: str) -> None:
        """Guarda metadata de un modelo"""
        async with self.pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO models 
                (version, created_at, accuracy, map50, map50_95, 
                 training_epochs, training_time_seconds, dataset_size, notes, gcs_path)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                ON CONFLICT (version) DO UPDATE SET
                    accuracy = EXCLUDED.accuracy,
                    map50 = EXCLUDED.map50,
                    map50_95 = EXCLUDED.map50_95,
                    training_epochs = EXCLUDED.training_epochs,
                    training_time_seconds = EXCLUDED.training_time_seconds,
                    dataset_size = EXCLUDED.dataset_size,
                    notes = EXCLUDED.notes,
                    gcs_path = EXCLUDED.gcs_path
            """,
                model.version,
                model.created_at,
                model.accuracy,
                model.map50,
                model.map50_95,
                model.training_epochs,
                model.training_time_seconds,
                model.dataset_size,
                model.notes,
                gcs_path
            )
    
    async def get_model_versions(self) -> List[ModelMetadata]:
        """Lista todas las versiones de modelos"""
        async with self.pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT version, created_at, accuracy, map50, map50_95,
                       training_epochs, training_time_seconds, dataset_size, notes
                FROM models
                ORDER BY created_at DESC
            """)
            
            return [
                ModelMetadata(
                    version=row['version'],
                    created_at=row['created_at'],
                    accuracy=row['accuracy'],
                    map50=row['map50'],
                    map50_95=row['map50_95'],
                    training_epochs=row['training_epochs'],
                    training_time_seconds=row['training_time_seconds'],
                    dataset_size=row['dataset_size'],
                    notes=row['notes']
                )
                for row in rows
            ]
    
    async def get_latest_model_version(self) -> Optional[str]:
        """Obtiene la versión más reciente del modelo"""
        async with self.pool.acquire() as conn:
            return await conn.fetchval("""
                SELECT version FROM models 
                ORDER BY created_at DESC LIMIT 1
            """)
    
    # ========================================================================
    # Operaciones de Entrenamientos
    # ========================================================================
    
    async def save_training(self, training: TrainingRecord) -> None:
        """Guarda registro de entrenamiento"""
        async with self.pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO trainings 
                (job_id, started_at, finished_at, status, model_version,
                 epochs, batch_size, learning_rate, final_accuracy, 
                 final_map50, error_message)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
                ON CONFLICT (job_id) DO UPDATE SET
                    finished_at = EXCLUDED.finished_at,
                    status = EXCLUDED.status,
                    model_version = EXCLUDED.model_version,
                    final_accuracy = EXCLUDED.final_accuracy,
                    final_map50 = EXCLUDED.final_map50,
                    error_message = EXCLUDED.error_message
            """,
                training.job_id,
                training.started_at,
                training.finished_at,
                training.status,
                training.model_version,
                training.epochs,
                training.batch_size,
                training.learning_rate,
                training.final_accuracy,
                training.final_map50,
                training.error_message
            )
    
    async def get_training_history(self, limit: int = 20) -> List[TrainingRecord]:
        """Obtiene historial de entrenamientos"""
        async with self.pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT job_id, started_at, finished_at, status, model_version,
                       epochs, batch_size, learning_rate, final_accuracy,
                       final_map50, error_message
                FROM trainings
                ORDER BY started_at DESC
                LIMIT $1
            """, limit)
            
            return [
                TrainingRecord(
                    job_id=row['job_id'],
                    started_at=row['started_at'],
                    finished_at=row['finished_at'],
                    status=row['status'],
                    model_version=row['model_version'],
                    epochs=row['epochs'],
                    batch_size=row['batch_size'],
                    learning_rate=row['learning_rate'],
                    final_accuracy=row['final_accuracy'],
                    final_map50=row['final_map50'],
                    error_message=row['error_message']
                )
                for row in rows
            ]
    
    # ========================================================================
    # Helpers
    # ========================================================================
    
    def _row_to_inference(self, row) -> InferenceRecord:
        """Convierte una fila de DB a InferenceRecord"""
        return InferenceRecord(
            request_id=row['request_id'],
            timestamp=row['timestamp'],
            model_version=row['model_version'],
            inference_time_ms=row['inference_time_ms'],
            detection_count=row['detection_count'],
            detections=json.loads(row['detections']) if isinstance(row['detections'], str) else (row['detections'] or []),
            image_url=row['image_url'],
            verification_status=row.get('verification_status', VerificationStatus.PENDING) or VerificationStatus.PENDING,
            verified_detections=json.loads(row['verified_detections']) if isinstance(row.get('verified_detections'), str) else row.get('verified_detections'),
            verified_by=row.get('verified_by'),
            verified_at=row.get('verified_at'),
            min_confidence=row.get('min_confidence')
        )
