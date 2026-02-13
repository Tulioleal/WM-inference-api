# 🗑️ Waste Detection Inference API

API REST para detección y clasificación de desechos en imágenes usando **YOLOv8**. Forma parte de un sistema distribuido que incluye inferencia, almacenamiento de datos verificados y soporte para reentrenamiento continuo del modelo.

## Clases detectadas

| ID | Clase         |
|----|---------------|
| 0  | Biodegradable |
| 1  | Cartón        |
| 2  | Vidrio        |
| 3  | Metal         |
| 4  | Papel         |
| 5  | Plástico      |

## Arquitectura

```
┌─────────────┐     ┌──────────────────┐     ┌─────────────┐
│   Cliente    │────▶│  Inference API   │────▶│  PostgreSQL  │
│  (imagen)   │◀────│  (FastAPI)       │     │  (asyncpg)   │
└─────────────┘     └───────┬──────────┘     └─────────────┘
                            │
                    ┌───────┴──────────┐
                    │  Google Cloud     │
                    │  Storage          │
                    │  - Modelos (.pt)  │
                    │  - Imágenes       │
                    │  - Anotaciones    │
                    └──────────────────┘
```

La API se despliega en **Google Kubernetes Engine (GKE)** y consume recursos de infraestructura provisionados desde un repositorio separado de IaC (namespace, secrets, service accounts con Workload Identity, ConfigMaps de buckets).

## Estructura del proyecto

```
├── main.py              # Aplicación FastAPI y endpoints
├── model_manager.py     # Carga, versionado e inferencia del modelo YOLOv8
├── database.py          # Gestión de PostgreSQL (inferencias, modelos, entrenamientos)
├── storage.py           # Operaciones con Google Cloud Storage
├── config.py            # Variables de entorno con pydantic-settings
├── requirements.txt     # Dependencias Python
├── Dockerfile           # Imagen Docker (Python 3.11 + PyTorch CPU)
└── deployment.yaml      # Manifests de Kubernetes (Deployment, Service, ConfigMap)
```

## Endpoints principales

### Inferencia

| Método | Ruta | Descripción |
|--------|------|-------------|
| `POST` | `/predict` | Recibe una imagen y retorna las detecciones con clase, confianza y bounding box. Opcionalmente guarda la imagen y anotaciones YOLO en GCS. |
| `GET` | `/inferences/{request_id}` | Detalle de una inferencia específica. |
| `GET` | `/images/{request_id}` | Sirve la imagen original desde GCS. |

### Verificación y reentrenamiento

| Método | Ruta | Descripción |
|--------|------|-------------|
| `PUT` | `/inferences/{request_id}/verify` | Permite verificar, corregir o descartar detecciones. |
| `POST` | `/inferences/{request_id}/feedback` | Feedback rápido del usuario (correcto / incorrecto). |
| `GET` | `/training/export` | Exporta datos verificados listos para reentrenamiento. |
| `POST` | `/training/export-to-gcs` | Exporta request IDs verificados a un JSON en GCS para el job de entrenamiento. |

### Modelos

| Método | Ruta | Descripción |
|--------|------|-------------|
| `GET` | `/models` | Lista versiones de modelos con métricas. |
| `POST` | `/models/{version}/activate` | Hot-swap de la versión activa del modelo. |

### Operaciones

| Método | Ruta | Descripción |
|--------|------|-------------|
| `GET` | `/health` | Health check (modelo, DB, GCS). |
| `GET` | `/metrics` | Métricas del servicio (total inferencias, tiempos, uptime). |

La documentación interactiva completa está disponible en `/docs` (Swagger UI).

## Variables de entorno

| Variable | Descripción | Default |
|----------|-------------|---------|
| `DATABASE_URL` | Connection string de PostgreSQL | *requerido* |
| `GCS_MODELS_BUCKET` | Bucket de GCS para modelos | *requerido* |
| `GCS_IMAGES_BUCKET` | Bucket de GCS para imágenes e inferencias | *requerido* |
| `MODEL_VERSION` | Versión del modelo a cargar | `latest` |
| `CONFIDENCE_THRESHOLD` | Umbral mínimo de confianza | `0.5` |
| `IOU_THRESHOLD` | Umbral de IoU para NMS | `0.45` |
| `LOG_LEVEL` | Nivel de logging | `INFO` |
| `MAX_IMAGE_SIZE_MB` | Tamaño máximo de imagen aceptado | `10` |
| `MAX_BATCH_SIZE` | Tamaño máximo de batch | `32` |
| `DEBUG` | Modo debug | `false` |

## Ejecución local

### Requisitos previos

- Python 3.11+
- Credenciales de GCP configuradas (`GOOGLE_APPLICATION_CREDENTIALS` o `gcloud auth`)
- PostgreSQL accesible

### Instalación

```bash
# Crear entorno virtual
python -m venv venv
source venv/bin/activate

# Instalar PyTorch CPU
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Instalar dependencias
pip install -r requirements.txt
```

### Ejecutar

```bash
# Configurar variables (o usar archivo .env)
export DATABASE_URL="postgresql://..."
export GCS_MODELS_BUCKET="mi-bucket-modelos"
export GCS_IMAGES_BUCKET="mi-bucket-imagenes"

python main.py
```

La API estará disponible en `http://localhost:8000`.

## Docker

```bash
# Build
docker build -t inference-api .

# Run
docker run -p 8000:8000 \
  -e DATABASE_URL="postgresql://..." \
  -e GCS_MODELS_BUCKET="mi-bucket-modelos" \
  -e GCS_IMAGES_BUCKET="mi-bucket-imagenes" \
  inference-api
```

La imagen usa PyTorch CPU y corre con un usuario no-root por seguridad.

## Despliegue en Kubernetes

El archivo `deployment.yaml` define los recursos propios de la aplicación. Los recursos de infraestructura (namespace, secrets de DB, service account con Workload Identity, ConfigMap de buckets) se gestionan desde un repositorio de IaC separado.

```bash
kubectl apply -f deployment.yaml
```

## Flujo de verificación

Las inferencias pasan por un ciclo de verificación que alimenta el reentrenamiento:

```
predict → pending → [usuario verifica] → verified / corrected / discarded
                  → [baja confianza]   → needs_review → ...
```

Las inferencias marcadas como `verified` o `corrected` pueden exportarse como datos de entrenamiento, donde las correcciones del usuario reemplazan las detecciones originales.

## Stack tecnológico

- **FastAPI** + **Uvicorn** — framework async
- **YOLOv8** (Ultralytics) — modelo de detección
- **PyTorch** (CPU) — backend de inferencia
- **asyncpg** — cliente async para PostgreSQL
- **Google Cloud Storage** — almacenamiento de modelos, imágenes y anotaciones
- **Docker** + **Kubernetes (GKE)** — contenedorización y orquestación