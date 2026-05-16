import io
import time
from collections import Counter
from pathlib import Path

import mlflow
from mlflow import MlflowClient
import numpy as np
from PIL import Image
from ultralytics import YOLO
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import uvicorn
import base64

# ── MLflow setup
MLFLOW_TRACKING_URI = 'mlruns'
REGISTERED_NAME = 'hairnet-compliance-detector'

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.end_run()
mlflow.set_experiment('hairnet-monitoring')
monitoring_run = mlflow.start_run(run_name='api-monitoring')
monitoring_run_id = monitoring_run.info.run_id
client = MlflowClient()

call_count = 0  # incremented on every /predict call


# ── Load Production model
def load_production_model() -> YOLO:
    versions = client.get_latest_versions(REGISTERED_NAME, stages=['Production'])
    if not versions:
        raise RuntimeError(f'No Production model found in registry: {REGISTERED_NAME}')

    version = versions[0]
    run_id = version.run_id
    exp_id = mlflow.get_run(run_id).info.experiment_id
    best_pt = Path('mlruns') / exp_id / run_id / 'artifacts' / 'weights' / 'best.pt'

    if not best_pt.exists():
        raise RuntimeError(f'best.pt not found at {best_pt}')

    print(f'Loading Production model  v{version.version}  from {best_pt}')
    return YOLO(str(best_pt)), version.version


model, model_version = load_production_model()

# ── FastAPI app
app = FastAPI(
    title='HairNet Compliance Detection API',
    description='Detects hairnet compliance in images using the Production YOLO model.',
    version='1.0.0',
)


@app.get('/')
def root():
    return {
        'model': REGISTERED_NAME,
        'version': model_version,
        'stage': 'Production',
        'endpoints': {
            'predict': 'POST /predict  — upload an image, get detections as JSON',
            'health': 'GET  /health   — liveness check',
        },
    }


@app.get('/health')
def health():
    return {'status': 'ok', 'model_loaded': model is not None}


@app.post('/predict')
async def predict(file: UploadFile = File(...)):
    global call_count

    # ── Validate file type
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail='File must be an image.')

    # ── Read image
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert('RGB')
    img_arr = np.array(image)

    # ── Run inference
    t0 = time.perf_counter()
    results = model(img_arr, verbose=False)
    elapsed = round((time.perf_counter() - t0) * 1000, 2)

    # ── Annotated image → base64
    annotated = results[0].plot()
    buffer = io.BytesIO()
    Image.fromarray(annotated).save(buffer, format='JPEG')
    img_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

    # ── Parse detections
    detections = []
    for result in results:
        for box in result.boxes:
            cls_id = int(box.cls[0])
            detections.append({
                'class': result.names[cls_id],
                'confidence': round(float(box.conf[0]), 4),
                'box': {
                    'x1': round(float(box.xyxy[0][0]), 1),
                    'y1': round(float(box.xyxy[0][1]), 1),
                    'x2': round(float(box.xyxy[0][2]), 1),
                    'y2': round(float(box.xyxy[0][3]), 1),
                },
            })

    # ── Log to MLflow
    call_count += 1
    avg_conf = sum(d['confidence'] for d in detections) / len(detections) if detections else 0
    class_counts = Counter(d['class'] for d in detections)

    mlflow.log_metrics({
        'avg_confidence': avg_conf,
        'num_detections': len(detections),
        'inference_ms': elapsed,
        **{f'count_{k}': v for k, v in class_counts.items()},
    }, step=call_count)
    mlflow.log_text(img_b64, artifact_file=f'call_{call_count}_{file.filename}.txt')

    return JSONResponse({
        'filename': file.filename,
        'model_version': model_version,
        'inference_ms': elapsed,
        'num_detections': len(detections),
        'detections': detections,
        'annotated_image_base64': img_b64,
    })


if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000, reload=False)
