# HairNet Compliance Detection

An end-to-end MLOps project that trains, tracks, registers, and serves YOLO object detection models to detect hairnet compliance in food production environments.

---

## What This Project Does

The system detects three classes in images:
- **hairnet** — worker wearing a hairnet correctly
- **No hairnet** — worker not wearing a hairnet (violation)
- **person** — person detected (without hairnet classification)

Eight YOLO models were trained and compared using MLflow. The best model is served as a REST API that returns detections and an annotated image.

---

## Project Structure

```
├── dataset_download_and_explore.ipynb  # EDA — connects to Roboflow API, no local download
├── mlflow_training.ipynb               # Trains 8 YOLO models on Google Colab
├── test_evaluation.ipynb               # Evaluates each model on the test set
├── model_registry.ipynb                # Registers top 4 models, promotes best to Production
├── deployment.ipynb                    # Starts the API and sends test images
├── app.py                              # FastAPI REST API with MLflow monitoring
├── requirements.txt
├── .env.example                        # Copy to .env and fill in your credentials
└── mlruns/                             # MLflow tracking data (local filesystem)
```

---

## Setup

```bash
git clone https://github.com/WajeehAlamoudi/HairNet-Compliance-Detection.git
cd HairNet-Compliance-Detection

python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt

cp .env.example .env          # then fill in your Roboflow credentials
```

---

## Important Notes

### MLflow Tracking — Filesystem Backend
MLflow was configured with a **local filesystem backend** (`mlruns/` folder). This means all experiment data, metrics, and artifacts are stored as files on disk — no database or remote server needed.

```python
mlflow.set_tracking_uri('mlruns')
```

> Note: MLflow has deprecated the filesystem backend as of February 2026 in favour of a database backend (e.g. `sqlite:///mlflow.db`). This project keeps it simple with the filesystem backend since training was done on Google Colab and the `mlruns/` folder was downloaded locally after training.

### Dataset — Roboflow (Online Only During EDA)
The dataset lives on [Roboflow Universe](https://universe.roboflow.com/yolo-training-ywjft/hairnet-compliance-detection/dataset/1). During EDA the project connects to the Roboflow API without downloading anything. The actual download only happens in Colab during training.

- **423 images** — 283 train / 93 valid / 47 test
- **3 classes** — hairnet (1,325), No hairnet (2,065), person (3,429)
- **License** — CC BY 4.0

### Training — Google Colab (T4 GPU)
Training was done on Colab with a T4 GPU. The `mlflow_training.ipynb` notebook is designed to run there. After training, the `mlruns/` folder was downloaded and placed in the project root.

---

## Running the API

Make sure you are in the project root with the venv active, then:

```bash
python app.py
```

The server starts at `http://localhost:8000`.

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Model info and available endpoints |
| `/health` | GET | Liveness check |
| `/predict` | POST | Upload an image, get detections as JSON |

Interactive docs: `http://localhost:8000/docs`

### Example Response from `/predict`

```json
{
  "filename": "worker.jpg",
  "model_version": "4",
  "inference_ms": 142.5,
  "num_detections": 2,
  "detections": [
    {"class": "No hairnet", "confidence": 0.91, "box": {"x1": 120, "y1": 45, "x2": 310, "y2": 280}},
    {"class": "person",     "confidence": 0.87, "box": {"x1": 100, "y1": 30, "x2": 340, "y2": 420}}
  ],
  "annotated_image_base64": "<base64 string>"
}
```

To view the annotated image, paste the base64 string into your browser:
```
data:image/jpeg;base64,<paste here>
```

---

## Viewing MLflow UI

```bash
mlflow ui
```

Open `http://localhost:5000` to see:
- **hairnet-compliance-detection** — all 8 training runs with metrics and artifacts
- **hairnet-monitoring** — live API monitoring run tracking confidence and detections per call

---

## Models Trained

| Run | Model | Epochs | Augmentation |
|---|---|---|---|
| yolov26n-baseline | yolo26n | 30 | Off |
| yolov26m-baseline | yolo26m | 30 | Off |
| yolo12n-baseline | yolo12n | 30 | Off |
| yolo12m-baseline | yolo12m | 30 | Off |
| yolov26n-augmented | yolo26n | 50 | On |
| yolov26m-augmented | yolo26m | 50 | On |
| yolo12n-augmented | yolo12n | 50 | On |
| yolo12m-augmented | yolo12m | 50 | On |

Top 4 (by validation class loss) were registered in the MLflow Model Registry. The best by `test/mAP50` was promoted to **Production**.
