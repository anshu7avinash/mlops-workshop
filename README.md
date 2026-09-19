# MLOps Workshop: E-commerce Recommendations

This repository is a small end-to-end MLOps workshop project. It trains a content-based product recommendation model, tracks data and pipeline outputs with DVC, records experiments with MLflow, serves recommendations through FastAPI, and provides a Streamlit user interface.

## What This Project Demonstrates

- Reproducible model training with DVC.
- Data versioning for `model/products.csv`.
- Parameterized training through `params.yaml`.
- Experiment tracking and artifacts with MLflow.
- A FastAPI inference service.
- A Streamlit frontend.
- Local container orchestration with Docker Compose.
- Kubernetes deployment manifests for the API and UI.

## Repository Layout

```text
.
├── api/                    FastAPI inference service and Dockerfile
├── k8s/                    Kubernetes namespace, API, and UI manifests
├── model/                  Training code, data, metrics, and dependencies
├── tracking-server/        MLflow tracking server Dockerfile
├── ui/                     Streamlit application and Dockerfile
├── workshop_assets/        Alternate workshop input data
├── docker-compose.yml      Local API, UI, and optional MLflow services
├── dvc.yaml                DVC training pipeline
├── dvc.lock                Locked pipeline dependency information
└── params.yaml             Model and experiment parameters
```

## Model Workflow

`model/train_model.py` reads the product catalogue and builds a nearest-neighbours recommender using:

- One-hot encoding for product category.
- Standard scaling for price and rating.
- Cosine distance by default.
- MLflow logging for parameters, metrics, metadata, and the trained model.

The current parameters are in `params.yaml`:

```yaml
model:
  n_neighbors: 6
  metric: cosine
  top_k: 3
  experiment_name: ecommerce-recommendation-workshop
```

The current tracked metrics are stored in `model/metrics.json`. They are proxy metrics because this workshop catalogue does not contain user click or purchase labels.

## Local Setup on macOS/Linux

Python 3.12 is recommended for this workshop. It has broad support for the scientific Python wheels used by the project, including PyArrow.

```bash
cd /Users/avinashanshu/Downloads/mlops-workshop

# Install Python 3.12 if it is not already installed.
brew install python@3.12

# Create and activate the project environment.
/opt/homebrew/bin/python3.12 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip setuptools wheel
python -m pip install -r model/requirements.txt
```

The model requirements include DVC 3.x:

```text
dvc>=3.50,<4
```

Verify the environment:

```bash
which python
which dvc
python -c "import joblib, mlflow, pandas, sklearn; print('Ready')"
dvc --version
```

Every new terminal needs to activate the same environment again:

```bash
cd /Users/avinashanshu/Downloads/mlops-workshop
source .venv/bin/activate
```

Packages installed into `.venv` persist across terminals. A new terminal does not delete them, but it does not inherit the previous terminal's activation or exported environment variables.

## DVC Setup and Data Versioning

Initialize Git and DVC once if starting from a fresh clone or a new local repository:

```bash
git init
dvc init
dvc remote add -d workshop-storage .dvc-storage
```

The local DVC storage directory is intentionally ignored because it is a generated cache/local remote. The `.dvc` metadata files and `dvc.lock` are committed instead.

Track the current catalogue:

```bash
dvc add model/products.csv
git add model/products.csv.dvc .gitignore
```

Run the pipeline:

```bash
dvc repro
```

Force a full training run even when DVC believes the stage is up to date:

```bash
dvc repro -f
```

The training stage runs `python model/train_model.py` and produces:

- `model/recommendation_model.pkl`
- `model/metrics.json`
- `model/model_metadata.json`
- MLflow run data

Generated model files and local MLflow data are ignored by Git. DVC and MLflow metadata needed to reproduce or inspect the workflow remain in the repository.

## MLflow Tracking

Start the MLflow server with Docker Desktop running:

```bash
docker compose --profile tools up --build mlflow
```

This repository maps host port `5001` to the MLflow container's port `5000`, because host port `5000` was already occupied on the development machine.

Open MLflow at:

```text
http://localhost:5001
```

In another terminal, activate the same environment and configure the training process:

```bash
cd /Users/avinashanshu/Downloads/mlops-workshop
source .venv/bin/activate
export MLFLOW_TRACKING_URI=http://localhost:5001
dvc repro -f
```

Keep the MLflow Docker terminal running while training. Use a second terminal for DVC commands.

## Change the Data and Retrain

The alternate catalogue is located at `workshop_assets/products_v2.csv`. Replace the tracked catalogue, then update DVC and reproduce the pipeline:

```bash
cp workshop_assets/products_v2.csv model/products.csv
dvc add model/products.csv
dvc repro

git add model/products.csv.dvc dvc.lock
git commit -m "Retrain recommendation model with product v2 data"
```

If MLflow is running on the local Docker port, make sure `MLFLOW_TRACKING_URI` is still exported before running the command.

## Run the Application with Docker Compose

Build and start the API and UI:

```bash
docker compose up --build api ui
```

Services:

- FastAPI: `http://localhost:8000`
- FastAPI interactive docs: `http://localhost:8000/docs`
- Streamlit UI: `http://localhost:8501`

The UI uses `API_BASE_URL=http://api:8000` inside the Compose network. The API container expects the generated model at `model/recommendation_model.pkl`, so run the DVC pipeline before building the API image.

Useful API endpoints:

```text
GET  /
GET  /products
GET  /model-info
POST /recommend
```

Example request:

```bash
curl -X POST http://localhost:8000/recommend \
  -H 'Content-Type: application/json' \
  -d '{"product_id": 1}'
```

Stop the services with:

```bash
docker compose down
```

## Kubernetes Deployment

The manifests use locally built images named `recommendation-api:local` and `recommendation-ui:local`. Build those images first:

```bash
docker build -f api/Dockerfile -t recommendation-api:local .
docker build -f ui/Dockerfile -t recommendation-ui:local .
```

Apply the namespace and workloads:

```bash
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/api.yaml
kubectl apply -f k8s/ui.yaml
```

The API is exposed internally as `recommendation-api:8000`. The UI is exposed through a Kubernetes `NodePort`. The exact access command depends on the Kubernetes distribution; for Docker Desktop, inspect the assigned service port with:

```bash
kubectl -n mlops-workshop get services
```

Check deployment status:

```bash
kubectl -n mlops-workshop get pods
kubectl -n mlops-workshop get deployments
```

## Files Intentionally Excluded from Git

The following are local environments, caches, runtime state, or generated outputs and should not be uploaded to GitHub:

- `.venv/`, `.venv*`: Python virtual environments.
- `.vscode/`: local editor settings.
- `mlflow-data/`: MLflow server database and artifacts.
- `mlruns/`: local MLflow file-store runs.
- `.dvc/cache/` and `.dvc/tmp/`: DVC local cache files.
- `.dvc-storage/`: local DVC remote storage.
- `model/recommendation_model.pkl`: generated model binary.
- `model/metrics.json` and `model/model_metadata.json`: generated training outputs.

These files are recreated locally by the setup, training, Docker, and MLflow commands. Keeping them out of Git avoids committing machine-specific or potentially large runtime data.

## Troubleshooting

### `dvc: command not found`

Activate the project environment and confirm that DVC is installed into that environment:

```bash
source .venv/bin/activate
which python
python -m pip show dvc
python -m pip install -r model/requirements.txt
dvc --version
```

### `ModuleNotFoundError: No module named 'joblib'` or `mlflow`

The requirements installation did not finish successfully. Install the requirements into the active environment, then verify imports:

```bash
source .venv/bin/activate
python -m pip install -r model/requirements.txt
python -c "import joblib, mlflow, pandas, sklearn; print('Ready')"
```

### PyArrow tries to compile and cannot find CMake

This usually means the interpreter is too new for the available prebuilt wheel. Use Python 3.12 for the virtual environment, recreate `.venv`, and install the requirements again.

### Port 5000 is already in use

The Compose file maps MLflow to host port `5001`:

```yaml
ports:
  - "5001:5000"
```

Use `http://localhost:5001` and export `MLFLOW_TRACKING_URI=http://localhost:5001`.

## GitHub Repository

The workshop source is published at:

https://github.com/anshu7avinash/mlops-workshop
