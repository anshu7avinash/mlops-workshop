FROM python:3.10-slim

WORKDIR /mlflow

RUN pip install --no-cache-dir "mlflow>=2.16,<3"

EXPOSE 5000

CMD ["mlflow", "server", "--backend-store-uri", "sqlite:////mlflow/mlflow.db", "--serve-artifacts", "--artifacts-destination", "/mlflow/artifacts", "--host", "0.0.0.0", "--port", "5000"]
