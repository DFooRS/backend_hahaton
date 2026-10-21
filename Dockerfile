FROM python:3.12-slim

WORKDIR /app
COPY requirements.txt ./
COPY ml_backend/requirements.txt ./ml_backend/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
COPY app ./app
COPY ml_backend ./ml_backend
COPY scripts ./scripts
COPY migrations ./migrations
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
