FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app.py model.joblib ./
ENV GRADIO_SERVER_NAME=0.0.0.0 GRADIO_ANALYTICS_ENABLED=False PYTHONUNBUFFERED=1
EXPOSE 10000
CMD ["python", "app.py"]
