FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app.py .
COPY rice_leaf_app ./rice_leaf_app
ENV GRADIO_SERVER_NAME=0.0.0.0 PORT=7860
EXPOSE 7860
CMD ["python", "app.py"]
