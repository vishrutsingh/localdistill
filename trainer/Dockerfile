FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY train.py dataset_exporter.py db.py quality.py ./
ENTRYPOINT ["python", "train.py"]