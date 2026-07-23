FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY proxy_server.py db.py quality.py ./
EXPOSE 8787
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8787/health')" || exit 1
CMD ["python", "proxy_server.py", "--port", "8787"]