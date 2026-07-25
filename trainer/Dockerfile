FROM unsloth/unsloth:latest
WORKDIR /app
RUN pip install --no-cache-dir lm-eval requests
COPY trainer/ /app/trainer/
COPY proxy/db.py proxy/quality.py /app/
COPY entrypoint.sh /app/
RUN chmod +x /app/entrypoint.sh
ENTRYPOINT ["/app/entrypoint.sh"]