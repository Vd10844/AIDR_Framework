# Use the stable 2.6.1 release. This image is large but contains pre-installed paddle.
FROM paddlepaddle/paddle:2.6.1

WORKDIR /app

# Pin dependencies and use a very long timeout for unreliable networks
COPY requirements.txt .
RUN pip install --no-cache-dir --default-timeout=1000 -r requirements.txt

COPY . .
RUN mkdir -p data/hitl logs

EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
