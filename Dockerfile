FROM python:3.12-slim

ENV DEBIAN_FRONTEND=noninteractive \
    OMP_NUM_THREADS=1 \
    HOST=0.0.0.0 \
    PORT=8000 \
    MAX_CONCURRENT_JOBS=1 \
    FACEFUSION_MEMORY_STRATEGY=tolerant

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ffmpeg \
    git \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

COPY . /app/

EXPOSE 8000 7860

CMD ["python", "api_server.py"]
