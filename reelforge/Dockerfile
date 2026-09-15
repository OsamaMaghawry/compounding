# ReelForge as a service. Everything it needs is in the image except the speech
# model, which is downloaded on first use and cached in the `models` volume.
FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg git ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt requirements-web.txt requirements-asr.txt ./
RUN pip install --no-cache-dir -r requirements.txt \
                               -r requirements-web.txt \
                               -r requirements-asr.txt

COPY . .
RUN pip install --no-cache-dir -e . && python -m reelforge setup || true

# Model cache and uploads live on volumes, so a rebuild does not lose them.
ENV HF_HOME=/models \
    REELFORGE_DATA=/data \
    PYTHONUNBUFFERED=1
VOLUME ["/data", "/models"]
EXPOSE 8000

# Set REELFORGE_PASSWORD, or a random one is printed in the logs at startup.
CMD ["python", "-m", "reelforge", "serve", "--host", "0.0.0.0", "--port", "8000", "--data", "/data"]
