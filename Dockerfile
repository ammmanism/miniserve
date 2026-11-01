FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app

WORKDIR /app

# CPU torch is the reference build for this repo (no GPU on the dev box).
# On a CUDA host replace the index URL with the matching CUDA wheel.
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir torch==2.5.1 \
      --index-url https://download.pytorch.org/whl/cpu

COPY pyproject.toml README.md ./
COPY miniserve ./miniserve
RUN pip install --no-cache-dir -e . "transformers==5.13.0"

COPY bench ./bench
COPY tests ./tests
COPY docs ./docs
COPY REPRODUCE.md AGENTS.md ./

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=10s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')" || exit 1

# Stub generator server (validates plumbing/metrics, not throughput).
CMD ["uvicorn", "miniserve.server:create_app", "--factory", \
     "--host", "0.0.0.0", "--port", "8000"]
