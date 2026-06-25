# syntax=docker/dockerfile:1

FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PATH="/app/.venv/bin:${PATH}"

WORKDIR /app

RUN sed -i 's|http://deb.debian.org/debian|http://mirrors.aliyun.com/debian|g' /etc/apt/sources.list.d/debian.sources \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        ffmpeg \
        freetds-dev \
        gcc \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml uv.lock ./
RUN python -m pip install --no-cache-dir -i https://mirrors.aliyun.com/pypi/simple \
        "fastapi>=0.124.0" \
        "httpx>=0.28.1" \
        "langgraph>=1.2.6" \
        "pydantic>=2.12.0" \
        "pydantic-settings>=2.12.0" \
        "pyjwt>=2.10.0" \
        "pymssql>=2.3.0" \
        "python-multipart>=0.0.20" \
        "pyyaml>=6.0.0" \
        "sse-starlette>=3.0.3" \
        "tencentcloud-sdk-python-asr>=3.1.112" \
        "uvicorn[standard]>=0.38.0"

COPY app ./app
COPY auth ./auth

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3).read()" || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers", "--forwarded-allow-ips", "*"]
