FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1     PYTHONUNBUFFERED=1

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends     build-essential git curl ca-certificates python3-dev     && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY pyproject.toml README.md ./
COPY src ./src
COPY configs ./configs

# Create non-root user
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000
CMD ["python","-m","lemnisiana.dashboard.app"]
