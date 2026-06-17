FROM python:3.12-slim

# uv: fast, reproducible installs
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Install deps first (cached layer)
COPY pyproject.toml uv.lock* ./
RUN uv sync --frozen --no-dev || uv sync --no-dev

# App source
COPY . .

ENV FLAIR_DB_PATH=/data/flair_website.db
VOLUME ["/data"]
EXPOSE 8000

# PORT is honored by most PaaS (Render/Railway/Fly); defaults to 8000.
CMD ["sh", "-c", "uv run uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
