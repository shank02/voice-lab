FROM python:3.11-slim

# System deps: ffmpeg for MP3 decoding, git for pip installs
RUN apt-get update && \
    apt-get install -y --no-install-recommends ffmpeg git && \
    rm -rf /var/lib/apt/lists/*

# Install uv for fast dependency management
RUN pip install --no-cache-dir uv

WORKDIR /app

# Copy dependency files first (cache-friendly layer)
COPY pyproject.toml uv.lock* ./

# Install project dependencies (no dev deps in the image)
RUN uv sync --no-dev --no-install-project

# Copy the rest of the project
COPY . .

# Install the project itself
RUN uv sync --no-dev
