FROM python:3.13

RUN apt-get update -y && \
    apt-get install -y \
    python3-opencv \
    gnome-screenshot && \
    rm -rf /var/lib/apt/lists/*
# Instalar UV
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /home/src

COPY pyproject.toml uv.lock .python-version ./
COPY src ./src
COPY detector_neumonia.py ./
# Crear el entorno e instalar dependencias con UV
RUN uv sync --dev

COPY . .

CMD ["uv", "run", "python", "detector_neumonia.py"]