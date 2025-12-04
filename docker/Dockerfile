# Dockerfile multi-stage para la API FastAPI con soporte Deep Learning

# Stage 1: Build dependencies
FROM python:3.11-slim as builder

WORKDIR /app

# Instalar dependencias del sistema para compilar paquetes y TensorFlow
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    gfortran \
    libopenblas-dev \
    liblapack-dev \
    libhdf5-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Copiar requirements y instalar dependencias
# Instalar TensorFlow primero para optimizar caché de capas
COPY requirements.txt .
RUN pip install --no-cache-dir --user tensorflow>=2.15.0 && \
    pip install --no-cache-dir --user -r requirements.txt

# Stage 2: Runtime
FROM python:3.11-slim

WORKDIR /app

# Instalar dependencias del sistema necesarias para TensorFlow en runtime
RUN apt-get update && apt-get install -y \
    libgomp1 \
    libhdf5-310 \
    && rm -rf /var/lib/apt/lists/*

# Instalar requests para healthcheck
RUN pip install --no-cache-dir requests

# Copiar dependencias instaladas desde el builder
COPY --from=builder /root/.local /root/.local

# Asegurar que los scripts en .local están en PATH
ENV PATH=/root/.local/bin:$PATH

# Variables de entorno para TensorFlow (optimización)
ENV TF_CPP_MIN_LOG_LEVEL=2
ENV TF_FORCE_GPU_ALLOW_GROWTH=true

# Copiar código de la aplicación
COPY . .

# Crear directorios necesarios
RUN mkdir -p logs models data/raw data/processed

# Exponer puerto
EXPOSE 8000

# Script de entrada que detecta qué servicio ejecutar
# Basado en la variable de entorno SERVICE_TYPE
# Por defecto ejecuta la API si SERVICE_TYPE no está definido
CMD ["python", "entrypoint.py"]

