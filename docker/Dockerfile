# Dockerfile multi-stage para la API FastAPI

# Stage 1: Build dependencies
FROM python:3.11-slim as builder

WORKDIR /app

# Instalar dependencias del sistema para compilar paquetes
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copiar requirements y instalar dependencias
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Stage 2: Runtime
FROM python:3.11-slim

WORKDIR /app

# Copiar dependencias instaladas desde el builder
COPY --from=builder /root/.local /root/.local

# Asegurar que los scripts en .local están en PATH
ENV PATH=/root/.local/bin:$PATH

# Copiar código de la aplicación
COPY . .

# Crear directorios necesarios
RUN mkdir -p logs models data/raw data/processed

# Exponer puerto
EXPOSE 8000

# Comando para ejecutar la API
# Usar run_api.py que maneja $PORT correctamente para Railway
CMD ["python", "run_api.py"]

