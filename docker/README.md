# Docker Setup

## Ejecución

Para ejecutar todo el sistema con Docker Compose:

```bash
docker compose up --build
```

Esto iniciará:
- PostgreSQL (puerto 5432)
- FastAPI API (puerto 8000)
- Dash Dashboard (puerto 8050)

## Servicios

### Base de Datos (db)
- Imagen: postgres:15-alpine
- Volumen persistente: postgres_data
- Variables de entorno en docker-compose.yml

### API (api)
- Construido desde docker/Dockerfile
- Multi-stage build para optimizar tamaño
- Expone puerto 8000

### Dashboard (dashboard)
- Construido desde docker/Dockerfile.dashboard
- Expone puerto 8050
- Se conecta a la API y base de datos

## Volúmenes

Los datos de PostgreSQL se almacenan en un volumen persistente llamado `postgres_data`.

## Networks

Todos los servicios están en la red `bank_marketing_network` para comunicación interna.

