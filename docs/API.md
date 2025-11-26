# Documentación de la API

## Endpoints Disponibles

### 1. Health Check

**GET** `/api/v1/health`

Verifica el estado de la API.

**Response**:
```json
{
  "status": "healthy",
  "timestamp": "2025-11-25 19:00:00"
}
```

### 2. Predicción

**POST** `/api/v1/predict`

Realiza una predicción para determinar si un cliente suscribirá un depósito.

**Request Body** (JSON):

| Campo | Tipo | Descripción | Validación |
|-------|------|-------------|------------|
| age | integer | Edad del cliente | 0 ≤ age ≤ 120 |
| job | string | Tipo de trabajo | Valores válidos: admin., blue-collar, entrepreneur, etc. |
| marital | string | Estado civil | Valores: single, married, divorced |
| education | string | Nivel educativo | Valores: primary, secondary, tertiary, unknown |
| default | string | Tiene crédito en default? | Valores: yes, no |
| balance | float | Balance promedio anual | Sin restricciones |
| housing | string | Tiene préstamo hipotecario? | Valores: yes, no |
| loan | string | Tiene préstamo personal? | Valores: yes, no |
| contact | string | Tipo de contacto | Valores: cellular, telephone, unknown |
| day | integer | Día del último contacto | 1 ≤ day ≤ 31 |
| month | string | Mes del último contacto | Valores: jan, feb, mar, ..., dec |
| duration | integer | Duración del último contacto (segundos) | duration ≥ 0 |
| campaign | integer | Número de contactos en esta campaña | campaign ≥ 1 |
| pdays | integer | Días desde último contacto | Sin restricciones (típicamente -1 si no contactado) |
| previous | integer | Número de contactos anteriores | previous ≥ 0 |
| poutcome | string | Resultado de campaña anterior | Valores: success, failure, other, unknown |

**Example Request**:
```json
{
  "age": 59,
  "job": "admin.",
  "marital": "married",
  "education": "secondary",
  "default": "no",
  "balance": 2343.0,
  "housing": "yes",
  "loan": "no",
  "contact": "unknown",
  "day": 5,
  "month": "may",
  "duration": 1042,
  "campaign": 1,
  "pdays": -1,
  "previous": 0,
  "poutcome": "unknown"
}
```

**Response**:
```json
{
  "prediction": 1,
  "probability": 0.8567,
  "class_name": "yes"
}
```

**Códigos de Respuesta**:
- `200 OK`: Predicción exitosa
- `422 Unprocessable Entity`: Datos de entrada inválidos
- `500 Internal Server Error`: Error en el servidor

### 3. Métricas

**GET** `/api/v1/metrics`

Obtiene las métricas actuales del modelo calculadas en el conjunto de test.

**Response**:
```json
{
  "model": "Decision Tree - Bank Marketing",
  "accuracy": 0.8567,
  "precision": 0.8234,
  "recall": 0.7891,
  "f1_score": 0.8059,
  "auc_roc": 0.9123,
  "confusion_matrix": [[1500, 200], [150, 800]],
  "classification_report": {
    "0": {
      "precision": 0.9091,
      "recall": 0.8824,
      "f1-score": 0.8955,
      "support": 1700
    },
    "1": {
      "precision": 0.8000,
      "recall": 0.8421,
      "f1-score": 0.8205,
      "support": 950
    }
  },
  "timestamp": "2025-11-25 19:00:00",
  "roc_curve": {
    "fpr": [0.0, 0.01, ...],
    "tpr": [0.0, 0.05, ...]
  },
  "precision_recall_curve": {
    "precision": [0.5, 0.51, ...],
    "recall": [1.0, 0.99, ...]
  }
}
```

**Nota Importante**: Cada llamada a este endpoint guarda automáticamente las métricas en la base de datos PostgreSQL.

**Códigos de Respuesta**:
- `200 OK`: Métricas calculadas exitosamente
- `500 Internal Server Error`: Error en el cálculo o guardado

## Estructura de Errores

Todos los errores siguen el formato:

```json
{
  "detail": "Descripción del error"
}
```

## Ejemplos de Uso

### Python con requests

```python
import requests

# Predicción
response = requests.post(
    "http://localhost:8000/api/v1/predict",
    json={
        "age": 59,
        "job": "admin.",
        # ... resto de campos
    }
)
result = response.json()
print(f"Predicción: {result['prediction']}, Probabilidad: {result['probability']}")

# Métricas
response = requests.get("http://localhost:8000/api/v1/metrics")
metrics = response.json()
print(f"Accuracy: {metrics['accuracy']}")
```

### cURL

```bash
# Health check
curl http://localhost:8000/api/v1/health

# Predicción
curl -X POST http://localhost:8000/api/v1/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 59,
    "job": "admin.",
    ...
  }'

# Métricas
curl http://localhost:8000/api/v1/metrics
```

