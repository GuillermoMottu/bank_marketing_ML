# Bank Marketing Decision Tree - Proyecto Completo

Implementación completa de un sistema de predicción de campañas bancarias usando **DecisionTreeClassifier** de scikit-learn, con API REST (FastAPI), base de datos PostgreSQL, dashboard de monitoreo (Dash), dockerización completa, tests y documentación profesional.

## 📋 Descripción del Proyecto

Este proyecto implementa un modelo de Machine Learning para predecir si un cliente suscribirá un depósito a término fijo basándose en información demográfica y de comportamiento. El sistema incluye:

- **Preprocesamiento completo** con Pipeline y ColumnTransformer
- **Modelo DecisionTreeClassifier** optimizado con GridSearchCV
- **API REST** con FastAPI para predicciones y métricas
- **Base de datos PostgreSQL** para almacenar histórico de métricas y predicciones
- **Dashboard interactivo** con Dash y Plotly para monitoreo
- **Prueba de hipótesis estadística** con prueba binomial exacta
- **Dockerización completa** para despliegue fácil
- **Guardado automático** de todas las predicciones en la base de datos
- **Manejo robusto de errores** con mensajes claros y códigos HTTP apropiados

## 🏗️ Estructura del Proyecto

```
bank-marketing-dt-project/
├── data/
│   ├── raw/                    # Datos originales (bank.csv)
│   └── processed/              # Datos preprocesados
├── notebooks/
│   ├── 01_eda.ipynb           # Análisis exploratorio
│   └── 02_hypothesis_test.ipynb  # Prueba de hipótesis
├── src/
│   ├── preprocessing/          # Pipeline de preprocesamiento
│   ├── model/                  # Entrenamiento y evaluación
│   ├── api/                    # API FastAPI
│   ├── database/               # Modelos y conexión PostgreSQL
│   ├── dashboard/              # Dashboard Dash
│   └── utils/                  # Utilidades y configuración
├── models/                     # Modelos entrenados (.pkl)
├── tests/                      # Tests unitarios e integración
├── docs/                       # Documentación adicional
├── docker/                     # Dockerfiles y docker-compose
├── requirements.txt            # Dependencias Python
├── train_model.py             # Script para entrenar el modelo
└── README.md                  # Este archivo
```

## 🚀 Instalación

### Requisitos Previos

- Python 3.11+
- Docker y Docker Compose (para despliegue con Docker)
- PostgreSQL 15+ (si se ejecuta sin Docker)

### Instalación Local

1. **Clonar el repositorio**:
```bash
git clone <repository-url>
cd bank-marketing-dt-project
```

2. **Crear entorno virtual**:
```bash
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate
```

3. **Instalar dependencias**:
```bash
pip install -r requirements.txt
```

4. **Configurar variables de entorno**:
```bash
cp .env.example .env
# Editar .env con tus configuraciones
```

5. **Entrenar el modelo** (primera vez):
```bash
python train_model.py
```

Esto creará:
- `models/preprocessing_pipeline.pkl`
- `models/decision_tree_model.pkl`

**⚠️ Nota importante**: Si planeas usar Docker, es recomendable entrenar el modelo dentro del contenedor para asegurar compatibilidad de versiones:
```bash
docker-compose up -d api
docker exec bank_marketing_api python train_model.py
```

## 🐳 Ejecución con Docker (Recomendado)

El método más simple para ejecutar todo el sistema es usando Docker Compose:

```bash
docker compose up --build
```

Esto iniciará automáticamente:
- **PostgreSQL** en el puerto 5432
- **API FastAPI** en el puerto 8000
- **Dashboard Dash** en el puerto 8050

### Acceso a los Servicios

- **API**: http://localhost:8000
- **Documentación API**: http://localhost:8000/docs
- **Dashboard**: http://localhost:8050

## 📡 Endpoints de la API

### POST `/api/v1/predict`

Realiza una predicción para un cliente. **Cada predicción se guarda automáticamente en la base de datos**.

**Request Body**:
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
  "probability": 0.85,
  "class_name": "yes"
}
```

**Nota**: Cada predicción se guarda automáticamente en la tabla `prediction_history` con los datos de entrada, resultado y timestamp.

### GET `/api/v1/metrics`

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
  "classification_report": {...},
  "timestamp": "2025-11-25 19:00:00",
  "roc_curve": {...},
  "precision_recall_curve": {...}
}
```

**Nota**: Cada llamada a este endpoint guarda automáticamente las métricas en PostgreSQL.

### GET `/api/v1/health`

Health check de la API.

**Response**:
```json
{
  "status": "healthy",
  "timestamp": "2025-11-25 19:00:00"
}
```

## 📊 Dashboard

El dashboard proporciona una interfaz web completa para:

### Pestaña de Métricas

- **Tarjetas de métricas actuales**: Accuracy, Precision, Recall, F1-Score, AUC-ROC
- **Gráficos históricos**: Evolución de todas las métricas a lo largo del tiempo
- **Matriz de confusión interactiva**: Visualización en heatmap
- **Curvas ROC y Precision-Recall**: Del último cálculo de métricas
- **Tabla histórica completa**: Todas las métricas guardadas en la base de datos

### Pestaña de Predicción

- **Formulario interactivo**: Para ingresar características del cliente
- **Predicción en tiempo real**: Muestra probabilidad y clase predicha
- **Validación de entrada**: Asegura datos correctos antes de predecir
- **Guardado automático**: Cada predicción se guarda automáticamente en la base de datos

El dashboard se actualiza automáticamente cada 30 segundos y maneja correctamente las operaciones concurrentes de base de datos.

**Consejos para mejores resultados:**
- Usar `default="no"` y `loan="no"` aumenta significativamente las probabilidades
- Contactos por `cellular` tienen mejor tasa de éxito
- Duración alta (>400 segundos) indica mayor interés
- Si hay historial previo, `poutcome="success"` es muy positivo

## 📈 Prueba de Hipótesis Estadística

El proyecto incluye una prueba de hipótesis estadística completa para validar que el modelo es significativamente mejor que el azar:

### Hipótesis

- **H0**: Accuracy ≤ 0.5 (rendimiento no mejor que el azar)
- **H1**: Accuracy > 0.5 (modelo es mejor que el azar)

### Implementación

Se utiliza una **prueba binomial exacta** (`scipy.stats.binomtest`) con:

- Nivel de significancia: α = 0.05
- Alternativa: one-sided (greater)
- P-value calculado y decisión automática

### Errores Tipo I y Tipo II

- **Error Tipo I (α)**: Rechazar H0 cuando es verdadera → Concluir que el modelo es mejor cuando no lo es
- **Error Tipo II (β)**: No rechazar H0 cuando H1 es verdadera → No detectar que el modelo es útil

Ver el notebook `notebooks/02_hypothesis_test.ipynb` para detalles completos.

## 🧪 Tests

Ejecutar los tests:

```bash
pytest tests/ -v
```

Con cobertura:

```bash
pytest tests/ --cov=src --cov-report=html
```

La cobertura mínima objetivo es **80%**.

## 🔧 Configuración

### Variables de Entorno

Crear un archivo `.env` con:

```env
# Database
DB_HOST=db
DB_PORT=5432
DB_NAME=bank_marketing
DB_USER=postgres
DB_PASSWORD=postgres

# API
API_HOST=0.0.0.0
API_PORT=8000

# Dashboard
DASHBOARD_HOST=0.0.0.0
DASHBOARD_PORT=8050
```

## 📚 Uso del Modelo

### Entrenamiento

Para entrenar o reentrenar el modelo:

```bash
python train_model.py
```

El script realiza:
1. Carga de datos desde `data/raw/bank.csv`
2. Creación y ajuste del pipeline de preprocesamiento
3. División train/test estratificada (80/20)
4. Optimización de hiperparámetros con GridSearchCV (cv=10)
5. Guardado del modelo y pipeline

### Predicción desde Python

```python
from src.api.dependencies import ModelService

# Cargar servicio
service = ModelService()

# Hacer predicción
input_data = {
    "age": 59,
    "job": "admin.",
    # ... resto de campos
}
prediction, probability = service.predict(input_data)
print(f"Predicción: {prediction}, Probabilidad: {probability}")
```

## 🗄️ Base de Datos

La base de datos PostgreSQL utiliza SQLAlchemy async y almacena automáticamente:

### Tabla `metrics_history`

Almacena el histórico de métricas del modelo:

```sql
CREATE TABLE metrics_history (
    id SERIAL PRIMARY KEY,
    model_name VARCHAR(255),
    metrics_json JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

**Características:**
- Cada vez que se llama al endpoint `/metrics`, se crea un nuevo registro
- Almacena métricas completas: accuracy, precision, recall, F1-score, AUC-ROC, matriz de confusión, curvas ROC y Precision-Recall
- Permite seguimiento histórico del rendimiento del modelo

### Tabla `prediction_history`

Almacena el histórico de todas las predicciones realizadas:

```sql
CREATE TABLE prediction_history (
    id SERIAL PRIMARY KEY,
    input_data JSONB NOT NULL,
    prediction INTEGER NOT NULL,
    probability VARCHAR(20) NOT NULL,
    class_name VARCHAR(10) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

**Características:**
- Guardado automático de cada predicción realizada
- Incluye datos de entrada completos del cliente en formato JSON
- Permite auditoría y análisis posterior del comportamiento del modelo
- Facilita el seguimiento de predicciones exitosas vs. fallidas

**Pool de conexiones:**
- Configurado para manejar múltiples operaciones concurrentes
- `pool_pre_ping=True` para verificar conexiones vivas
- Reciclaje automático de conexiones

### Consultar Historial

Para consultar el historial de predicciones desde Python:

```python
import asyncio
from src.database.connection import AsyncSessionLocal
from src.database.crud import get_all_predictions

async def consultar_predicciones():
    async with AsyncSessionLocal() as session:
        predicciones = await get_all_predictions(session, limit=100)
        for pred in predicciones[:10]:
            print(f"ID: {pred.id}, Predicción: {pred.class_name}, "
                  f"Probabilidad: {pred.probability}, Fecha: {pred.created_at}")

asyncio.run(consultar_predicciones())
```

O directamente desde PostgreSQL:

```sql
-- Ver últimas 10 predicciones
SELECT id, prediction, probability, class_name, created_at 
FROM prediction_history 
ORDER BY created_at DESC 
LIMIT 10;

-- Contar predicciones por clase
SELECT class_name, COUNT(*) 
FROM prediction_history 
GROUP BY class_name;

-- Ver predicciones con alta probabilidad
SELECT * FROM prediction_history 
WHERE probability::float > 0.8 
ORDER BY created_at DESC;
```

## 🔍 Preprocesamiento

El pipeline de preprocesamiento incluye:

1. **Manejo de valores "unknown"**: Convertidos a categorías propias (`unknown_<columna>`)
2. **One-Hot Encoding**: Para variables categóricas (job, marital, education, etc.)
3. **RobustScaler**: Para variables numéricas (age, balance, duration, etc.)
4. **ColumnTransformer**: Organiza las transformaciones por tipo de columna

## 🎯 Hiperparámetros Optimizados

El GridSearchCV optimiza:

- `max_depth`: [5, 10, 15, 20, None]
- `min_samples_split`: [2, 5, 10, 20]
- `min_samples_leaf`: [1, 2, 4, 8]
- `criterion`: ['gini', 'entropy']
- `class_weight`: [None, 'balanced']

Con validación cruzada de **10 folds** y scoring **ROC-AUC**.

## 📝 Notas Técnicas

- **Reproducibilidad**: `random_state=42` en todas las operaciones aleatorias
- **División estratificada**: Para mantener proporciones de clases en train/test
- **Validación cruzada**: 10 folds para GridSearchCV
- **Scoring**: ROC-AUC (mejor para datasets potencialmente desbalanceados)
- **Manejo de errores**: Sistema robusto con códigos HTTP apropiados (500 para errores internos, 503 para servicio no disponible)
- **Operaciones asíncronas**: Base de datos con SQLAlchemy async para mejor rendimiento
- **Isolación de event loops**: Dashboard con manejo correcto de operaciones asíncronas en contexto síncrono

## 🐛 Troubleshooting

### El modelo no se encuentra

**Opción 1 - Entrenar localmente:**
```bash
python train_model.py
```

**Opción 2 - Entrenar dentro del contenedor Docker (recomendado):**
```bash
docker exec bank_marketing_api python train_model.py
```

Esto asegura que el modelo se entrene con la misma versión de scikit-learn que usa la API.

### Error 503 Service Unavailable en predicciones

Este error indica que el modelo no está disponible. Verifica:
1. Que el modelo existe en `models/decision_tree_model.pkl`
2. Que el pipeline existe en `models/preprocessing_pipeline.pkl`
3. Si estás usando Docker, entrena el modelo dentro del contenedor para evitar problemas de versiones

### Error de conexión a la base de datos

Verifica que PostgreSQL esté corriendo y que las variables de entorno estén correctamente configuradas:

```bash
docker ps  # Verificar que el contenedor de la BD esté corriendo
docker logs bank_marketing_db  # Ver logs de la base de datos
```

### Dashboard no muestra datos

1. Asegúrate de haber llamado al menos una vez al endpoint `/metrics` para generar datos históricos
2. El dashboard se actualiza automáticamente cada 30 segundos
3. Si hay errores de concurrencia, los contenedores ya están configurados para manejarlos correctamente

### Error de versiones de scikit-learn

Si entrenaste el modelo localmente y obtienes errores al cargarlo en Docker:
- Entrena el modelo dentro del contenedor Docker para asegurar compatibilidad de versiones
- Ejecuta: `docker exec bank_marketing_api python train_model.py`

### Reconstruir imágenes Docker después de cambios en el código

Si has hecho cambios en el código y necesitas reconstruir las imágenes:

```bash
docker-compose down
docker-compose build --no-cache
docker-compose up -d
```

## 📄 Licencia

Este proyecto es académico y cumple con los requisitos del documento sprint.pdf y las especificaciones del profesor.

## 👥 Autor

Proyecto desarrollado como parte de un curso académico de Machine Learning y MLOps.

---

## ✨ Características Adicionales

### Guardado Automático de Predicciones

Todas las predicciones se guardan automáticamente en la base de datos, permitiendo:
- Auditoría completa de todas las predicciones realizadas
- Análisis de patrones en las predicciones
- Seguimiento de la efectividad del modelo en producción
- Análisis de qué características llevan a predicciones exitosas

### Manejo Robusto de Errores

El sistema incluye manejo avanzado de errores:
- Verificación de disponibilidad del modelo antes de predecir
- Mensajes de error claros y descriptivos
- Códigos HTTP apropiados (503 para servicio no disponible)
- Logging detallado para debugging

### Compatibilidad de Versiones

El sistema detecta y previene problemas de compatibilidad:
- Verificación de versiones de scikit-learn
- Recomendación de entrenar el modelo dentro del contenedor Docker
- Mensajes claros cuando el modelo no está disponible

---

**¡El proyecto está listo para ejecutarse con `docker compose up --build`!** 🚀

**Nota**: Asegúrate de entrenar el modelo dentro del contenedor Docker para evitar problemas de compatibilidad de versiones.

