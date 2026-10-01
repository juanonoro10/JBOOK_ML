# Heart Disease MLOps

Proyecto integrador de Machine Learning. Juan Camilo Oñoro Araujo (200177329) y María Carolina Cantillo Orozco (200179105). Universidad del Norte, 2026.

El objetivo es construir, evaluar, desplegar y monitorear un modelo que predice si un paciente tiene enfermedad cardíaca (heartdisease = 1) a partir de 11 variables clínicas, aplicando prácticas de MLOps en un entorno local. El dataset es el Heart Failure Prediction Dataset de Kaggle: 918 pacientes de 5 bases de datos clínicas (Cleveland, Hungría, Suiza, Long Beach VA y Statlog).

## Resultados

| Métrica | Valor |
|---|---|
| Modelo final | RandomForestClassifier (elegido por validación cruzada entre 5 familias y 90 combinaciones) |
| AUC de validación cruzada | 0.934 |
| AUC de validación cruzada anidada | 0.926 ± 0.012 |
| AUC en test (184 pacientes) | 0.934, con IC 95 % de 0.893 a 0.968 |
| Accuracy en test | 0.891 |
| Sensibilidad y especificidad (umbral 0.5) | 0.941 y 0.829 |

## Arquitectura y tecnologías

| Herramienta | Para qué se usa |
|---|---|
| scikit-learn | Pipeline de preprocesamiento y modelo, GridSearchCV y validación cruzada |
| FastAPI y uvicorn | API REST que sirve las predicciones |
| Docker | Empaquetar la API con sus dependencias exactas en una imagen |
| Kubernetes (Minikube) | Desplegar y mantener vivo el contenedor localmente |
| GitHub Actions | Revisar el estilo (flake8) y correr las pruebas (pytest) en cada push |
| Evidently | Reportes de deriva de datos entre entrenamiento y producción |

## Estructura

```
heart-disease-mlops/
├── app/
│   ├── api.py                    API con FastAPI
│   ├── model.joblib              Pipeline entrenado (preprocesamiento + modelo)
│   └── model_metadata.json       Columnas, categorías, métricas y versión de scikit-learn
├── data/
│   ├── heart.csv                 Dataset original
│   ├── train.csv                 80 % de entrenamiento (referencia del monitoreo)
│   └── test.csv                  20 % de prueba
├── docker/
│   ├── Dockerfile
│   └── requirements.txt          Dependencias mínimas de la imagen, con versiones fijas
├── k8s/
│   ├── deployment.yaml
│   └── service.yaml
├── notebooks/
│   ├── 1_model_leakage_demo.ipynb    Etapa 1: EDA, preprocesamiento, data leakage y comparación de modelos
│   ├── 2_model_pipeline_cv.ipynb     Etapa 2: modelo final con validación segura y exportación
│   └── 3_monitoreo_drift.ipynb       Etapa 6: reportes de deriva con Evidently
├── tests/
│   ├── test_api.py
│   └── test_model.py
├── figuras/                      Gráficas generadas por los notebooks
├── .github/workflows/ci.yml
├── drift_report.html             Reporte de deriva train contra test
├── drift_report_simulado.html    Reporte de deriva con un escenario simulado
├── requirements.txt              Entorno de desarrollo (notebooks y pruebas)
└── README.md
```

## Etapas del proyecto

### Etapa 0. Estructura

Qué se hace: se crea una carpeta para cada parte del ciclo de vida del modelo (análisis, API, contenedor, orquestación, CI y monitoreo).

Por qué: separar las responsabilidades permite cambiar una parte sin romper las demás. Por ejemplo, la imagen de Docker solo copia app/ y docker/requirements.txt, no los notebooks ni los datos.

### Etapa 1. Análisis exploratorio y data leakage

Qué se hace: en el notebook 1 se analizan las variables, los valores faltantes, los outliers y la relación de cada variable con la enfermedad. Después se demuestra el efecto de la fuga de datos y se comparan 7 modelos con Pipeline y GridSearchCV.

Por qué: el EDA encontró que el 18.7 % de los valores de cholesterol son ceros imposibles y que no faltan al azar (el 88.4 % de esos pacientes está enfermo). Sin detectarlo, el modelo aprendería de datos falsos. La demostración de fuga muestra por qué todo el preprocesamiento debe ir dentro del Pipeline: una selección de variables hecha antes de la validación cruzada da AUC = 0.716 con variables que son ruido puro, cuando la respuesta correcta es 0.5.

### Etapa 2. Entrenamiento seguro

Qué se hace: en el notebook 2 se separan train y test antes de cualquier ajuste, se elige el modelo por validación cruzada, se evalúa una sola vez en test (matriz de confusión, curva ROC y AUC) y se exporta el Pipeline completo a app/model.joblib.

Por qué: si test se usara para escoger el modelo, dejaría de ser una evaluación independiente. El Pipeline exportado incluye la imputación y el escalado, así que recibe los datos crudos del paciente y la API no tiene que repetir ninguna transformación.

### Etapa 3. API con FastAPI y Docker

Qué se hace: app/api.py carga el modelo y expone el endpoint POST /predict, que recibe los datos de un paciente y devuelve la probabilidad de enfermedad y la predicción. La imagen de Docker empaqueta la API con las mismas versiones de librerías con las que se entrenó el modelo.

Por qué: la API permite usar el modelo desde cualquier aplicación sin instalar Python ni scikit-learn. Docker garantiza que funcione igual en cualquier equipo. Esto importa porque un archivo joblib solo carga de forma confiable con la misma versión de scikit-learn (1.3.0).

Endpoints:

| Método y ruta | Qué hace |
|---|---|
| GET /health | Verifica que la API está viva (lo usa Kubernetes) |
| GET /model-info | Modelo, hiperparámetros y métricas |
| POST /predict | Predicción para un paciente |
| GET /docs | Documentación interactiva para probar la API desde el navegador |

Ejemplo de petición a /predict:

```json
{
  "Age": 58, "Sex": "M", "ChestPainType": "ASY", "RestingBP": 140, "Cholesterol": 289,
  "FastingBS": 1, "RestingECG": "ST", "MaxHR": 110, "ExerciseAngina": "Y", "Oldpeak": 2.0, "ST_Slope": "Flat"
}
```

Respuesta:

```json
{"heart_disease_probability": 0.9746, "prediction": 1}
```

Un 0 en Cholesterol o RestingBP se interpreta como dato no registrado y el modelo lo imputa con la mediana de entrenamiento.

### Etapa 4. Orquestación con Kubernetes

Qué se hace: k8s/deployment.yaml le indica a Kubernetes qué imagen correr y cuántas copias mantener. k8s/service.yaml da una dirección fija para llegar a la API. El Deployment revisa /health para saber cuándo el contenedor está listo y para reiniciarlo si deja de responder.

Por qué: Docker ejecuta un contenedor, pero no lo vuelve a levantar si se cae ni reparte el tráfico entre varias copias. Kubernetes se encarga de eso, que es lo que se necesita en producción.

### Etapa 5. Integración continua con GitHub Actions

Qué se hace: en cada push, .github/workflows/ci.yml instala las dependencias, revisa el estilo con flake8 y corre 15 pruebas con pytest. Las pruebas verifican que la API responde bien, que rechaza datos inválidos, que el modelo carga con la versión correcta de scikit-learn y que su AUC en test no baja de 0.90.

Por qué: así un cambio que rompa la API o empeore el modelo se detecta automáticamente antes de desplegarlo, sin depender de que alguien lo pruebe a mano.

### Etapa 6. Monitoreo con Evidently

Qué se hace: en el notebook 3 se compara la distribución de cada variable y de las predicciones entre train y los datos nuevos, y se genera drift_report.html. También se simula un cambio de población (pacientes mayores, más colesterol sin registrar y más dolor asintomático) para comprobar que el monitoreo lo detecta.

Por qué: en producción no se conoce el diagnóstico real de inmediato, así que no se puede medir el accuracy día a día. Si los pacientes que llegan son distintos a los de entrenamiento, el modelo puede volverse poco confiable sin avisar. Comparar las distribuciones es la señal temprana para revisar o reentrenar el modelo.

## Cómo ejecutarlo

Todos los comandos se ejecutan desde la carpeta heart-disease-mlops.

1. Entorno de desarrollo:

```bash
pip install -r requirements.txt
```

2. Notebooks, en orden: 1_model_leakage_demo, 2_model_pipeline_cv (genera app/model.joblib y data/train.csv y data/test.csv) y 3_monitoreo_drift (genera drift_report.html).

3. API local sin Docker:

```bash
uvicorn app.api:app --port 8000
```

Luego se abre http://127.0.0.1:8000/docs.

4. Docker:

```bash
docker build -t heart-api:1.0 -f docker/Dockerfile .
docker run -p 8000:8000 heart-api:1.0
```

5. Kubernetes con Minikube:

```bash
minikube start
minikube image load heart-api:1.0
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
kubectl get pods
kubectl get svc
minikube service heart-service --url
```

El último comando devuelve la URL de la API. En Windows con el driver de Docker, ese comando abre un túnel y la terminal debe quedar abierta mientras se usa la API. Una alternativa es reenviar el puerto del Service y abrir http://127.0.0.1:8080/docs:

```bash
kubectl port-forward svc/heart-service 8080:80
```

Si se prefiere Docker Hub, se sube la imagen como TU_USUARIO/heart-api:1.0 y se cambia la línea image en deployment.yaml.

Para apagar todo al terminar:

```bash
kubectl delete -f k8s/
minikube stop
```

6. Pruebas y estilo, igual que en el CI:

```bash
flake8 app/ tests/
pytest tests/ -v
```

## Limitaciones

1. El dataset tiene 918 pacientes y el conjunto de prueba 184. Por eso el intervalo de confianza del AUC es amplio (±0.04) y las diferencias entre los mejores modelos no son significativas.
2. El indicador de colesterol faltante refleja en parte el hospital de origen y no solo la fisiología del paciente. Un hospital nuevo que sí registre el colesterol podría cambiar el comportamiento del modelo.
3. El umbral de 0.5 produce 14 falsos positivos de 82 pacientes sanos en test. El notebook 2 propone un umbral alternativo de 0.55, guardado en los metadatos.
4. El modelo es un ejercicio académico y no reemplaza un diagnóstico médico.
