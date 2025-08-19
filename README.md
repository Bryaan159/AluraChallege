# README.md: Telecom X - Parte 2: Análisis de Evasión de Clientes

## Propósito del Análisis

El objetivo principal de este proyecto es predecir la evasión (churn o cancelación) de clientes en la empresa Telecom X, utilizando variables relevantes como el tiempo de contrato (`Meses_Contratados`), el gasto total (`Cobro_Total`), el tipo de contrato (`Tipo_Contrato`) y otros factores demográficos y de servicio. A través de un análisis exploratorio de datos (EDA), preparación de datos y modelado predictivo, se busca identificar patrones que influyen en la cancelación para proponer estrategias de retención. Esto permite a la empresa reducir la tasa de evasión (~26.5% según el dataset), mejorar la fidelidad de los clientes y optimizar recursos.

El análisis se basa en datos de clientes (cargados desde un JSON), con énfasis en modelos de machine learning como Regresión Logística y Random Forest para predecir `Evasion` (1: sí, 0: no).

## Estructura del Proyecto

El proyecto está organizado de la siguiente manera para facilitar la reproducibilidad y la navegación:

- **challengex.py**: El script principal en Python que incluye la extracción de datos, limpieza, EDA, preprocesamiento, modelado y evaluación. Es el corazón del análisis.
- **TelecomX_Clean.csv**: Archivo CSV generado con los datos limpios y procesados (después de manejar nulos, codificación y creación de columnas derivadas como `Cuentas_Diarias`).
- **visualizaciones/**: Carpeta que contiene gráficos generados durante el EDA y la evaluación (por ejemplo, boxplots, scatter plots, matrices de correlación y matrices de confusión). Ejemplos: `distribucion_churn.png`, `matriz_correlacion.png`.
- **README.md**: Este archivo, que describe el proyecto, el proceso y las instrucciones de ejecución.
- **requirements.txt**: Lista de dependencias (bibliotecas necesarias para ejecutar el script).

El repositorio sigue una estructura estándar: datos en root, scripts en root, y outputs en subcarpetas.

## Proceso de Preparación de los Datos

La preparación de datos es crucial para garantizar la calidad y el rendimiento de los modelos. Se realizó lo siguiente:

### Clasificación de Variables
- **Variables Numéricas**: `Meses_Contratados` (tenure), `Cobro_Mensual` (Charges.Monthly), `Cobro_Total` (Charges.Total), `Cuentas_Diarias` (derivada de Cobro_Mensual / 30). Estas representan métricas cuantitativas como tiempo y gastos.
- **Variables Categóricas**: `gender`, `Partner`, `Dependents`, `PhoneService`, `MultipleLines`, `InternetService` (`Servicio_Internet`), `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`, `Contract` (`Tipo_Contrato`), `PaperlessBilling`, `PaymentMethod` (`Metodo_Pago`). Estas capturan atributos cualitativos como servicios y métodos de pago.

### Etapas de Normalización o Codificación
- **Manejo de Nulos**: Se identificaron 11 valores nulos en `Cobro_Total`, convertidos a numéricos con `pd.to_numeric(errors='coerce')` y eliminados o imputados (mediana para numéricas, moda para categóricas en etapas posteriores).
- **Codificación**: 
  - Binarias (e.g., `Partner`, `Dependents`): Mapeadas a 0/1 (`No` → 0, `Yes` → 1).
  - Categóricas multi-clase (e.g., `Tipo_Contrato`, `Metodo_Pago`): Codificadas con `OneHotEncoder(drop='first')` para evitar multicolinealidad.
- **Normalización/Estandarización**: Aplicada solo a numéricas en modelos sensibles (e.g., Regresión Logística con `StandardScaler` para media=0, desviación=1). No se aplica en Random Forest, ya que no es sensible a la escala.
- **Balanceo Opcional**: Se exploró undersampling (reducir clase mayoritaria) y oversampling (duplicar minoritaria) para manejar el desbalance de `Evasion` (~26.5% positivos).

### Separación de Datos
- División 80/20: 80% para entrenamiento (~5626 filas), 20% para prueba (~1406 filas), usando `train_test_split(stratify=y)` para mantener la proporción de `Evasion`.
- Justificación: Evita data leakage (preprocesamiento se ajusta solo en entrenamiento) y asegura evaluación representativa.

Estos pasos aseguran datos limpios, escalados y listos para modelado, reduciendo sesgos y mejorando la generalización.

## Justificaciones para las Decisiones en Modelización

- **Elección de Modelos**: 
  - Regresión Logística: Simple, interpretable y sensible a escalas (por eso estandarización). Justificación: Buena para relaciones lineales y baseline para churn binario.
  - Random Forest: Robusto a desbalance y no linealidades, no requiere normalización. Justificación: Captura interacciones complejas (e.g., `Meses_Contratados` con `Tipo_Contrato`).
- **Métricas de Evaluación**: Exactitud, precisión, recall, F1-score y matriz de confusión. Justificación: F1-score priorizado por desbalance (evita sobrevalorar exactitud).
- **Manejo de Overfitting/Underfitting**: Validación cruzada (CV=5) para detectar. Regresión Logística mostró underfitting (simplicidad); Random Forest leve overfitting (complejidad). Ajustes: `class_weight='balanced'`, limitar profundidad en RF.
- **Balanceo**: Opcional undersampling/oversampling justificado por desbalance, mejorando recall para `Evasion=1`.
- **Preprocesamiento**: Estandarización en LR para evitar dominio de variables con rangos grandes (e.g., `Cobro_Total`). No en RF para mantener eficiencia.

Estas decisiones se basan en el EDA (e.g., correlaciones altas en `Meses_Contratados`) y el desbalance, priorizando interpretabilidad y rendimiento.

## Ejemplos de Gráficos e Insights Obtenidos en EDA

El EDA reveló patrones clave, con gráficos guardados en `visualizaciones/`:

- **Gráfico de Barras/Pastel de Distribución de Churn**: 
  - Insight: ~73.5% permanecen, ~26.5% cancelan. Alta evasión en contratos mensuales (~42%) vs. bienales (~2%).
  - Ejemplo: `distribucion_churn.png` muestra barras con conteos (5163 No, 1869 Sí).

- **Matriz de Correlación**:
  - Insight: `Meses_Contratados` correlacionado negativamente con `Evasion` (-0.35); `Cobro_Total` (-0.20). Multicolinealidad entre `Cobro_Total` y `Meses_Contratados` (~0.83).
  - Ejemplo: Heatmap en `matriz_correlacion.png`.

- **Boxplots y Scatter Plots**:
  - Boxplot `Meses_Contratados` vs `Evasion`: Mediana baja para cancelaciones (~10 meses), alta para permanencia (~37). Insight: Evasión temprana.
  - Scatter `Meses_Contratados` vs `Cobro_Total`: Puntos de evasión en bajo tiempo/gasto. Insight: Clientes con <20 meses y <1000 gasto son de alto riesgo.
  - Ejemplo: `boxplot_tenure_churn.png`, `scatter_tenure_total.png`.

Insights generales: Contratos mensuales, fibra óptica y pagos no automáticos aumentan evasión; mayor tiempo y gasto reducen riesgo.

## Instrucciones para Ejecutar el Cuaderno

### Requisitos Previos
- Python 3.8+.
- Instala las bibliotecas necesarias ejecutando:
  ```
  pip install -r requirements.txt
  ```
  Contenido de `requirements.txt`:
  ```
  pandas
  numpy
  scikit-learn
  matplotlib
  seaborn
  ```
- Descarga el dataset original (TelecomX_Data.json) desde el enlace proporcionado en el script.

### Pasos para Ejecutar
1. Clona el repositorio o descarga los archivos.
2. Abre el script `challengex.py` en un editor como VS Code o Jupyter Notebook (convierte a .ipynb si es necesario).
3. Ejecuta el script paso a paso:
   - Carga los datos: Usa `requests.get(url)` para obtener el JSON y convertir a DataFrame.
   - Genera `TelecomX_Clean.csv` ejecutando la sección de limpieza.
   - Para modelado: Carga `TelecomX_Clean.csv` y ejecuta las secciones de preprocesamiento y entrenamiento.
4. Visualizaciones: Se generan automáticamente y se guardan en `visualizaciones/`.
5. Notas: Si hay errores de nulos, verifica `TelecomX_Clean.csv`. Ejecuta en un entorno como Google Colab para gráficos interactivos.

Este README hace el proyecto reproducible y profesional. Para dudas, contacta al autor.