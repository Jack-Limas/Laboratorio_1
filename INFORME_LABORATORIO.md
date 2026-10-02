# Informe de Laboratorio 1: Minería de Datos & Business Intelligence
**Marco Metodológico: CRISP-DM (Cross-Industry Standard Process for Data Mining)**  
**Asignatura:** Inteligencia de Negocios y Minería de Datos  
**Institución:** Universidad Cooperativa de Colombia  

---

## 1. Introducción y Metodología CRISP-DM

El presente laboratorio aborda el modelado predictivo y el análisis cuantitativo sobre tres fenómenos de estudio: el comportamiento financiero del tipo de cambio (precio del dólar), variables biomédicas determinantes en la regulación glucémica, y patrones de demanda en el consumo de energía eléctrica.

Para garantizar rigor, reproducibilidad e integración con el negocio, se implementó el ciclo de vida del estándar **CRISP-DM**:

```
 ┌──────────────────────┐
 │ Comprensión Negocio  │◄──┐
 └──────────┬───────────┘   │
            ▼               │
 ┌──────────────────────┐   │
 │ Comprensión Datos    │───┤
 └──────────┬───────────┘   │
            ▼               │
 ┌──────────────────────┐   │
 │ Preparación Datos    │───┤
 └──────────┬───────────┘   │
            ▼               │
 ┌──────────────────────┐   │
 │      Modelado        │───┤
 └──────────┬───────────┘   │
            ▼               │
 ┌──────────────────────┐   │
 │     Evaluación       │───┘
 └──────────┬───────────┘
            ▼
 ┌──────────────────────┐
 │     Despliegue       │
 └──────────────────────┘
```

1. **Comprensión del Negocio (Business Understanding):** Definición de objetivos analíticos para formular modelos de Regresión Lineal Múltiple que permitan la estimación precisa y la toma de decisiones informada.
2. **Comprensión de los Datos (Data Understanding):** Exploración de las fuentes estructuradas (`dolar_data.csv`, `glucosa_data.csv`, `energia_data.csv`), identificando escalas, distribuciones y relaciones bivariadas preliminares.
3. **Preparación de los Datos (Data Preparation):** Validación de consistencia tipológica, ausencia de valores nulos o corruptos, y segmentación entre matrices de diseño de covariables independientes ($X$) y vectores continuos de respuesta ($Y$).
4. **Modelado (Modeling):** Ajuste de modelos por Mínimos Cuadrados Ordinarios (OLS - *Ordinary Least Squares*) calculando coeficientes de ponderación $\beta_j$ e intercepto $\beta_0$.
5. **Evaluación (Evaluation):** Análisis de bondad de ajuste ($R^2$), error cuadrático medio (MSE), raíz del error cuadrático medio (RMSE) e impacto estandarizado (Betas).
6. **Despliegue (Deployment):** Persistencia de los artefactos entrenados mediante serialización con `joblib` y desarrollo de un cuadro de mando web interactivo en **Streamlit** respaldado por un pipeline de orquestación `run_all.py`.

---

## 2. Desarrollo Técnico por Ejercicio

### Ejercicio 1: Predicción del Precio del Dólar

#### A. Ecuación Paramétrica del Modelo
$$\text{Precio\_Dolar} = 3978.9846 + 4.9991 \cdot \text{Dia} - 338.0598 \cdot \text{Inflacion} - 2.5328 \cdot \text{Tasa\_interes}$$

#### B. Interpretación Detallada de Coeficientes
* **Intercepto ($\beta_0 = 3978.9846$ COP):** Representa el valor base esperado del dólar cuando el día del mes es cero (extrapolación teórica al origen) y tanto la inflación como la tasa de interés son nulas.
* **$\beta_1$ (Día = $+4.9991$):** Por cada día transcurrido dentro del periodo analizado, el precio del dólar experimenta un incremento promedio constante de **~5.00 COP**, manteniendo invariables la inflación y la tasa de interés. Denota una tendencia alcista secular temporal dentro de la serie muestral.
* **$\beta_2$ (Inflación = $-338.0598$):** Un aumento unitario absoluto en la tasa de inflación registrada generaría una contracción de 338.06 COP en el tipo de cambio. En términos porcentuales (ej. un incremento de 0.01 o 1%), la variación marginal observada es de aproximadamente **-3.38 COP**.
* **$\beta_3$ (Tasa de Interés = $-2.5328$):** Por cada incremento de un punto porcentual (1.0%) en la tasa de interés de referencia fijada por la autoridad monetaria, el precio del dólar se deprecia en **2.53 COP**, reflejando la atracción de capitales de corto plazo que aprecian la moneda local frente a la divisa extranjera.

#### C. Métricas de Evaluación
| Métrica | Valor Numérico | Interpretación |
| :--- | :---: | :--- |
| **$R^2$ (Bondad de Ajuste)** | **0.9952** | El modelo explica el **99.52%** de la varianza total observada en el precio del dólar. Ajuste predictivo casi perfecto. |
| **MSE** | **2524.7152** | Promedio de las desviaciones cuadráticas residuales. |
| **RMSE** | **50.2465 COP** | Desviación promedio del pronóstico respecto al valor real. En una moneda que cotiza alrededor de los 4,000 COP, representa un error relativo de apenas el **1.25%**. |

#### D. Análisis de Ponderación e Impacto
Al estandarizar los coeficientes para neutralizar las escalas dispares (los días van de 1 a 31, las tasas en porcentaje y la inflación en decimales):
* **Coeficiente Beta Estandarizado del Día:** **0.9978**
* **Coeficiente Beta Estandarizado de la Inflación:** **-0.0023**
* **Coeficiente Beta Estandarizado de la Tasa:** **-0.0017**

**Conclusión:** La variable **`Dia` (tiempo/tendencia)** posee el mayor peso e impacto absoluto en el modelo. Explica de forma determinante la trayectoria del precio en el dataset analizado.

---

### Ejercicio 2: Predicción de Niveles de Glucosa

#### A. Ecuación Paramétrica del Modelo
$$\text{Nivel\_Glucosa} = 66.3150 + 1.2337 \cdot \text{Edad} + 0.8832 \cdot \text{IMC} - 2.0104 \cdot \text{Actividad\_Fisica}$$

#### B. Interpretación de Coeficientes y Relevancia Clínica
* **Intercepto ($\beta_0 = 66.3150$ mg/dL):** Corresponde a la glucosa basal fisiológica teórica de un individuo hipotético sin edad acumulada, IMC nulo y sin ejercicio.
* **$\beta_1$ (Edad = $+1.2337$):** Por cada año biológico adicional de vida del paciente, la glucosa en sangre tiende a aumentar en promedio **1.23 mg/dL**. Este comportamiento modela con precisión la progresiva disminución en la sensibilidad a la insulina asociada al envejecimiento celular.
* **$\beta_2$ (IMC = $+0.8832$):** Por cada unidad de incremento en el Índice de Masa Corporal (kg/m²), los niveles de glucemia se incrementan en **0.88 mg/dL**, validando el impacto del exceso de tejido adiposo en la resistencia periférica a la insulina.
* **$\beta_3$ (Actividad Física = $-2.0104$):** Cada hora adicional de ejercicio físico a la semana reduce la glucosa en **2.01 mg/dL**.

#### C. Métricas de Evaluación
| Métrica | Valor Numérico | Interpretación |
| :--- | :---: | :--- |
| **$R^2$** | **0.6855** | Las tres covariables explican el **68.55%** de la variabilidad biológica de la glucosa en sangre. |
| **MSE** | **235.0377** | Magnitud cuadrática media del error residual. |
| **RMSE** | **15.3309 mg/dL** | Margen de dispersión media de las predicciones clínicas. Adecuado para tamizajes preliminares. |

#### D. Determinación de Impacto Extremo Positivo y Negativo
* **Mayor Impacto Positivo (Factor de Riesgo Principal):** **`Edad` ($\beta = +1.2337$)**, seguida de cerca por el `IMC`. Son los inductores dominantes de la elevación glucémica.
* **Mayor Impacto Negativo (Factor Protector Clave):** **`Actividad_Fisica` ($\beta = -2.0104$)**. Constituye la palanca conductual modificable más potente para contrarrestar la hiperglucemia.

---

### Ejercicio 3: Consumo de Energía Eléctrica

#### A. Ecuación Paramétrica del Modelo
$$\text{Consumo\_Energia} = 101.3830 + 9.9660 \cdot \text{Temperatura} + 5.0118 \cdot \text{Hora} - 3.0892 \cdot \text{Dia\_Semana}$$

#### B. Interpretación de Coeficientes y Dinámica de Red
* **Intercepto ($\beta_0 = 101.3830$ kWh):** Carga base fija de consumo pasivo de la infraestructura (sistemas de soporte continuo, servidores y refrigeración básica).
* **$\beta_1$ (Temperatura = $+9.9660$):** Por cada grado centígrado (°C) que aumenta la temperatura ambiental, el consumo eléctrico se eleva en **~9.97 kWh**. Este fenómeno responde directamente a la alta demanda térmica de climatización artificial (HVAC / aire acondicionado).
* **$\beta_2$ (Hora = $+5.0118$):** Cada hora de avance en el transcurso del día aporta un incremento de **5.01 kWh** en la demanda hasta alcanzar las franjas pico de la jornada laboral y vespertina.
* **$\beta_3$ (Día de la Semana = $-3.0892$):** A medida que se avanza del lunes (1) hacia el domingo (7), el consumo global se reduce en promedio **3.09 kWh** por día, concordando con el cese parcial de actividades fabriles, comerciales y académicas durante los fines de semana.

#### C. Métricas de Evaluación
| Métrica | Valor Numérico | Interpretación |
| :--- | :---: | :--- |
| **$R^2$** | **0.9005** | El **90.05%** de la fluctuación del consumo energético es explicada por la temperatura, la hora y el día. |
| **MSE** | **405.2295** | Error cuadrático de la red de potencia. |
| **RMSE** | **20.1303 kWh** | Nivel de incertidumbre predictiva en un sistema con consumos promedio superiores a 350 kWh (~5.7% de error relativo). |

#### D. Identificación de la Variable con Mayor Impacto
* Coeficiente estandarizado de la **Temperatura:** **0.7814** (Impacto dominante).
* Coeficiente estandarizado de la **Hora:** **0.5435**.
* Coeficiente estandarizado del **Día de la Semana:** **-0.0968**.

**Conclusión:** La **`Temperatura`** es el factor más influyente sobre el consumo eléctrico global, duplicando el efecto de las variables de calendario.

---

## 3. Análisis de Visualizaciones Generadas

### A. Gráficos de Regresión - Dólar (`static/plots/dolar_plots.png`)
* **Día vs Precio del Dólar:** Muestra una correlación lineal positiva limpia y casi determinística ($r \approx 0.99$). Los puntos se alinean de manera compacta sobre la recta de ajuste.
* **Inflación vs Precio / Tasa vs Precio:** Exhiben nubes de dispersión difusa con pendientes suaves ligeramente negativas, lo que confirma que en este conjunto de datos la estacionalidad/día domina por completo sobre el ruido de las variables financieras agregadas.

### B. Gráficos de Regresión - Glucosa (`static/plots/glucosa_plots.png`)
* **Edad vs Glucosa:** Tendencia ascendente moderada y consistente; a mayor edad cronológica, la nube de puntos se desplaza hacia valores por encima del umbral de prediabetes (100 mg/dL).
* **IMC vs Glucosa:** Tendencia positiva continua. Los individuos con obesidad (IMC > 30) concentran los registros más altos de glucosa.
* **Actividad Física vs Glucosa:** Pendiente negativa evidente. Aquellos pacientes que dedican más de 5 horas semanales al ejercicio presentan concentraciones predominantemente por debajo de 110 mg/dL.

### C. Heatmap y Regresiones - Energía (`static/plots/energia_plots.png`)
* **Mapa de Calor de Correlación de Pearson:** Revela correlación positiva fuerte entre `Temperatura` y `Consumo_Energia` ($r \approx 0.78$), correlación moderada-fuerte con `Hora` ($r \approx 0.54$), y correlación negativa leve con `Dia_Semana` ($r \approx -0.10$).
* **Dispersión:** Refleja la fuerte sensibilidad de la demanda eléctrica a los picos de temperatura vespertinos.

---

## 4. Despliegue e Interfaz de Usuario

### Arquitectura Técnica
1. **Serialización con Joblib (`models/*.joblib`):**
   - Se exportaron los objetos de estimador entrenados junto con sus metadatos (coeficientes, interceptos, métricas y estadísticas descriptivas de rango y promedio).
   - `joblib` garantiza serialización binaria ultrarrápida y desacopla la etapa pesada de entrenamiento de la inferencia web.
2. **Aplicación Web Interactiva (`app.py`):**
   - Construida con **Streamlit**.
   - Integra un menú lateral para alternar entre los 3 proyectos y una vista consolidada de métricas globales.
   - Provee validadores numéricos con rangos mínimos y máximos contextualizados en los datos.
   - Despliega en tiempo real tarjetas métricas, ecuaciones formateadas en KaTeX, intervalos de error $\pm 1 \text{ RMSE}$, y diagnósticos interpretativos clínicos/financieros.
3. **Orquestador Central (`run_all.py`):**
   - Automatiza el ciclo completo: entrena, genera gráficos, exporta modelos, imprime un informe ejecutivo en consola y arranca el servidor web en un solo paso.

---

## 5. Conclusiones Técnicas

1. **Efectividad del Estándar CRISP-DM:** La aplicación estricta de las 6 fases permitió transitar de manera ordenada desde la formulación analítica de preguntas de negocio hasta el despliegue de micro-servicios predictivos interactivos.
2. **Precisión del Ajuste Lineal:** Tanto en la serie financiera del dólar ($R^2 = 99.52\%$) como en la demanda energética ($R^2 = 90.05\%$), la regresión lineal múltiple demostró una capacidad explicativa excepcional gracias al comportamiento cuasi-lineal de los determinantes dominantes (`Dia` y `Temperatura`).
3. **Interpretación Causal en Salud:** En el modelo de glucosa ($R^2 = 68.55\%$), el análisis cuantificó cómo el ejercicio semanal actúa como un potente factor protector ($\beta = -2.01$), mitigando eficazmente los efectos deletéreos de la edad y del sobrepeso.
4. **Viabilidad para Entornos de Producción:** El flujo desacoplado (análisis por lotes $\rightarrow$ persistencia binaria con `joblib` $\rightarrow$ frontend reactivo con Streamlit) garantiza una solución liviana, de baja latencia y alta mantenibilidad.
