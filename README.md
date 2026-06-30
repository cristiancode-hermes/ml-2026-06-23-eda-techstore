# 🎯 TechStore — Análisis Exploratorio de Datos (EDA)

**Nivel:** 2 (Análisis Exploratorio) | **Skills:** EDA, Estadística descriptiva, Correlaciones, Heatmaps, Pairplots, Outliers

> 📅 2026-06-23 | 📊 Proyecto #6 de la ruta ML

---

## 📖 Qué vas a aprender

El **Análisis Exploratorio de Datos (EDA)** es el paso más importante antes de construir cualquier modelo de Machine Learning. Es como cuando un médico te hace análisis de sangre antes de recetarte algo — sin entender los datos, cualquier modelo será una caja negra que no sabes si funciona bien.

En este proyecto vas a aprender:

### 🔍 1. Primer vistazo a los datos
Antes de hacer cualquier cosa, hay que conocer el dataset:
- ¿Cuántas filas y columnas tiene? → `df.shape`
- ¿Qué tipo de datos hay? → `df.dtypes`
- ¿Hay valores nulos? → `df.isnull().sum()`
- ¿Cómo se ven las primeras filas? → `df.head()`

### 📐 2. Estadística descriptiva
No puedes mejorar lo que no puedes medir. La estadística descriptiva te da números clave:
- **Media** → el valor promedio
- **Mediana** → el valor del medio (más robusta que la media ante outliers)
- **Desviación estándar** → qué tan dispersos están los datos
- **Mínimo, Máximo, Cuartiles** → los límites de tus datos
- **Skewness (asimetría)** → ¿los datos están centrados o se van a un lado?
- **Kurtosis (curtosis)** → ¿la distribución es puntiaguda o plana?

### 📊 3. Distribuciones
Cada variable numérica tiene una "forma" cuando la graficamos:
- **Histograma** → barras que muestran la frecuencia de cada valor
- **Boxplot** → caja que muestra mediana, cuartiles y outliers
- **Distribución normal (campana de Gauss)** → cuando los datos se agrupan simétricamente alrededor de la media

### 🏷️ 4. Variables categóricas
Las variables de texto también se analizan:
- **Frecuencias** → ¿cuántos pedidos hay por ciudad?
- **Proporciones** → ¿qué porcentaje es Premium vs Standard?
- **Gráficos de barras** → para visualizar las categorías

### 🔗 5. Relaciones entre variables
Aquí empieza la magia del EDA: ¿cómo se relacionan las variables?
- **Groupby + agg** → promedios por categoría
- **Boxplots comparativos** → ¿los Premium gastan más?
- **Scatter plots** → ¿hay relación entre precio y puntuación?
- **Violin plots** → como boxplot pero más detallado

### 🧮 6. Correlación
La correlación mide LA RELACIÓN LINEAL entre dos variables numéricas:

| Valor | Significado |
|-------|-------------|
| r = 1.0 | Correlación PERFECTA positiva (sube A → sube B) |
| r = -1.0 | Correlación PERFECTA negativa (sube A → baja B) |
| r = 0.0 | Sin correlación lineal |
| r > 0.7 | Correlación FUERTE |
| r = 0.4-0.7 | Correlación MODERADA |
| r = 0.2-0.4 | Correlación DÉBIL |

**⚠️ Correlación NO es causalidad.** Solo porque dos variables se muevan juntas no significa que una cause la otra.

### 🎯 7. Outliers (valores atípicos)
Los outliers pueden ser errores, fraudes o datos reales pero extremos. Los detectamos con:
- **Boxplot** → los puntos fuera de los "bigotes" son outliers visuales
- **Método IQR** → un valor es outlier si está fuera de [Q1 - 1.5×IQR, Q3 + 1.5×IQR]

### 💡 8. Insights de negocio
El EDA no es solo matemáticas — tiene que contar una historia que ayude a tomar decisiones.

---

## 🧪 Código paso a paso

El código completo está en `project.py`. Aquí te explico cada sección:

### Paso 1: Importar librerías
```python
import pandas as pd      # Manipulación de datos
import numpy as np       # Operaciones numéricas
import matplotlib.pyplot as plt  # Gráficos base
import seaborn as sns    # Gráficos estadísticos más bonitos
```

### Paso 2: Cargar el dataset
```python
df = pd.read_csv('data/dataset_completo.csv')
```
Cargamos el dataset que generamos en el proyecto anterior de merge-join. Tiene 155 pedidos con información de clientes, productos y reseñas.

### Paso 3: Primer vistazo
```python
print(df.shape)              # (155, 18) → 155 filas, 18 columnas
print(df.dtypes)             # Tipo de cada columna
print(df.head())             # Primeras 5 filas
print(df.describe())         # Estadísticas de columnas numéricas
print(df.isnull().sum())     # Valores nulos por columna
```

### Paso 4: Distribuciones
```python
# Histograma con media y mediana
plt.hist(df['total'], bins=20)
plt.axvline(media, color='red', label=f'Media: {media:.2f}')
plt.axvline(mediana, color='green', label=f'Mediana: {mediana:.2f}')
```

### Paso 5: Correlación
```python
corr_matrix = df[cols].corr(method='pearson')
sns.heatmap(corr_matrix, annot=True, cmap='RdBu_r', center=0)
```

### Paso 6: Pairplot
```python
sns.pairplot(df, vars=['total', 'precio', 'stock', 'puntuacion'], hue='segmento')
```

### Paso 7: Outliers con IQR
```python
Q1 = datos.quantile(0.25)
Q3 = datos.quantile(0.75)
IQR = Q3 - Q1
lim_inf = Q1 - 1.5 * IQR
lim_sup = Q3 + 1.5 * IQR
outliers = datos[(datos < lim_inf) | (datos > lim_sup)]
```

---

## 📊 Resultados

### 📁 Dataset analizado
| Archivo | Filas | Columnas | Fuente |
|---------|-------|----------|--------|
| `data/dataset_completo.csv` | 155 | 18 | Proyecto merge-join |

### 📊 Visualizaciones generadas

| Archivo | Descripción |
|---------|-------------|
| `docs/01_mapa_nulos.png` | Mapa de calor de valores nulos (líneas púrpura = datos faltantes) |
| `docs/02_histogramas_distribuciones.png` | Histogramas de total, precio, stock, puntuación con media/mediana |
| `docs/03_boxplots_outliers.png` | Boxplots con cuartiles para detección visual de outliers |
| `docs/04_categorias_barras.png` | Barras de ciudades, segmentos, estados y categorías |
| `docs/05_relaciones_variables.png` | Boxplots, violinplots y scatter de relaciones |
| `docs/06_heatmap_correlacion.png` | Matriz de correlación con heatmap |
| `docs/07_pairplot_completo.png` | Pairplot de todas las variables coloreado por segmento |
| `docs/08_outliers_iqr.png` | Boxplots con límites IQR para detección de outliers |

### 🔍 Hallazgos principales

```
💰 Gasto por segmento:
  Standard  → $X,XXX.XX promedio
  Gold      → $X,XXX.XX promedio
  Premium   → $X,XXX.XX promedio (el que más gasta)

📍 Ciudades:
  Bogotá    → mayor volumen de pedidos
  [Ciudad2] → mayor ticket promedio

📦 Categorías:
  [Categoría] → la más popular
  [Categoría] → la mejor puntuada

🧮 Correlaciones:
  total vs precio  → r = X.XXX (¿los productos caros generan pedidos más altos?)
  precio vs puntuación → r = X.XXX (¿caro = mejor? ¿o no hay relación?)
```

---

## 💡 Conceptos clave

### Estadística descriptiva
- **Media:** suma de valores / cantidad. Sensible a outliers.
- **Mediana:** valor del medio cuando ordenas los datos. Robusta a outliers.
- **Desviación estándar:** qué tan dispersos están los datos. Bajo = datos cercanos a la media.
- **Skewness:** mide la asimetría. Positivo = cola a la derecha. Negativo = cola a la izquierda.
- **Kurtosis:** mide qué tan "puntiaguda" es la distribución. Alta = valores extremos.
- **IQR (Rango Intercuartílico):** Q3 - Q1. Mide la dispersión del 50% central de los datos.

### EDA visual
- **Histograma:** muestra la forma de la distribución con barras.
- **Boxplot:** muestra mediana, cuartiles y outliers de un vistazo.
- **Violinplot:** como boxplot pero muestra la densidad (forma) de la distribución.
- **Heatmap de correlación:** colores intensos = correlación fuerte.
- **Pairplot:** todas las combinaciones de variables en una sola figura.

### Outliers
- **Método IQR:** estándar en ciencia de datos. Un valor es outlier si está fuera del rango [Q1 - 1.5×IQR, Q3 + 1.5×IQR].
- **No todos los outliers son malos:** pueden ser fraudes (gastos enormes), errores (precio negativo) o datos legítimos (un cliente que compró 100 unidades).
- **Decisión:** en ML, decides si eliminar, transformar o mantener los outliers según el contexto.

---

## 🚀 Siguiente paso

¡Completaste el primer proyecto de **Nivel 2**! 🎉

Has aprendido los fundamentos del EDA. El próximo proyecto será:

### 🔧 Feature Engineering — Crear nuevas variables para ML

Aprenderás a:
- Extraer componentes de fechas (día, mes, día_semana)
- Crear rangos de precios (bins o buckets)
- Calcular ratios (gasto por producto, precio vs promedio)
- Codificar variables categóricas (one-hot encoding)
- Normalizar/escalar variables numéricas

El Feature Engineering es donde los científicos de datos pasan el 80% de su tiempo — y donde realmente se gana el ML. ¡Prepárate!
