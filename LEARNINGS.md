# Learnings — 2026-06-23

## Qué se hizo
Primer proyecto de Nivel 2 (Análisis Exploratorio de Datos) usando el dataset
completo de TechStore generado en el proyecto anterior de merge-join.

Realizamos un EDA completo cubriendo:
- Estructura del dataset (shape, info, dtypes, head/tail)
- Valores nulos: detección y visualización (heatmap)
- Estadística descriptiva: describe, media, mediana, desviación, asimetría (skewness), curtosis
- Distribuciones: histogramas con media/mediana
- Variables categóricas: value_counts, gráficos de barras
- Relaciones entre variables: groupby, boxplots, violinplots, scatter
- Correlación de Pearson y heatmap
- Pairplot con segmentación por categoría
- Outliers por método IQR con visualización

## Conceptos nuevos
- [x] EDA: describe, correlaciones, distribuciones
- [x] Heatmaps y pairplots
- [x] Estadística descriptiva (media, mediana, desv.std, skew, kurtosis)
- [x] Detección de outliers (método IQR)
- [x] Boxplots y violinplots
- [x] Matriz de correlación e interpretación
- [x] Skewness (asimetría) y Kurtosis (curtosis)
- [x] Pairplot de seaborn
- [x] Insights de negocio desde datos

## Dificultades encontradas
- Ninguna significativa. El dataset ya estaba limpio del proyecto anterior.

## Notas para el próximo proyecto
- El EDA cubrió 4 skills de Nivel 2 en un solo proyecto
- Próximo paso: Feature Engineering (crear nuevas variables útiles para ML)
- El dataset tiene fechas que pueden descomponerse en día/mes/día_semana/fin_semana
- Se pueden crear agregaciones, ratios, categorías de precio, etc.
