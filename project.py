#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════╗
║  TechStore — Análisis Exploratorio de Datos (EDA)              ║
║  Nivel 2 | Skill: EDA, estadística, correlaciones, outliers    ║
╚══════════════════════════════════════════════════════════════════╝

¿Qué es el EDA?
────────────────
El Análisis Exploratorio de Datos (EDA) es el paso MÁS IMPORTANTE
antes de construir cualquier modelo de ML. Es como cuando un médico
hace análisis de sangre antes de recetar medicamentos.

En EDA vamos a:
  1. Entender la estructura de los datos (forma, tipos, nulos)
  2. Calcular estadísticas descriptivas (media, mediana, std, percentiles)
  3. Analizar distribuciones (¿los datos son normales? ¿sesgados?)
  4. Encontrar relaciones entre variables (correlaciones)
  5. Detectar anomalías y outliers (valores atípicos)
  6. Generar insights de negocio

¡Empecemos!
"""

# ============================================================
# PASO 1: Importamos las librerías necesarias
# ============================================================
# pandas → manipulación de datos (DataFrames)
# numpy  → operaciones numéricas
# matplotlib → gráficos base
# seaborn → gráficos estadísticos más bonitos (se basa en matplotlib)
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings

# Ignoramos warnings molestos de versiones
warnings.filterwarnings('ignore')

# Configuramos matplotlib para que muestre los textos bien en español
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['figure.dpi'] = 100
plt.rcParams['figure.figsize'] = (12, 6)

# Colores profesionales para los gráficos
COLORES = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#3B1F2B']
sns.set_palette(COLORES)
sns.set_style("whitegrid")

# Creamos directorio para guardar gráficos
os.makedirs('docs', exist_ok=True)

print("=" * 70)
print("🔬 TECHSTORE — ANÁLISIS EXPLORATORIO DE DATOS (EDA)")
print("=" * 70)
print()

# ============================================================
# PASO 2: Cargar los datos
# ============================================================
# Cargamos el dataset completo que generamos en el proyecto anterior
# (merge-join). Este dataset combina clientes + pedidos + productos + reseñas
# ============================================================

print("📂 Cargando dataset...")
df = pd.read_csv('data/dataset_completo.csv')
print(f"   ✅ Dataset cargado: {df.shape[0]} filas × {df.shape[1]} columnas")
print()

# ============================================================
# PASO 3: PRIMER VISTAZO — Conoce tus datos
# ============================================================
# Antes de hacer cualquier análisis, tenemos que responder:
#   - ¿Cuántos datos tenemos? (shape)
#   - ¿Qué tipo de datos? (dtypes)
#   - ¿Hay valores nulos? (isnull)
#   - ¿Cómo se ven las primeras filas? (head)
# ============================================================

print("─" * 70)
print("🔍 PRIMER VISTAZO A LOS DATOS")
print("─" * 70)
print()

# --- 3.1: Info general del dataset ---
print("📋 Info del dataset (tipos de datos, memoria):")
# .info() nos da: nombre de columnas, tipo de dato, cantidad de no-nulos y memoria
# Es como la ficha técnica del dataset
print(df.info())
print()

# --- 3.2: Primeras y últimas filas ---
print("👀 Primeras 5 filas: las cabeceras de los datos:")
print(df.head().to_string())
print()
print("👀 Últimas 5 filas: para ver cómo terminan los datos:")
print(df.tail().to_string())
print()

# --- 3.3: Tipos de datos ---
print("📊 Tipos de cada columna:")
print(df.dtypes.to_string())
print()

# --- 3.4: Columnas disponibles ---
print("🏷️  Columnas del dataset:")
for i, col in enumerate(df.columns, 1):
    print(f"   {i:2d}. {col}")
print()

# --- 3.5: Estadísticas rápidas de columnas numéricas ---
print("📐 Estadísticas descriptivas (columnas numéricas):")
# .describe() calcula automáticamente: count, mean, std, min, 25%, 50%, 75%, max
# Es el resumen MÁS IMPORTANTE para entender las variables numéricas
print(df.describe().to_string())
print()

# --- 3.6: Estadísticas de columnas categóricas ---
print("🏷️  Columnas categóricas (object):")
# Seleccionamos solo las columnas de tipo 'object' (texto/categorías)
cat_cols = df.select_dtypes(include=['object']).columns
for col in cat_cols:
    print(f"\n   → {col}: {df[col].nunique()} valores únicos")
    print(f"     Valores: {df[col].unique()[:8]}...")
print()

# ============================================================
# PASO 4: VALORES NULOS — El enemigo silencioso
# ============================================================
# Los valores nulos (NaN) son datos que faltan.
# En ML real, SIEMPRE hay nulos. Ignorarlos causa errores.
# Aquí vemos cuántos hay y dónde.
# ============================================================

print("─" * 70)
print("⚠️  ANÁLISIS DE VALORES NULOS")
print("─" * 70)

# Calculamos la cantidad y el porcentaje de nulos por columna
nulos = pd.DataFrame({
    'Columna': df.columns,
    'Nulos': df.isnull().sum().values,
    'Porcentaje': (df.isnull().sum().values / len(df) * 100).round(2)
})
nulos = nulos[nulos['Nulos'] > 0].sort_values('Nulos', ascending=False)

print(f"\n📊 Columnas con valores nulos (de {len(df)} filas totales):")
print(nulos.to_string(index=False))
print()

# Explicación de por qué hay nulos:
print("💡 ¿Por qué hay nulos en reseñas?")
print("   - 75 pedidos NO tienen reseña (los clientes no opinaron)")
print("   - Esto es NORMAL en datos reales → tasa de reseña ~52%")
print("   - En ML, decidirías si imputar (rellenar) o eliminar según el contexto")
print()

# Visualización 1: Mapa de calor de nulos
# Un heatmap de nulos muestra DÓNDE faltan datos de forma visual
# Las líneas blancas horizontales = filas con datos faltantes
plt.figure(figsize=(14, 6))
sns.heatmap(df.isnull(), cbar=False, cmap='viridis', yticklabels=False)
plt.title('Mapa de Valores Nulos\n(Líneas amarillas = datos presentes, Púrpura = nulos)', 
          fontsize=14, fontweight='bold')
plt.xlabel('Columnas')
plt.ylabel('Filas (155 pedidos)')
plt.tight_layout()
plt.savefig('docs/01_mapa_nulos.png', dpi=120, bbox_inches='tight')
plt.close()
print("   ✅ Gráfico guardado: docs/01_mapa_nulos.png")
print()

# ============================================================
# PASO 5: ANÁLISIS DE DISTRIBUCIONES
# ============================================================
# Una distribución nos dice CÓMO se comporta una variable.
# ¿Es simétrica? ¿Tiene sesgo? ¿Hay valores extremos?
# Usamos histogramas (barras) y boxplots (cajas y bigotes).
# ============================================================

print("─" * 70)
print("📊 ANÁLISIS DE DISTRIBUCIONES")
print("─" * 70)

# Seleccionamos las variables numéricas principales para analizar
# Excluimos IDs (no tienen significado estadístico)
vars_numericas = ['total', 'precio', 'stock', 'puntuacion']

for var in vars_numericas:
    print(f"\n   📈 Variable: {var.upper()}")
    
    # Obtenemos los valores (quitando nulos para puntuacion)
    datos = df[var].dropna()
    
    # Estadísticas clave para entender la distribución
    media = datos.mean()
    mediana = datos.median()
    desvio = datos.std()
    minimo = datos.min()
    maximo = datos.max()
    
    print(f"      Media:    ${media:,.2f}" if var in ['total', 'precio'] else f"      Media:    {media:.2f}")
    print(f"      Mediana:  ${mediana:,.2f}" if var in ['total', 'precio'] else f"      Mediana:  {mediana:.2f}")
    print(f"      Desv. Std: ${desvio:,.2f}" if var in ['total', 'precio'] else f"      Desv. Std: {desvio:.2f}")
    print(f"      Mínimo:   ${minimo:,.2f}" if var in ['total', 'precio'] else f"      Mínimo:   {minimo:.2f}")
    print(f"      Máximo:   ${maximo:,.2f}" if var in ['total', 'precio'] else f"      Máximo:   {maximo:.2f}")
    
    # Asimetría (skewness): 
    #   skew ≈ 0 → distribución simétrica (como una campana)
    #   skew > 0 → sesgada a la derecha (cola larga hacia la derecha)
    #   skew < 0 → sesgada a la izquierda (cola larga hacia la izquierda)
    skew = datos.skew()
    if abs(skew) < 0.5:
        skew_desc = "simétrica ✅"
    elif skew > 0:
        skew_desc = f"sesgada a la derecha ({skew:.2f}) → mayoría de valores bajos, algunos altos"
    else:
        skew_desc = f"sesgada a la izquierda ({skew:.2f}) → mayoría de valores altos, algunos bajos"
    print(f"      Asimetría: {skew_desc}")
    
    # Curtosis (kurtosis): mide qué tan "puntiaguda" es la distribución
    # kurt > 0 → más puntiaguda que normal (valores extremos)
    # kurt < 0 → más aplanada que normal
    kurt = datos.kurtosis()
    print(f"      Curtosis: {kurt:.2f} ({'más' if kurt > 0 else 'menos'} puntiaguda que una normal)")
    print()

# --- 5.1: Histogramas de variables numéricas ---
# El histograma agrupa los datos en "bins" (cajones) y cuenta cuántos caen en cada uno
# Nos permite ver la FORMA de la distribución
print("   📊 Generando histogramas...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
vars_plot = [('total', 'Total del Pedido ($)', 'Distribución del Total de Pedidos'),
             ('precio', 'Precio del Producto ($)', 'Distribución de Precios de Productos'),
             ('stock', 'Stock Disponible (uds)', 'Distribución de Stock'),
             ('puntuacion', 'Puntuación (1-5)', 'Distribución de Puntuaciones')]

for ax, (var, xlabel, title) in zip(axes.flatten(), vars_plot):
    datos = df[var].dropna()
    ax.hist(datos, bins=20, color=COLORES[0], edgecolor='white', alpha=0.8)
    ax.axvline(datos.mean(), color='red', linestyle='--', linewidth=2, label=f'Media: {datos.mean():.2f}')
    ax.axvline(datos.median(), color='green', linestyle='-', linewidth=2, label=f'Mediana: {datos.median():.2f}')
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Frecuencia')
    ax.legend()
    # Añadimos texto con skewness
    ax.text(0.98, 0.95, f'Skew: {datos.skew():.2f}', transform=ax.transAxes,
            ha='right', va='top', fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.suptitle('📊 Distribuciones de Variables Numéricas', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('docs/02_histogramas_distribuciones.png', dpi=120, bbox_inches='tight')
plt.close()
print("   ✅ Gráfico guardado: docs/02_histogramas_distribuciones.png")
print()

# --- 5.2: Boxplots (diagramas de caja y bigotes) ---
# El boxplot muestra: mediana, cuartiles (Q1, Q3), y valores atípicos (outliers)
# 
#   ┌────────────────────────────────────────────────────────────┐
#   │  ∙───┃███┃═══∙                                            │
#   │ min  Q1   mediana  Q3  max                                 │
#   │ (los puntos fuera de los bigotes = OUTLIERS)               │
#   └────────────────────────────────────────────────────────────┘
#
# Los outliers son valores que están:
#   - Por debajo de Q1 - 1.5*IQR
#   - Por encima de Q3 + 1.5*IQR
#   Donde IQR = Q3 - Q1 (rango intercuartílico)

print("   📦 Generando boxplots...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

for ax, (var, xlabel, title) in zip(axes.flatten(), vars_plot):
    datos = df[var].dropna()
    bp = ax.boxplot(datos, patch_artist=True, widths=0.4)
    bp['boxes'][0].set_facecolor(COLORES[0])
    bp['boxes'][0].set_alpha(0.7)
    bp['fliers'][0].set_marker('o')
    bp['fliers'][0].set_markerfacecolor('red')
    bp['fliers'][0].set_markersize(8)
    
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.set_ylabel(xlabel)
    ax.set_xticklabels([var.upper()])
    
    # Añadimos estadísticas clave al lado del boxplot
    q1, q3 = np.percentile(datos, [25, 75])
    iqr = q3 - q1
    ax.text(1.1, q1, f'Q1: {q1:.1f}', fontsize=9, va='center')
    ax.text(1.1, q3, f'Q3: {q3:.1f}', fontsize=9, va='center')
    ax.text(1.1, datos.median(), f'Med: {datos.median():.1f}', fontsize=9, va='center', fontweight='bold')

plt.suptitle('📦 Boxplots — Detección Visual de Outliers', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('docs/03_boxplots_outliers.png', dpi=120, bbox_inches='tight')
plt.close()
print("   ✅ Gráfico guardado: docs/03_boxplots_outliers.png")
print()

# ============================================================
# PASO 6: ANÁLISIS DE CATEGORÍAS
# ============================================================
# Las variables categóricas (texto) también se analizan:
# - Frecuencia de cada categoría
# - ¿Hay categorías dominantes?
# - ¿Cómo se relacionan con variables numéricas?
# ============================================================

print("─" * 70)
print("🏷️  ANÁLISIS DE VARIABLES CATEGÓRICAS")
print("─" * 70)

# --- 6.1: Distribución de pedidos por ciudad ---
print("\n📍 Distribución de pedidos por ciudad:")
ciudades = df['ciudad'].value_counts()
for ciudad, count in ciudades.items():
    pct = count / len(df) * 100
    barra = '█' * int(pct / 2)
    print(f"   {ciudad:12s} → {count:3d} pedidos ({pct:5.1f}%) {barra}")

# --- 6.2: Distribución por segmento de cliente ---
print("\n👥 Distribución por segmento de cliente:")
segmentos = df['segmento'].value_counts()
for seg, count in segmentos.items():
    pct = count / len(df) * 100
    barra = '█' * int(pct / 2)
    print(f"   {seg:10s} → {count:3d} pedidos ({pct:5.1f}%) {barra}")

# --- 6.3: Distribución por estado del pedido ---
print("\n📦 Distribución por estado del pedido:")
estados = df['estado'].value_counts()
for est, count in estados.items():
    pct = count / len(df) * 100
    barra = '█' * int(pct / 2)
    print(f"   {est:12s} → {count:3d} pedidos ({pct:5.1f}%) {barra}")

# --- 6.4: Categorías de producto ---
print("\n📁 Distribución por categoría de producto:")
cats = df['categoria'].value_counts()
for cat, count in cats.items():
    pct = count / len(df) * 100
    barra = '█' * int(pct / 2)
    print(f"   {cat:14s} → {count:3d} pedidos ({pct:5.1f}%) {barra}")
print()

# --- 6.5: Gráfico de barras para categorías principales ---
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Ciudades
ax1 = axes[0, 0]
ciudades.plot(kind='bar', ax=ax1, color=COLORES[0], edgecolor='white')
ax1.set_title('Pedidos por Ciudad', fontsize=13, fontweight='bold')
ax1.set_xlabel('Ciudad')
ax1.set_ylabel('Cantidad de Pedidos')
ax1.tick_params(axis='x', rotation=0)

# Segmentos
ax2 = axes[0, 1]
segmentos.plot(kind='bar', ax=ax2, color=COLORES[1], edgecolor='white')
ax2.set_title('Pedidos por Segmento de Cliente', fontsize=13, fontweight='bold')
ax2.set_xlabel('Segmento')
ax2.set_ylabel('Cantidad de Pedidos')
ax2.tick_params(axis='x', rotation=0)

# Estados
ax3 = axes[1, 0]
estados.plot(kind='bar', ax=ax3, color=COLORES[2], edgecolor='white')
ax3.set_title('Pedidos por Estado', fontsize=13, fontweight='bold')
ax3.set_xlabel('Estado')
ax3.set_ylabel('Cantidad de Pedidos')
ax3.tick_params(axis='x', rotation=0)

# Categorías
ax4 = axes[1, 1]
cats.plot(kind='bar', ax=ax4, color=COLORES[3], edgecolor='white')
ax4.set_title('Pedidos por Categoría de Producto', fontsize=13, fontweight='bold')
ax4.set_xlabel('Categoría')
ax4.set_ylabel('Cantidad de Pedidos')
ax4.tick_params(axis='x', rotation=15)

plt.suptitle('🏷️ Distribución de Variables Categóricas', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('docs/04_categorias_barras.png', dpi=120, bbox_inches='tight')
plt.close()
print("   ✅ Gráfico guardado: docs/04_categorias_barras.png")
print()

# ============================================================
# PASO 7: RELACIONES ENTRE VARIABLES
# ============================================================
# Ahora buscamos conexiones:
# - ¿Los clientes Premium gastan más?
# - ¿Hay relación entre precio del producto y puntuación?
# - ¿Varía el gasto por ciudad?
# ============================================================

print("─" * 70)
print("🔗 RELACIONES ENTRE VARIABLES")
print("─" * 70)

# --- 7.1: Gasto promedio por segmento ---
print("\n💰 Gasto promedio por segmento de cliente:")
gasto_segmento = df.groupby('segmento')['total'].agg(['mean', 'median', 'count', 'sum'])
for seg, row in gasto_segmento.iterrows():
    print(f"   {seg:10s} | Promedio: ${row['mean']:>7.2f} | Mediana: ${row['median']:>7.2f} | Pedidos: {int(row['count']):3d} | Total: ${row['sum']:>8.2f}")

# --- 7.2: Gasto promedio por ciudad ---
print("\n📍 Gasto promedio por ciudad:")
gasto_ciudad = df.groupby('ciudad')['total'].agg(['mean', 'median', 'count'])
for ciudad, row in gasto_ciudad.iterrows():
    print(f"   {ciudad:12s} | Promedio: ${row['mean']:>7.2f} | Mediana: ${row['median']:>7.2f} | Pedidos: {int(row['count']):3d}")

# --- 7.3: Puntuación promedio por categoría ---
print("\n⭐ Puntuación promedio por categoría de producto (solo reseñas):")
punt_cat = df.dropna(subset=['puntuacion']).groupby('categoria')['puntuacion'].agg(['mean', 'count', 'std'])
for cat, row in punt_cat.iterrows():
    print(f"   {cat:14s} | Promedio: {row['mean']:.2f}⭐ | Reseñas: {int(row['count']):2d} | Std: {row['std']:.2f}")

# --- 7.4: Gasto por categoría ---
print("\n📦 Gasto total por categoría de producto:")
gasto_cat = df.groupby('categoria')['total'].sum().sort_values(ascending=False)
for cat, total in gasto_cat.items():
    pct = total / gasto_cat.sum() * 100
    barra = '█' * int(pct)
    print(f"   {cat:14s} → ${total:>8.2f} ({pct:5.1f}%) {barra}")
print()

# --- 7.5: Visualizaciones de relaciones ---
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# Gráfico 1: Total por segmento (boxplot)
ax1 = axes[0, 0]
sns.boxplot(data=df, x='segmento', y='total', ax=ax1, palette=COLORES[:3], order=['Standard', 'Gold', 'Premium'])
ax1.set_title('Distribución de Gasto por Segmento', fontsize=13, fontweight='bold')
ax1.set_xlabel('Segmento')
ax1.set_ylabel('Total del Pedido ($)')
# Añadimos la media como punto rojo
means = df.groupby('segmento')['total'].mean()
for i, seg in enumerate(['Standard', 'Gold', 'Premium']):
    ax1.scatter(i, means[seg], color='red', s=100, zorder=5, marker='D', label='Media' if i == 0 else '')

# Gráfico 2: Total por ciudad (violin plot = como boxplot pero muestra la densidad)
ax2 = axes[0, 1]
sns.violinplot(data=df, x='ciudad', y='total', ax=ax2, palette=COLORES[:4])
ax2.set_title('Distribución de Gasto por Ciudad', fontsize=13, fontweight='bold')
ax2.set_xlabel('Ciudad')
ax2.set_ylabel('Total del Pedido ($)')

# Gráfico 3: Puntuación por categoría
ax3 = axes[1, 0]
punt_data = df.dropna(subset=['puntuacion'])
sns.boxplot(data=punt_data, x='categoria', y='puntuacion', ax=ax3, palette=COLORES[:4])
ax3.set_title('Puntuación por Categoría de Producto', fontsize=13, fontweight='bold')
ax3.set_xlabel('Categoría')
ax3.set_ylabel('Puntuación (1-5)')

# Gráfico 4: Precio vs Stock (dispersión = scatter)
ax4 = axes[1, 1]
scatter = ax4.scatter(df['precio'], df['stock'], c=df['total'], cmap='viridis', 
                      alpha=0.7, s=80, edgecolors='black', linewidth=0.5)
ax4.set_title('Precio vs Stock (color = total del pedido)', fontsize=13, fontweight='bold')
ax4.set_xlabel('Precio del Producto ($)')
ax4.set_ylabel('Stock Disponible (uds)')
plt.colorbar(scatter, ax=ax4, label='Total ($)')

plt.suptitle('🔗 Relaciones entre Variables', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('docs/05_relaciones_variables.png', dpi=120, bbox_inches='tight')
plt.close()
print("   ✅ Gráfico guardado: docs/05_relaciones_variables.png")
print()

# ============================================================
# PASO 8: MATRIZ DE CORRELACIÓN — ¡El corazón del EDA!
# ============================================================
# La correlación mide la RELACIÓN LINEAL entre dos variables numéricas.
#
#   r = 1.0  → correlación perfecta positiva  (sube A → sube B)
#   r = -1.0 → correlación perfecta negativa  (sube A → baja B)
#   r = 0.0  → sin correlación lineal
#
# IMPORTANTE: Correlación NO es causalidad!
#   Ej: Las ventas de helados y los ahogados están correlacionados...
#   pero no es que los helados causen ahogados — ambas aumentan en verano ☀️
# ============================================================

print("─" * 70)
print("🧮 MATRIZ DE CORRELACIÓN")
print("─" * 70)

# Seleccionamos columnas numéricas para la correlación
cols_corr = ['total', 'precio', 'stock', 'puntuacion']
# Creamos una copia del dataset solo con estas columnas y sin nulos en esas columnas
df_corr = df[cols_corr].dropna()

# Calculamos la matriz de correlación de Pearson
# Pearson mide la relación LINEAL entre dos variables
corr_matrix = df_corr.corr(method='pearson')

print("\n📊 Matriz de correlación de Pearson:")
print(corr_matrix.to_string())
print()

# Explicación de los valores
print("💡 Interpretación de la correlación:")
for i, col1 in enumerate(cols_corr):
    for j, col2 in enumerate(cols_corr):
        if i < j:  # Solo mostramos cada par una vez
            r = corr_matrix.loc[col1, col2]
            if abs(r) >= 0.7:
                fuerza = "FUERTE"
            elif abs(r) >= 0.4:
                fuerza = "MODERADA"
            elif abs(r) >= 0.2:
                fuerza = "DÉBIL"
            else:
                fuerza = "MUY DÉBIL / NULA"
            
            direccion = "positiva ↗️" if r > 0 else "negativa ↘️"
            print(f"   • {col1:12s} vs {col2:12s}: r = {r:+.3f} → Correlación {fuerza} ({direccion})")
print()

# --- 8.1: Heatmap de correlación ---
# El heatmap es la mejor forma VISUAL de ver correlaciones
# Rojo/azul intenso = correlación fuerte
# Blanco = sin correlación
plt.figure(figsize=(10, 8))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))  # Ocultamos la mitad superior (es simétrica)
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.3f', cmap='RdBu_r', 
            center=0, vmin=-1, vmax=1, square=True, 
            linewidths=1, cbar_kws={'label': 'Coeficiente de Correlación'})
plt.title('🧮 Matriz de Correlación entre Variables Numéricas', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('docs/06_heatmap_correlacion.png', dpi=120, bbox_inches='tight')
plt.close()
print("   ✅ Gráfico guardado: docs/06_heatmap_correlacion.png")
print()

# ============================================================
# PASO 9: PAIRPLOT — El todoterreno del EDA
# ============================================================
# El pairplot (o scatter matrix) muestra TODAS las combinaciones
# de variables en una sola figura:
#   - Diagonal: histograma de cada variable
#   - Fuera de diagonal: scatter plot de cada par
# Es IDEAL para detectar patrones visualmente
# ============================================================

print("─" * 70)
print("🔬 PAIRPLOT — Relaciones múltiples")
print("─" * 70)

# Creamos un pairplot coloreado por segmento de cliente
# Esto nos permite ver si los segmentos se comportan diferente
# Advertencia: el pairplot puede tardar un poco en datasets grandes
print("\n   ⏳ Generando pairplot (puede tomar unos segundos)...")

# Preparamos datos sin nulos para el pairplot
df_pair = df[cols_corr + ['segmento']].dropna(subset=cols_corr)
# Coloreamos los puntos por segmento para ver diferencias
g = sns.pairplot(df_pair, vars=cols_corr, hue='segmento', 
                 palette='Set1', plot_kws={'alpha': 0.6, 's': 50, 'edgecolor': 'none'},
                 diag_kws={'alpha': 0.6})
g.fig.suptitle('🔬 Pairplot — Todas las relaciones (coloreado por segmento)', 
               fontsize=16, fontweight='bold', y=1.02)

# Ajustamos posición del título
g.fig.subplots_adjust(top=0.95)
g.savefig('docs/07_pairplot_completo.png', dpi=120, bbox_inches='tight')
plt.close()
print("   ✅ Gráfico guardado: docs/07_pairplot_completo.png")
print()

# ============================================================
# PASO 10: DETECCIÓN DE OUTLIERS (MÉTODO CIENTÍFICO)
# ============================================================
# Los outliers (valores atípicos) pueden ser:
#   ✅ Errores de medición (hay que corregirlos)
#   ✅ Datos reales pero extremos (hay que decidir si incluirlos)
#   ✅ Fraude o anomalías (los más interesantes!)
#
# Usamos el método IQR (Rango Intercuartílico):
#   Un valor es outlier si está fuera de [Q1 - 1.5*IQR, Q3 + 1.5*IQR]
# ============================================================

print("─" * 70)
print("🎯 DETECCIÓN DE OUTLIERS — Método IQR")
print("─" * 70)

resultados_outliers = {}

for var in ['total', 'precio', 'stock']:
    print(f"\n   📊 Analizando outliers en: {var.upper()}")
    
    datos = df[var].dropna()
    Q1 = datos.quantile(0.25)
    Q3 = datos.quantile(0.75)
    IQR = Q3 - Q1
    
    # Límites para considerar algo como outlier
    limite_inferior = Q1 - 1.5 * IQR
    limite_superior = Q3 + 1.5 * IQR
    
    # Identificamos los outliers
    outliers = datos[(datos < limite_inferior) | (datos > limite_superior)]
    n_outliers = len(outliers)
    pct_outliers = n_outliers / len(datos) * 100
    
    print(f"      Q1: {Q1:.2f}")
    print(f"      Q3: {Q3:.2f}")
    print(f"      IQR: {IQR:.2f}")
    print(f"      Límite inferior: {limite_inferior:.2f}")
    print(f"      Límite superior: {limite_superior:.2f}")
    print(f"      Outliers encontrados: {n_outliers} ({pct_outliers:.1f}% de los datos)")
    
    if len(outliers) > 0:
        print(f"      Valores outlier: {sorted(outliers.values)[:10]}...")
    
    resultados_outliers[var] = {
        'Q1': Q1, 'Q3': Q3, 'IQR': IQR,
        'lim_inf': limite_inferior, 'lim_sup': limite_superior,
        'n_outliers': n_outliers
    }

print()

# --- 10.1: Visualización de outliers con subplots ---
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

for ax, var in zip(axes, ['total', 'precio', 'stock']):
    datos = df[var].dropna()
    r = resultados_outliers[var]
    
    # Boxplot con los outliers destacados
    bp = ax.boxplot(datos, patch_artist=True, widths=0.3)
    bp['boxes'][0].set_facecolor(COLORES[0])
    bp['boxes'][0].set_alpha(0.7)
    bp['fliers'][0].set_marker('o')
    bp['fliers'][0].set_markerfacecolor('red')
    bp['fliers'][0].set_markersize(8)
    
    # Líneas punteadas para los límites
    ax.axhline(y=r['lim_inf'], color='orange', linestyle='--', linewidth=2, label=f'Límite inf: {r["lim_inf"]:.1f}')
    ax.axhline(y=r['lim_sup'], color='red', linestyle='--', linewidth=2, label=f'Límite sup: {r["lim_sup"]:.1f}')
    
    ax.set_title(f'{var.upper()}\n({r["n_outliers"]} outliers = {r["n_outliers"]/len(datos)*100:.1f}%)', 
                 fontsize=13, fontweight='bold')
    ax.set_xticklabels([var.upper()])
    ax.legend(fontsize=8)

plt.suptitle('🎯 Detección de Outliers — Método IQR (Q1 - 1.5×IQR, Q3 + 1.5×IQR)', 
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('docs/08_outliers_iqr.png', dpi=120, bbox_inches='tight')
plt.close()
print("   ✅ Gráfico guardado: docs/08_outliers_iqr.png")
print()

# ============================================================
# PASO 11: INSIGHTS DE NEGOCIO
# ============================================================
# El EDA no es solo matemáticas — tiene que CONTAR UNA HISTORIA.
# Estos hallazgos ayudan a tomar decisiones de negocio.
# ============================================================

print("─" * 70)
print("💡 INSIGHTS DE NEGOCIO")
print("─" * 70)

# Insight 1: ¿Qué segmento da más ingresos?
ingreso_segmento = df.groupby('segmento')['total'].sum()
seg_top = ingreso_segmento.idxmax()
seg_pct = ingreso_segmento[seg_top] / ingreso_segmento.sum() * 100
print(f"\n   🏆 El segmento {seg_top} genera ${ingreso_segmento[seg_top]:,.2f} "
      f"({seg_pct:.1f}% del total de ingresos)")

# Insight 2: ¿Qué ciudad tiene el ticket promedio más alto?
ticket_ciudad = df.groupby('ciudad')['total'].mean()
ciudad_top = ticket_ciudad.idxmax()
print(f"   💰 {ciudad_top} tiene el ticket promedio más alto: ${ticket_ciudad[ciudad_top]:.2f}")

# Insight 3: ¿Categoría más vendida?
cat_vendida = df['categoria'].value_counts()
cat_top = cat_vendida.index[0]
print(f"   📦 La categoría más popular es '{cat_top}' ({cat_vendida[cat_top]} pedidos)")

# Insight 4: ¿Relación precio-puntuación?
corr_precio_punt = corr_matrix.loc['precio', 'puntuacion'] if 'precio' in corr_matrix.index and 'puntuacion' in corr_matrix.columns else 0
if abs(corr_precio_punt) < 0.1:
    print(f"   ⭐ El precio NO tiene correlación con la puntuación (r = {corr_precio_punt:.3f})")
    print(f"      → Los clientes no puntúan mejor solo porque el producto sea caro")
else:
    print(f"   ⭐ Precio vs Puntuación: r = {corr_precio_punt:.3f}")

# Insight 5: Tasa de reseñas
tasa_resenas = (len(df) - df['puntuacion'].isnull().sum()) / len(df) * 100
print(f"   📝 Solo el {tasa_resenas:.1f}% de los pedidos tienen reseña")
print(f"      → Esto es información valiosa: ¿cómo incentivar más reseñas?")
print()

# ============================================================
# PASO 12: RESUMEN Y CONCLUSIÓN
# ============================================================

print("=" * 70)
print("🧠 RESUMEN — HOY APRENDISTE:")
print("=" * 70)
print("""
  1️⃣  PRIMER VISTAZO (head, info, dtypes, describe)
      → A conocer la estructura general de cualquier dataset

  2️⃣  VALORES NULOS (isnull, heatmap de nulos)
      → A detectar datos faltantes y entender por qué existen

  3️⃣  DISTRIBUCIONES (histogramas, media, mediana, skewness)
      → A entender cómo se comporta cada variable numérica

  4️⃣  CATEGÓRICAS (value_counts, gráficos de barras)
      → A analizar frecuencias y dominancia de categorías

  5️⃣  RELACIONES (groupby, boxplots, violinplots, scatter)
      → A encontrar conexiones entre variables

  6️⃣  CORRELACIÓN (matriz de correlación, heatmap)
      → A medir relaciones lineales entre variables numéricas

  7️⃣  PAIRPLOT (scatter matrix con seaborn)
      → A visualizar TODAS las relaciones de un solo golpe

  8️⃣  OUTLIERS (boxplots, método IQR)
      → A detectar valores atípicos de forma científica

  9️⃣  INSIGHTS DE NEGOCIO
      → A convertir datos en decisiones (lo que más importa)
""")
print(f"   📊 Gráficos generados: {len([f for f in os.listdir('docs') if f.endswith('.png')])}")
print(f"   📁 Todos los gráficos en: docs/")
print(f"   📁 Dataset analizado: data/dataset_completo.csv")
print()
print("🚀 PRÓXIMO PROYECTO SUGERIDO:")
print("   Feature Engineering — Crear nuevas variables a partir de los datos existentes")
print("   (fechas, agregaciones, ratios, bins, encoding de categorías)")
print()

print("¡Excelente trabajo! El EDA es la base de todo ML. 🧠🔥")
