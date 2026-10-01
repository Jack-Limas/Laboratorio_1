"""
=============================================================================
LABORATORIO 1: MINERÍA DE DATOS - METODOLOGÍA CRISP-DM
Script Principal de Análisis, Modelado, Evaluación y Exportación
=============================================================================
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import joblib

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 0.8

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
PLOTS_DIR = os.path.join(BASE_DIR, 'static', 'plots')

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)


def print_section(title):
    print("\n" + "=" * 70)
    print(f" {title.upper()} ")
    print("=" * 70)


# =============================================================================
# EJERCICIO 1: MODELADO DEL PRECIO DEL DÓLAR
# =============================================================================
def analizar_precio_dolar():
    print_section("Ejercicio 1: Predicción del Precio del Dólar (CRISP-DM)")

    csv_path = os.path.join(DATA_DIR, 'dolar_data.csv')
    df = pd.read_csv(csv_path)
    print(f"[*] Datos cargados: {df.shape[0]} registros y {df.shape[1]} columnas.")

    features = ['Dia', 'Inflacion', 'Tasa_interes']
    target = 'Precio_Dolar'

    X = df[features]
    y = df[target]

    modelo = LinearRegression()
    modelo.fit(X, y)
    y_pred = modelo.predict(X)

    mse = mean_squared_error(y, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y, y_pred)

    coef_df = pd.DataFrame({
        'Variable': features,
        'Coeficiente': modelo.coef_,
        'Impacto_Abs': np.abs(modelo.coef_),
        'Std_Feature': X.std().values,
        'Coef_Estandarizado': modelo.coef_ * (X.std().values / y.std()),
        'Abs_Beta': np.abs(modelo.coef_ * (X.std().values / y.std()))
    }).sort_values(by='Impacto_Abs', ascending=False)

    print("\n[+] Resumen de Métricas:")
    print(f"    - MSE  : {mse:.4f}")
    print(f"    - RMSE : {rmse:.4f}")
    print(f"    - R^2  : {r2:.4f} ({r2*100:.2f}% de la varianza explicada)")
    print(f"    - Intercepto (b0): {modelo.intercept_:.4f}")
    print("\n[+] Coeficientes del Modelo:")
    for _, row in coef_df.iterrows():
        print(f"    - {row['Variable']}: {row['Coeficiente']:.4f} (Beta Est.: {row['Coef_Estandarizado']:.4f})")

    var_mayor_impacto = coef_df.sort_values(by='Abs_Beta', ascending=False).iloc[0]['Variable']
    print(f"\n[+] Variable con mayor impacto relativo: {var_mayor_impacto}")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), dpi=300)
    fig.suptitle('Ejercicio 1: Análisis y Regresión Lineal - Precio del Dólar', fontsize=16, fontweight='bold', y=1.02)

    colores = ['#1f77b4', '#ff7f0e', '#2ca02c']
    for idx, (col, color) in enumerate(zip(features, colores)):
        ax = axes[idx]
        sns.regplot(
            data=df, x=col, y=target, ax=ax,
            scatter_kws={'alpha': 0.55, 'color': color, 's': 30},
            line_kws={'color': '#d62728', 'linewidth': 2}
        )
        ax.set_title(f'{col} vs {target}\nCoef: {modelo.coef_[idx]:.4f}', fontsize=12, fontweight='bold')
        ax.set_xlabel(col, fontsize=11)
        ax.set_ylabel(target if idx == 0 else '', fontsize=11)
        ax.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plot_file = os.path.join(PLOTS_DIR, 'dolar_plots.png')
    plt.savefig(plot_file, bbox_inches='tight')
    plt.close()
    print(f"[+] Gráfico guardado en: {plot_file}")

    model_payload = {
        'model': modelo,
        'features': features,
        'target': target,
        'intercept': float(modelo.intercept_),
        'coefficients': dict(zip(features, [float(c) for c in modelo.coef_])),
        'metrics': {'mse': float(mse), 'rmse': float(rmse), 'r2': float(r2)},
        'stats': {
            'min': X.min().to_dict(),
            'max': X.max().to_dict(),
            'mean': X.mean().to_dict()
        }
    }
    model_file = os.path.join(MODELS_DIR, 'modelo_dolar.joblib')
    joblib.dump(model_payload, model_file)
    print(f"[+] Modelo exportado en: {model_file}")

    return model_payload


# =============================================================================
# EJERCICIO 2: MODELADO DE NIVELES DE GLUCOSA
# =============================================================================
def analizar_niveles_glucosa():
    print_section("Ejercicio 2: Predicción de Niveles de Glucosa (CRISP-DM)")

    csv_path = os.path.join(DATA_DIR, 'glucosa_data.csv')
    df = pd.read_csv(csv_path)
    print(f"[*] Datos cargados: {df.shape[0]} registros y {df.shape[1]} columnas.")

    features = ['Edad', 'IMC', 'Actividad_Fisica']
    target = 'Nivel_Glucosa'

    X = df[features]
    y = df[target]

    modelo = LinearRegression()
    modelo.fit(X, y)
    y_pred = modelo.predict(X)

    mse = mean_squared_error(y, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y, y_pred)

    coef_df = pd.DataFrame({
        'Variable': features,
        'Coeficiente': modelo.coef_,
        'Impacto_Abs': np.abs(modelo.coef_),
        'Std_Feature': X.std().values,
        'Coef_Estandarizado': modelo.coef_ * (X.std().values / y.std())
    }).sort_values(by='Impacto_Abs', ascending=False)

    print("\n[+] Resumen de Métricas:")
    print(f"    - MSE  : {mse:.4f}")
    print(f"    - RMSE : {rmse:.4f}")
    print(f"    - R^2  : {r2:.4f} ({r2*100:.2f}% de la varianza explicada)")
    print(f"    - Intercepto (b0): {modelo.intercept_:.4f}")
    print("\n[+] Coeficientes del Modelo:")
    for _, row in coef_df.iterrows():
        tipo_efecto = "POSITIVO (incrementa glucosa)" if row['Coeficiente'] > 0 else "NEGATIVO (reduce glucosa)"
        print(f"    - {row['Variable']}: {row['Coeficiente']:.4f} -> Efecto {tipo_efecto}")

    pos_impact = coef_df[coef_df['Coeficiente'] > 0].sort_values(by='Coeficiente', ascending=False)
    neg_impact = coef_df[coef_df['Coeficiente'] < 0].sort_values(by='Coeficiente', ascending=True)

    var_mayor_pos = pos_impact.iloc[0]['Variable'] if not pos_impact.empty else "Ninguna"
    var_mayor_neg = neg_impact.iloc[0]['Variable'] if not neg_impact.empty else "Ninguna"
    print(f"\n[+] Mayor impacto positivo: {var_mayor_pos}")
    print(f"[+] Mayor impacto negativo: {var_mayor_neg}")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), dpi=300)
    fig.suptitle('Ejercicio 2: Factores Determinantes en el Nivel de Glucosa', fontsize=16, fontweight='bold', y=1.02)

    colores = ['#9467bd', '#8c564b', '#17becf']
    for idx, (col, color) in enumerate(zip(features, colores)):
        ax = axes[idx]
        sns.regplot(
            data=df, x=col, y=target, ax=ax,
            scatter_kws={'alpha': 0.45, 'color': color, 's': 28},
            line_kws={'color': '#e377c2' if modelo.coef_[idx] > 0 else '#2ca02c', 'linewidth': 2}
        )
        ax.set_title(f'{col} vs {target}\nCoef: {modelo.coef_[idx]:.4f}', fontsize=12, fontweight='bold')
        ax.set_xlabel(col, fontsize=11)
        ax.set_ylabel(target if idx == 0 else '', fontsize=11)
        ax.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plot_file = os.path.join(PLOTS_DIR, 'glucosa_plots.png')
    plt.savefig(plot_file, bbox_inches='tight')
    plt.close()
    print(f"[+] Gráfico guardado en: {plot_file}")

    model_payload = {
        'model': modelo,
        'features': features,
        'target': target,
        'intercept': float(modelo.intercept_),
        'coefficients': dict(zip(features, [float(c) for c in modelo.coef_])),
        'metrics': {'mse': float(mse), 'rmse': float(rmse), 'r2': float(r2)},
        'impact': {
            'mayor_positivo': var_mayor_pos,
            'mayor_negativo': var_mayor_neg
        },
        'stats': {
            'min': X.min().to_dict(),
            'max': X.max().to_dict(),
            'mean': X.mean().to_dict()
        }
    }
    model_file = os.path.join(MODELS_DIR, 'modelo_glucosa.joblib')
    joblib.dump(model_payload, model_file)
    print(f"[+] Modelo exportado en: {model_file}")

    return model_payload


# =============================================================================
# EJERCICIO 3: MODELADO DE CONSUMO DE ENERGÍA
# =============================================================================
def analizar_consumo_energia():
    print_section("Ejercicio 3: Predicción de Consumo de Energía Eléctrica (CRISP-DM)")

    csv_path = os.path.join(DATA_DIR, 'energia_data.csv')
    df = pd.read_csv(csv_path)
    print(f"[*] Datos cargados: {df.shape[0]} registros y {df.shape[1]} columnas.")

    features = ['Temperatura', 'Hora', 'Dia_Semana']
    target = 'Consumo_Energia'

    X = df[features]
    y = df[target]

    modelo = LinearRegression()
    modelo.fit(X, y)
    y_pred = modelo.predict(X)

    mse = mean_squared_error(y, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y, y_pred)

    coef_df = pd.DataFrame({
        'Variable': features,
        'Coeficiente': modelo.coef_,
        'Impacto_Abs': np.abs(modelo.coef_),
        'Std_Feature': X.std().values,
        'Coef_Estandarizado': modelo.coef_ * (X.std().values / y.std()),
        'Abs_Beta': np.abs(modelo.coef_ * (X.std().values / y.std()))
    }).sort_values(by='Impacto_Abs', ascending=False)

    print("\n[+] Resumen de Métricas:")
    print(f"    - MSE  : {mse:.4f}")
    print(f"    - RMSE : {rmse:.4f}")
    print(f"    - R^2  : {r2:.4f} ({r2*100:.2f}% de la varianza explicada)")
    print(f"    - Intercepto (b0): {modelo.intercept_:.4f}")
    print("\n[+] Coeficientes del Modelo:")
    for _, row in coef_df.iterrows():
        print(f"    - {row['Variable']}: {row['Coeficiente']:.4f} (Beta Est.: {row['Coef_Estandarizado']:.4f})")

    var_mayor_impacto = coef_df.sort_values(by='Abs_Beta', ascending=False).iloc[0]['Variable']
    print(f"\n[+] Variable con mayor impacto global en consumo: {var_mayor_impacto}")

    fig = plt.figure(figsize=(18, 10), dpi=300)
    gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.25)
    fig.suptitle('Ejercicio 3: Matriz de Correlación y Regresión de Consumo Energético', fontsize=16, fontweight='bold', y=0.96)

    ax_heat = fig.add_subplot(gs[0, :])
    corr_matrix = df.corr()
    sns.heatmap(
        corr_matrix, annot=True, fmt='.3f', cmap='coolwarm',
        vmin=-1, vmax=1, linewidths=1.2, linecolor='white',
        cbar_kws={'label': 'Coeficiente de Correlación de Pearson'},
        ax=ax_heat
    )
    ax_heat.set_title('Matriz de Correlación de Pearson (Variables vs Consumo)', fontsize=13, fontweight='bold')

    colores = ['#e6550d', '#31a354', '#756bb1']
    sample_df = df.sample(min(len(df), 2000), random_state=42)
    for idx, (col, color) in enumerate(zip(features, colores)):
        ax = fig.add_subplot(gs[1, idx])
        sns.regplot(
            data=sample_df, x=col, y=target, ax=ax,
            scatter_kws={'alpha': 0.35, 'color': color, 's': 20},
            line_kws={'color': '#b30000', 'linewidth': 2}
        )
        ax.set_title(f'{col} vs {target}\nCoef: {modelo.coef_[idx]:.4f}', fontsize=12, fontweight='bold')
        ax.set_xlabel(col, fontsize=11)
        ax.set_ylabel(target if idx == 0 else '', fontsize=11)
        ax.grid(True, linestyle='--', alpha=0.6)

    plot_file = os.path.join(PLOTS_DIR, 'energia_plots.png')
    plt.savefig(plot_file, bbox_inches='tight')
    plt.close()
    print(f"[+] Gráfico guardado en: {plot_file}")

    model_payload = {
        'model': modelo,
        'features': features,
        'target': target,
        'intercept': float(modelo.intercept_),
        'coefficients': dict(zip(features, [float(c) for c in modelo.coef_])),
        'metrics': {'mse': float(mse), 'rmse': float(rmse), 'r2': float(r2)},
        'stats': {
            'min': X.min().to_dict(),
            'max': X.max().to_dict(),
            'mean': X.mean().to_dict()
        }
    }
    model_file = os.path.join(MODELS_DIR, 'modelo_energia.joblib')
    joblib.dump(model_payload, model_file)
    print(f"[+] Modelo exportado en: {model_file}")

    return model_payload


def main():
    print("Iniciando Pipeline Automatizado de Minería de Datos (CRISP-DM)...")
    res_dolar = analizar_precio_dolar()
    res_glucosa = analizar_niveles_glucosa()
    res_energia = analizar_consumo_energia()

    summary = {
        "dolar": res_dolar['metrics'],
        "glucosa": res_glucosa['metrics'],
        "energia": res_energia['metrics']
    }

    summary_file = os.path.join(MODELS_DIR, 'resumen_metricas.json')
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=4)

    print_section("Proceso completado con éxito")
    print(f"Modelos guardados en: {MODELS_DIR}")
    print(f"Gráficos guardados en: {PLOTS_DIR}")
    print(f"Resumen de métricas guardado en: {summary_file}")


if __name__ == '__main__':
    main()
