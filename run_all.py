"""
=============================================================================
LABORATORIO 1: MINERÍA DE DATOS - ORQUESTADOR GENERAL (run_all.py)
1. Ejecuta el pipeline de entrenamiento y análisis (main_analysis.py).
2. Muestra un resumen ejecutivo por consola con métricas clave.
3. Inicia el servidor web interactivo Streamlit (app.py).
=============================================================================
"""

import sys
import os
import subprocess
import json
import time

if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

def imprimir_banner():
    print("""
    =======================================================================
               LABORATORIO 1: MINERIA DE DATOS & BUSINESS INTELLIGENCE
                      PIPELINE CRISP-DM COMPLETO AUTOMATIZADO
    =======================================================================
    """)

def ejecutar_analisis():
    print("[1/3] Ejecutando analisis, modelado y exportacion de artefactos...")
    script_analisis = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'main_analysis.py')
    
    inicio = time.time()
    resultado = subprocess.run([sys.executable, script_analisis], capture_output=False)
    tiempo_total = time.time() - inicio

    if resultado.returncode != 0:
        print(f"\n[!] ERROR al ejecutar {script_analisis}. Codigo de salida: {resultado.returncode}")
        sys.exit(resultado.returncode)

    print(f"\n[OK] Entrenamiento y generacion de graficos completados con exito en {tiempo_total:.2f}s.")

def mostrar_resumen():
    print("\n" + "=" * 70)
    print(" [2/3] RESUMEN EJECUTIVO DE MODELOS ENTRENADOS ")
    print("=" * 70)

    base_dir = os.path.dirname(os.path.abspath(__file__))
    summary_file = os.path.join(base_dir, 'models', 'resumen_metricas.json')

    if os.path.exists(summary_file):
        with open(summary_file, 'r', encoding='utf-8') as f:
            metricas = json.load(f)

        print(f"\n1. PRECIO DEL DÓLAR:")
        print(f"   - R²:   {metricas['dolar']['r2']:.4f} ({metricas['dolar']['r2']*100:.2f}% de varianza)")
        print(f"   - MSE:  {metricas['dolar']['mse']:.2f}")
        print(f"   - RMSE: {metricas['dolar']['rmse']:.2f} COP")

        print(f"\n2. NIVELES DE GLUCOSA:")
        print(f"   - R²:   {metricas['glucosa']['r2']:.4f} ({metricas['glucosa']['r2']*100:.2f}% de varianza)")
        print(f"   - MSE:  {metricas['glucosa']['mse']:.2f}")
        print(f"   - RMSE: {metricas['glucosa']['rmse']:.2f} mg/dL")

        print(f"\n3. CONSUMO DE ENERGÍA:")
        print(f"   - R²:   {metricas['energia']['r2']:.4f} ({metricas['energia']['r2']*100:.2f}% de varianza)")
        print(f"   - MSE:  {metricas['energia']['mse']:.2f}")
        print(f"   - RMSE: {metricas['energia']['rmse']:.2f} kWh")
    else:
        print("[*] No se encontró el archivo de resumen de métricas.")

    print("\n" + "=" * 70)

def iniciar_app():
    print("\n[3/3] Iniciando la aplicación web interactiva (Streamlit)...")
    base_dir = os.path.dirname(os.path.abspath(__file__))
    app_file = os.path.join(base_dir, 'app.py')

    print("[*] Servidor web iniciando en http://localhost:8501")
    print("[*] Presiona Ctrl + C en la terminal para detener el servidor.\n")
    
    comando = [sys.executable, "-m", "streamlit", "run", app_file, "--server.headless=false"]
    try:
        subprocess.run(comando)
    except KeyboardInterrupt:
        print("\n[*] Servidor Streamlit detenido por el usuario.")

if __name__ == '__main__':
    imprimir_banner()
    ejecutar_analisis()
    mostrar_resumen()
    iniciar_app()
