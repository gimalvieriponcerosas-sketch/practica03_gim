# PAVEMENT AI
Aplicación educativa para explorar el deterioro del pavimento rígido y estimar el índice de condición del pavimento (PCI) con regresión lineal.
## Requisitos
- Python 3.10 o posterior
- `pip`
## Instalación y ejecución
```bash
python -m venv .venv
```
En Windows:
```powershell
.venv\Scripts\Activate.ps1
```
En Linux o macOS:
```bash
source .venv/bin/activate
```
Instala dependencias y ejecuta:
```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```
Streamlit mostrará una dirección local para abrir el dashboard en el navegador.
## Funciones
- Dashboard con exploración de registros, ajuste del modelo y métricas R², MAE y RMSE.
- Simulador PCI para antigüedad actual y tres escenarios futuros.
- Registro de inspecciones con tramo, fecha, tránsito, tipo y severidad de falla, PCI observado y fotografías opcionales.
- Descarga de los registros y de las predicciones como CSV.
- Carga de un CSV propio. Columnas esperadas: `tramo`, `fecha`, `antiguedad`, `transito`, `patologia`, `severidad`, `pci`.
- La pestaña de datos permite descargar un CSV de ejemplo con 60 filas simuladas.
## Datos y limitaciones
`datos/pavimento_simulado.csv` es un conjunto **ficticio y educativo**, generado para esta práctica; no representa mediciones de Huánuco ni una inspección real. La división de entrenamiento y prueba es aleatoria con semilla fija. El modelo utiliza únicamente antigüedad como variable predictora, así que sus métricas no demuestran desempeño en vías reales. Se limita la salida mostrada al rango PCI 0–100 y se advierte cuando una predicción supera el historial disponible.

El PCI debe ser determinado mediante evaluación técnica de acuerdo con la metodología aplicable. Esta aplicación no reemplaza la inspección de campo ni el criterio profesional.
