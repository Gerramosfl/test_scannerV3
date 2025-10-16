# OMR ArUco Scanner (GUI)

GUI en Python (Tkinter + OpenCV) para escanear y **calificar hojas de respuestas** mediante:
- **ArUco** en las cuatro esquinas (orientación + homografía).
- **Rectificación** a tamaño físico (carta por defecto).
- **OMR** mediante umbralización y análisis de relleno por alternativa.
- **Calibración interactiva por clics** (1 o 2 columnas) para ajustar el **layout en mm** al diseño exacto de tu hoja.

## Requisitos
Python 3.9+
```bash
pip install -r requirements.txt
```

## Uso
```bash
python omr_scanner_gui.py
```
1. Pestaña **Cámara** → encender cámara. Cuando se detecten 4 ArUco (TL=0, TR=1, BL=2, BR=3) verás la hoja **rectificada**.
2. Pestaña **Config hoja** → **Calibrar (1 columna)** o **Calibrar (2 columnas)** y sigue los clics (ventana 1:1).
3. (Opcional) Cargar **clave** (`answer_key_template.csv`) con columnas `question,answer`.
4. **Capturar y calificar** → exporta resultados a **CSV**.

## Parámetros clave (JSON)
- `page_size_mm`: por defecto `[215.9, 279.4]` (carta).
- `dpi`: resolución de la imagen rectificada (300 recomendado; bajar a 200 si necesitas más velocidad).
- `aruco`: diccionario, IDs por esquina, tamaño de marcador (`marker_size_mm=15`) y `offset_mm=7`.
- `grids`: define tu rejilla en **mm** (posición inicial, tamaño de óvalo, separaciones, columnas).

## Consejos
- Imprime siempre en **Tamaño real / 100%**.
- Si aparecen muchas respuestas **AMBIGUO**, sube `decision.min_margin`.
- Si salen **VACÍO**, baja `decision.min_fill` ligeramente.
- Para máxima precisión, calibra la cámara (distorsión) en entornos exigentes.

---
**Licencia:** MIT
