OMR ArUco Scanner (GUI) — con Calibración Interactiva
=====================================================

Requisitos:
    pip install opencv-contrib-python numpy pillow pandas

Archivos clave:
- omr_scanner_gui.py          -> GUI principal (Tkinter + OpenCV) con calibración por clics.
- sheet_config_template.json  -> Config de ejemplo (mm).
- answer_key_template.csv     -> Clave de ejemplo (question,answer).

Pasos:
1) Imprime tu hoja con ArUco: TL=0, TR=1, BL=2, BR=3; lado 15 mm; offset 7 mm.
2) Ejecuta:
       python omr_scanner_gui.py
3) Enciende la cámara (pestaña “Cámara”). Cuando diga “marcadores OK”, verás la hoja rectificada.
4) Ve a “Config hoja” y usa:
   - **Calibrar (1 columna)** o **Calibrar (2 columnas)**.
   - Sigue los clics en la ventana 1:1:
       (1) Esquina sup-izq del primer óvalo (Q1-A)
       (2) Esquina inf-der del mismo óvalo
       (3) Centro del siguiente óvalo a la derecha (Q1-B)
       (4) Centro del óvalo de la fila siguiente (Q2-A)
       (5) [solo 2 columnas] Centro del primer óvalo de la segunda columna (Q26-A aprox.)
   - El JSON se actualiza automáticamente con: top_left_mm, bubble_size_mm, option_dx_mm, row_dy_mm, [col_dx_mm].
5) (Opcional) Ajusta “questions”, “rows”, “columns”, “block_row_gap_mm” si tu hoja tiene bloques de 10 en 10 o varias columnas.
6) Carga la clave en la pestaña “Clave” (CSV con columnas: question,answer).
7) Pulsa “Capturar y calificar”. Puedes exportar CSV desde “Resultados”.

Notas:
- Toda la geometría está en **mm** y se convierte a px con el **DPI** (300 por defecto).
- Si la lectura es lenta, baja a 200 DPI en el JSON (menor tamaño de imagen rectificada).
- Para máxima precisión, calibra la cámara y corrige distorsión antes de detectar ArUco.
