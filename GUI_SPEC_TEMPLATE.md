# Especificación funcional — OMR ArUco Scanner (GUI)

> **Objetivo**: definir de forma operativa y verificable cómo debe verse y funcionar la GUI para escanear y calificar hojas de respuestas con marcadores ArUco.
> **Versión del documento**: 2025-10-16

---

## 1. Visión y metas
- **Qué problema resuelve** (1–2 frases).
- **Quiénes lo usan** (profesor/a, ayudantes, alumnos).
- **Resultado esperado** (p. ej., calificación instantánea con evidencia guardada).

## 2. Entorno objetivo
- **SO**: Windows 10/11 (PowerShell + VS Code).
- **Python**: 3.9–3.12
- **Cámara**: integrada/USB, ≥720p (DirectShow).
- **Impresión**: Carta 215.9×279.4 mm, 4 ArUco (TL=0, TR=1, BL=2, BR=3), lado 15 mm, offset 7 mm.
- **Rendimiento meta**: ≥10 fps en vista previa; captura+calificación ≤ 1.0 s.

## 3. Entradas y salidas
### Entradas
- **Cámara**: frames RGB.
- **Config JSON**: layout en mm (ver §7).
- **Clave CSV**: columnas `question,answer` (A–E, etc.).
- **(Opcional)** ID estudiante: QR o entrada manual.

### Salidas
- **CSV de resultados**: `timestamp, question, chosen_label, status, answer, correct, student_id, sheet_id, version`.
- **PNG/JPG**: evidencia `rectificada` y `overlay`.
- **Log**: TXT o CSV con errores y tiempos (opcional).

## 4. Flujos de usuario (UI/UX)
### 4.1 Flujo “Escanear y calificar”
1. Vista previa muestra “Buscando marcadores…” → “Marcadores OK” cuando detecta TL=0, TR=1, BL=2, BR=3.
2. **(Auto-disparo)** cuando la detección es estable N de los últimos M frames (p. ej., 8/10) → captura.
3. Rectificación (homografía) → OMR → comparación con clave.
4. Sonido “beep” y banner “Calificado: 43/50 (0.86) – OK/AMBIGUO/VACÍO por pregunta”.
5. Guardado automático de CSV/PNG(s) en carpeta configurada.

**Criterios de aceptación (GWT)**  
- *Dado* 4 marcadores visibles y estables, *cuando* activo auto-disparo, *entonces* se captura una sola vez y se guardan los artefactos sin duplicados.
- *Dado* una clave válida, *cuando* se califica, *entonces* el CSV contiene `correct=true/false` coherente con `answer`.

### 4.2 Flujo “Calibración”
- Botones: **Calibrar (1 columna)** / **(2 columnas)**. Ventana 1:1 guía 4–5 clics.  
- El JSON se actualiza con `top_left_mm, bubble_size_mm, option_dx_mm, row_dy_mm, [col_dx_mm]`.

**Criterios (GWT)**  
- *Dado* una hoja V3 impresa, *cuando* completo los clics, *entonces* el overlay coincide ±1 px con los óvalos.

### 4.3 Flujo “Resultados y exportación”
- Vista “Resultados” lista preguntas con `status` y `margin`.
- Botón **Exportar CSV** (incluye clave si cargada).

## 5. Mapa de pantallas
- **Izquierda**: Canvas de vídeo/overlay; estados (sin cámara / detectando / marcadores OK).
- **Derecha**: Pestañas
  - *Cámara*: índice de cámara, encender/apagar, “Capturar y calificar”, estado.
  - *Config hoja*: editor JSON + calibración por clics.
  - *Clave*: cargar/mostrar estado.
  - *Resultados*: lista; exportar CSV.

## 6. Lista de funcionalidades (priorizadas — MoSCoW)
### MUST (v1)
- Detección ArUco TL=0, TR=1, BL=2, BR=3; rectificación a Carta @ 300 dpi.
- Calibración por clics (1/2 columnas).
- OMR por “fill ratio” con `min_fill` y `min_margin` configurables.
- Carga de clave y calificación por pregunta; puntaje total.
- Exportar CSV; guardar PNG de overlay/rectificada.
- Auto-disparo con estabilidad N/M y cooldown.

### SHOULD
- Selección de carpeta de salida; plantilla de nombre: `YYYYmmdd_HHMMSS_{studentId}`.
- Sonidos (beep OK, error).
- Modo oscuro/claro.

### COULD
- Soporte A4.
- Undistort (calibración intrínseca) para mayor precisión.
- Lectura de QR (`student_id`).

### WON’T (por ahora)
- Red neuronal para OMR.
- Multi-página / duplex.

## 7. Datos y formatos
### 7.1 Config JSON (fragmento)
```json
{
  "page_size_mm": [215.9, 279.4],
  "dpi": 300,
  "aruco": {
    "dict": "DICT_4X4_50",
    "ids": {"TL":0,"TR":1,"BL":2,"BR":3},
    "marker_size_mm": 15.0,
    "offset_mm": 7.0
  },
  "grids": [{ "name":"main", "questions":50, "options":["A","B","C","D","E"],
              "top_left_mm":[25.0,40.0],
              "bubble_size_mm":[5.0,5.0],
              "option_dx_mm":8.0,
              "row_dy_mm":7.0,
              "columns":1, "rows":50, "col_dx_mm":60.0,
              "block_row_gap_mm":0.0 }],
  "thresholding": {"method":"otsu","invert":true},
  "decision": {"min_fill":0.25,"min_margin":0.10}
}
```

### 7.2 Clave CSV
```
question,answer
1,A
2,C
...
```

### 7.3 CSV de resultados (salida)
Campos mínimos: `timestamp,question,chosen_label,status,answer,correct,student_id,sheet_id,version`.

## 8. Requisitos no funcionales
- **Latencia**: captura→CSV ≤ 1.0 s en portátil i5/i7 reciente.
- **Robustez**: tolerar iluminación variable; umbrales ajustables.
- **Verificabilidad**: overlay guardado por cada captura.
- **Portabilidad**: Windows; (opcional) Linux.

## 9. Manejo de errores
- Sin cámara / acceso denegado.
- Menos de 4 marcadores.
- Clave inconsistente (p. ej., preguntas fuera de rango).
- Estructura de JSON inválida.

## 10. Métricas de calidad
- % preguntas **AMBIGUO** / **VACÍO** por hoja.
- Tiempo medio de calificación.
- Tasa de reintentos (auto-disparo).

## 11. Roadmap & entregables
- **v1.0** (MUST): fecha objetivo, criterios de aceptación completos.
- **v1.1** (SHOULD): QR + carpeta de salida + sonidos.
- **v1.2** (COULD): A4 + undistort.

## 12. Anexos
- Bocetos/imagenes de UI (pegar aquí).
- Ejemplos de hojas escaneadas (antes/después).
