#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OMR (Optical Mark Recognition) GUI with ArUco-based homography + Interactive Calibration
Author: ChatGPT (for Gerson)
Requirements:
    pip install opencv-contrib-python numpy pillow pandas
Run:
    python omr_scanner_gui.py
"""
import json
import time
import threading
import os
import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog

# ------------------------- Physical & ArUco helpers -------------------------
def mm_to_px(mm, dpi):
    return mm * dpi / 25.4

def px_to_mm(px, dpi):
    return px * 25.4 / dpi

def get_aruco_dictionary(name="DICT_4X4_50"):
    aruco = cv2.aruco
    mapping = {
        "DICT_4X4_50": aruco.DICT_4X4_50,
        "DICT_4X4_100": aruco.DICT_4X4_100,
        "DICT_5X5_50": aruco.DICT_5X5_50,
        "DICT_6X6_50": aruco.DICT_6X6_50,
        "DICT_ARUCO_ORIGINAL": aruco.DICT_ARUCO_ORIGINAL,
    }
    return aruco.getPredefinedDictionary(mapping.get(name, aruco.DICT_4X4_50))

def detect_aruco_centers(bgr, dict_name="DICT_4X4_50"):
    aruco = cv2.aruco
    dictionary = get_aruco_dictionary(dict_name)
    # DetectorParameters API varies by version
    try:
        params = aruco.DetectorParameters()
    except Exception:
        params = aruco.DetectorParameters_create()
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = aruco.detectMarkers(gray, dictionary, parameters=params)
    out = {}
    if ids is not None:
        ids = ids.flatten()
        for i, c in zip(ids, corners):
            pts = c[0]
            cx, cy = pts[:, 0].mean(), pts[:, 1].mean()
            out[int(i)] = (float(cx), float(cy))
    return out, corners, ids

def compute_homography_from_four_centers(centers_img, ids_map, page_w_mm, page_h_mm, dpi, marker_mm, offset_mm):
    """
    centers_img: dict {id: (cx, cy)} detected in the camera frame
    ids_map: dict like {'TL':0, 'TR':1, 'BL':2, 'BR':3}
    Returns: rectified_image_size (Wpx, Hpx), src_pts(4x2), dst_pts(4x2)
    """
    Wpx = int(round(mm_to_px(page_w_mm, dpi)))
    Hpx = int(round(mm_to_px(page_h_mm, dpi)))
    offs = mm_to_px(offset_mm, dpi)
    half = mm_to_px(marker_mm / 2.0, dpi)

    dst = {
        "TL": (offs + half,           offs + half),
        "TR": (Wpx - offs - half,     offs + half),
        "BL": (offs + half,           Hpx - offs - half),
        "BR": (Wpx - offs - half,     Hpx - offs - half),
    }

    try:
        src_pts = np.float32([
            centers_img[ids_map["TL"]],
            centers_img[ids_map["TR"]],
            centers_img[ids_map["BR"]],
            centers_img[ids_map["BL"]],
        ])
    except KeyError:
        missing = [k for k,v in ids_map.items() if v not in centers_img]
        raise RuntimeError(f"Faltan marcadores: {missing}")

    dst_pts = np.float32([
        dst["TL"],
        dst["TR"],
        dst["BR"],
        dst["BL"],
    ])
    return (Wpx, Hpx), src_pts, dst_pts

# ------------------------- Layout & OMR logic -------------------------
class GridLayout:
    """
    mm-based parametric grid:
    - top-left (mm), bubble size (mm), option_dx (mm), row_dy (mm)
    - questions, options (list of labels), rows/columns (if multi-column)
    """
    def __init__(self, spec):
        self.name = spec.get("name", "main")
        self.questions = int(spec.get("questions", 50))
        self.options = list(spec.get("options", ["A","B","C","D","E"]))
        self.top_left_mm = tuple(spec.get("top_left_mm", [25.0, 40.0]))
        self.bubble_size_mm = tuple(spec.get("bubble_size_mm", [5.0, 5.0]))
        self.option_dx_mm = float(spec.get("option_dx_mm", 8.0))
        self.row_dy_mm = float(spec.get("row_dy_mm", 7.0))
        self.columns = int(spec.get("columns", 1))
        self.rows = int(spec.get("rows", self.questions))
        self.col_dx_mm = float(spec.get("col_dx_mm", 60.0))
        self.block_row_gap_mm = float(spec.get("block_row_gap_mm", 0.0))

    def iter_cells_mm(self):
        """
        Yields tuples: (q_index, opt_index, label, (x_mm, y_mm, w_mm, h_mm))
        Order by questions top-to-bottom, left-to-right (column blocks).
        """
        q_per_col = int(np.ceil(self.questions / self.columns))
        for ci in range(self.columns):
            col_x_mm = self.top_left_mm[0] + ci * self.col_dx_mm
            for ri in range(self.rows):
                q_index = ci * q_per_col + ri
                if q_index >= self.questions:
                    break
                # Optional gap after each 10 questions (visual blocks)
                gap = self.block_row_gap_mm if (ri>0 and (ri % 10)==0) else 0.0
                row_y_mm = self.top_left_mm[1] + ri * (self.row_dy_mm + gap)
                for oi, label in enumerate(self.options):
                    x_mm = col_x_mm + oi * self.option_dx_mm
                    y_mm = row_y_mm
                    w_mm, h_mm = self.bubble_size_mm
                    yield (q_index+1, oi, label, (x_mm, y_mm, w_mm, h_mm))

def grade_rectified(rectified_bgr, cfg):
    """
    rectified_bgr: canonical page image (already warped)
    cfg: full JSON config dict
    Returns: (results_df, overlay_bgr)
    """
    dpi = int(cfg["dpi"])
    gray = cv2.cvtColor(rectified_bgr, cv2.COLOR_BGR2GRAY)

    thr = cfg.get("thresholding", {"method":"otsu","invert":True})
    dec = cfg.get("decision", {"min_fill":0.25, "min_margin":0.1})

    all_rows = []
    overlay = rectified_bgr.copy()

    for grid_spec in cfg["grids"]:
        grid = GridLayout(grid_spec)
        Q = grid.questions
        options = grid.options
        num_opt = len(options)
        q_fills = {q: [0.0]*num_opt for q in range(1, Q+1)}

        for (q, oi, label, (x_mm, y_mm, w_mm, h_mm)) in grid.iter_cells_mm():
            x = int(round(mm_to_px(x_mm, dpi)))
            y = int(round(mm_to_px(y_mm, dpi)))
            w = int(round(mm_to_px(w_mm, dpi)))
            h = int(round(mm_to_px(h_mm, dpi)))
            x2, y2 = x + w, y + h
            x = max(0, x); y = max(0, y)
            crop = gray[y:y2, x:x2]
            if crop.size == 0:
                continue
            # Threshold (Otsu or Adaptive) + small opening
            if thr.get("method","otsu") == "otsu":
                if thr.get("invert", True):
                    _, bw = cv2.threshold(crop, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                else:
                    _, bw = cv2.threshold(crop, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            else:
                bw = cv2.adaptiveThreshold(crop, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                           cv2.THRESH_BINARY_INV if thr.get("invert", True) else cv2.THRESH_BINARY,
                                           31, 5)
            kernel = np.ones((3,3), np.uint8)
            bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel, iterations=1)
            fill = float(np.count_nonzero(bw)) / float(bw.size + 1e-9)
            q_fills[q][oi] = fill

            color = (0,255,0) if fill >= dec.get("min_fill",0.25) else (0,0,255)
            cv2.rectangle(overlay, (x,y), (x2,y2), color, 1)

        for q in range(1, Q+1):
            fills = q_fills[q]
            best_idx = int(np.argmax(fills))
            best = fills[best_idx]
            sorted_vals = sorted(fills, reverse=True)
            margin = (sorted_vals[0] - sorted_vals[1]) if len(sorted_vals) > 1 else sorted_vals[0]
            status = "OK"
            chosen_label = options[best_idx]
            if best < dec.get("min_fill", 0.25):
                status = "VACIO"
                chosen_label = ""
            elif margin < dec.get("min_margin", 0.1):
                status = "AMBIGUO"
            all_rows.append({
                "question": q,
                "chosen_index": best_idx if chosen_label else -1,
                "chosen_label": chosen_label,
                "fills": fills,
                "best": best,
                "margin": margin,
                "status": status,
                "grid": grid.name
            })

    df = pd.DataFrame(all_rows).sort_values(["grid","question"]).reset_index(drop=True)
    return df, overlay

# ------------------------- GUI -------------------------
class OMRGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("OMR ArUco Scanner")
        self.geometry("1250x840")

        self.cap = None
        self.camera_index = 0
        self.running = False
        self.frame = None
        self.rectified = None
        self.overlay = None
        self.last_results = None
        self.answer_key = None
        self.cfg = self.default_config()

        # Calibration state
        self.calibrating = False
        self.calib_points = []  # list of (x,y) in rectified space
        self.calib_mode = None  # 'singlecol' or 'multicol'

        # UI
        self.create_widgets()

        # Start camera by default
        self.toggle_camera()

    # ---------------- UI Elements ----------------
    def create_widgets(self):
        # Left: video
        left = ttk.Frame(self)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(left, bg="black")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind("<Button-1>", self.on_canvas_click)

        # Right: controls
        right = ttk.Notebook(self)
        right.pack(side=tk.RIGHT, fill=tk.Y)

        # Tab 1: Camera
        tab_cam = ttk.Frame(right)
        right.add(tab_cam, text="Cámara")

        ttk.Label(tab_cam, text="Índice de cámara:").pack(anchor="w", padx=10, pady=5)
        self.cam_var = tk.IntVar(value=self.camera_index)
        ttk.Spinbox(tab_cam, from_=0, to=10, textvariable=self.cam_var, width=5).pack(anchor="w", padx=10)

        ttk.Button(tab_cam, text="Encender/Apagar cámara", command=self.toggle_camera).pack(anchor="w", padx=10, pady=5)
        ttk.Button(tab_cam, text="Capturar y calificar", command=self.capture_and_grade).pack(anchor="w", padx=10, pady=5)

        self.status_var = tk.StringVar(value="Estado: esperando cámara...")
        ttk.Label(tab_cam, textvariable=self.status_var, wraplength=260).pack(anchor="w", padx=10, pady=10)

        # Tab 2: Config
        tab_cfg = ttk.Frame(right)
        right.add(tab_cfg, text="Config hoja")

        self.cfg_text = tk.Text(tab_cfg, width=50, height=30)
        self.cfg_text.pack(side=tk.LEFT, padx=5, pady=5)
        self.refresh_cfg_text()

        btns = ttk.Frame(tab_cfg); btns.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)
        ttk.Button(btns, text="Cargar JSON...", command=self.load_cfg).pack(fill=tk.X, pady=3)
        ttk.Button(btns, text="Guardar JSON...", command=self.save_cfg).pack(fill=tk.X, pady=3)
        ttk.Button(btns, text="Aplicar JSON (desde panel)", command=self.apply_cfg_text).pack(fill=tk.X, pady=3)

        ttk.Separator(btns, orient="horizontal").pack(fill=tk.X, pady=8)
        ttk.Label(btns, text="Calibración rápida (clics)").pack(pady=3)

        ttk.Button(btns, text="Calibrar (1 columna)", command=lambda: self.start_calibration("singlecol")).pack(fill=tk.X, pady=3)
        ttk.Button(btns, text="Calibrar (2 columnas)", command=lambda: self.start_calibration("multicol")).pack(fill=tk.X, pady=3)
        ttk.Button(btns, text="Cancelar calibración", command=self.cancel_calibration).pack(fill=tk.X, pady=3)

        # Tab 3: Answer Key
        tab_key = ttk.Frame(right)
        right.add(tab_key, text="Clave")

        ttk.Button(tab_key, text="Cargar clave CSV...", command=self.load_key).pack(anchor="w", padx=10, pady=5)
        self.key_status = tk.StringVar(value="Clave: (no cargada)")
        ttk.Label(tab_key, textvariable=self.key_status).pack(anchor="w", padx=10, pady=5)

        # Tab 4: Resultados
        tab_res = ttk.Frame(right)
        right.add(tab_res, text="Resultados")

        self.res_text = tk.Text(tab_res, width=50, height=30)
        self.res_text.pack(side=tk.LEFT, padx=5, pady=5)
        rbtns = ttk.Frame(tab_res); rbtns.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)
        ttk.Button(rbtns, text="Exportar CSV...", command=self.export_results).pack(fill=tk.X, pady=3)

    # --------------- Config handling ---------------
    def default_config(self):
        return {
            "page_size_mm": [215.9, 279.4],
            "dpi": 300,
            "aruco": {
                "dict": "DICT_4X4_50",
                "ids": {"TL":0, "TR":1, "BL":2, "BR":3},
                "marker_size_mm": 15.0,
                "offset_mm": 7.0
            },
            "grids": [
                {
                    "name": "main",
                    "questions": 50,
                    "options": ["A", "B", "C", "D", "E"],
                    # Placeholders; will be calibrated
                    "top_left_mm": [25.0, 40.0],
                    "bubble_size_mm": [5.0, 5.0],
                    "option_dx_mm": 8.0,
                    "row_dy_mm": 7.0,
                    "columns": 1,
                    "rows": 50,
                    "col_dx_mm": 60.0,
                    "block_row_gap_mm": 0.0
                }
            ],
            "thresholding": {"method":"otsu", "invert": True},
            "decision": {"min_fill": 0.25, "min_margin": 0.10}
        }

    def refresh_cfg_text(self):
        self.cfg_text.delete("1.0", tk.END)
        self.cfg_text.insert(tk.END, json.dumps(self.cfg, indent=2, ensure_ascii=False))

    def load_cfg(self):
        path = filedialog.askopenfilename(filetypes=[("JSON","*.json")])
        if not path: return
        with open(path, "r", encoding="utf-8") as f:
            self.cfg = json.load(f)
        self.refresh_cfg_text()
        messagebox.showinfo("Config", f"Configuración cargada: {os.path.basename(path)}")

    def save_cfg(self):
        path = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON","*.json")])
        if not path: return
        self.apply_cfg_text()
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.cfg, f, indent=2, ensure_ascii=False)
        messagebox.showinfo("Config", f"Guardado: {os.path.basename(path)}")

    def apply_cfg_text(self):
        try:
            cfg = json.loads(self.cfg_text.get("1.0", tk.END))
            self.cfg = cfg
            messagebox.showinfo("Config", "Configuración aplicada desde el panel.")
        except Exception as e:
            messagebox.showerror("Config", f"JSON inválido:\n{e}")

    # --------------- Camera ---------------
    def toggle_camera(self):
        if self.running:
            self.running = False
            if self.cap:
                self.cap.release()
                self.cap = None
            self.status_var.set("Estado: cámara parada.")
        else:
            self.camera_index = int(self.cam_var.get())
            self.cap = cv2.VideoCapture(self.camera_index)
            if not self.cap.isOpened():
                messagebox.showerror("Cámara", f"No se pudo abrir la cámara {self.camera_index}")
                return
            self.running = True
            self.status_var.set("Estado: cámara encendida.")
            threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        while self.running and self.cap and self.cap.isOpened():
            ok, frame = self.cap.read()
            if not ok:
                continue
            self.frame = frame
            preview = frame.copy()
            try:
                centers, corners, ids = detect_aruco_centers(frame, self.cfg["aruco"]["dict"])
                if ids is not None:
                    cv2.aruco.drawDetectedMarkers(preview, corners, ids)
                ids_map = self.cfg["aruco"]["ids"]
                page_w_mm, page_h_mm = self.cfg["page_size_mm"]
                dpi = int(self.cfg["dpi"])
                marker_mm = float(self.cfg["aruco"]["marker_size_mm"])
                offset_mm = float(self.cfg["aruco"]["offset_mm"])
                size, src_pts, dst_pts = compute_homography_from_four_centers(
                    centers, ids_map, page_w_mm, page_h_mm, dpi, marker_mm, offset_mm
                )
                H, _ = cv2.findHomography(src_pts, dst_pts, method=cv2.RANSAC, ransacReprojThreshold=3.0)
                Wpx, Hpx = size
                rectified = cv2.warpPerspective(frame, H, (Wpx, Hpx))
                self.rectified = rectified
                overlay = rectified.copy()

                # Draw current grid overlay
                for grid_spec in self.cfg["grids"]:
                    grid = GridLayout(grid_spec)
                    for (_q, _oi, _lab, (x_mm,y_mm,w_mm,h_mm)) in grid.iter_cells_mm():
                        x = int(round(mm_to_px(x_mm, dpi))); y = int(round(mm_to_px(y_mm, dpi)))
                        w = int(round(mm_to_px(w_mm, dpi))); h = int(round(mm_to_px(h_mm, dpi)))
                        cv2.rectangle(overlay, (x,y), (x+w,y+h), (255,0,0), 1)

                # If calibrating, draw clicked points
                if self.calibrating and self.calib_points:
                    for (x,y) in self.calib_points:
                        cv2.circle(overlay, (int(x),int(y)), 6, (0,255,255), -1)

                self.overlay = overlay
                disp = overlay
                self.status_var.set("Estado: marcadores OK; listo para calificar.")
            except Exception as e:
                disp = preview
                self.status_var.set(f"Estado: buscando 4 marcadores... ({e})")
            self.show_on_canvas(disp)

    def show_on_canvas(self, bgr):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        im = Image.fromarray(rgb)
        # Fit to canvas size
        cw = self.canvas.winfo_width() if self.canvas.winfo_width() > 1 else 800
        ch = self.canvas.winfo_height() if self.canvas.winfo_height() > 1 else 600
        im.thumbnail((cw, ch), Image.LANCZOS)
        self._tk_img = ImageTk.PhotoImage(im)
        self.canvas.create_image(cw//2, ch//2, image=self._tk_img, anchor=tk.CENTER)

    # --------------- Calibration ---------------
    def start_calibration(self, mode="singlecol"):
        if self.rectified is None:
            messagebox.showwarning("Calibración", "Aún no hay rectificación (¿se ven los 4 ArUco?).")
            return
        self.calibrating = True
        self.calib_mode = mode
        self.calib_points = []
        if mode == "singlecol":
            inst = (
                "Calibración (1 columna):\n"
                "1) Clic en ESQUINA SUPERIOR-IZQUIERDA del primer óvalo (Q1-A).\n"
                "2) Clic en ESQUINA INFERIOR-DERECHA del mismo óvalo (Q1-A).\n"
                "3) Clic en el CENTRO del siguiente óvalo a la derecha (Q1-B).\n"
                "4) Clic en el CENTRO del óvalo de la siguiente FILA (Q2-A).\n"
                "5) (Opcional) Si usas bloques con separación extra cada 10 preguntas, no hagas nada ahora; puedes ajustar luego 'block_row_gap_mm'.\n"
                "Al terminar los 4 clics, se actualizará el JSON."
            )
        else:
            inst = (
                "Calibración (2 columnas):\n"
                "1) Clic en ESQUINA SUPERIOR-IZQUIERDA del primer óvalo (Q1-A).\n"
                "2) Clic en ESQUINA INFERIOR-DERECHA del mismo óvalo (Q1-A).\n"
                "3) Clic en el CENTRO del siguiente óvalo a la derecha (Q1-B).\n"
                "4) Clic en el CENTRO del óvalo de la siguiente FILA (Q2-A).\n"
                "5) Clic en el CENTRO del primer óvalo de la SEGUNDA COLUMNA (p.ej., Q26-A).\n"
                "Al terminar los 5 clics, se actualizará el JSON."
            )
        messagebox.showinfo("Instrucciones", inst)

    def cancel_calibration(self):
        self.calibrating = False
        self.calib_points = []
        self.calib_mode = None
        self.status_var.set("Calibración cancelada.")

    def on_canvas_click(self, event):
        if not self.calibrating or self.rectified is None:
            return
        # Translate click from canvas coordinates back to image coordinates is non-trivial
        # Because we use thumbnail(), we can't map exactly. So we accept clicks only roughly.
        # To make it exact, we display rectified at 1:1 scale in canvas when calibrating.
        # Let's handle this by storing the displayed image size and offsets.
        # Simplify: open a separate toplevel window that shows rectified at 1:1 during calibration.
        self.open_calibration_window()

    def open_calibration_window(self):
        # One-time window per calibration session
        if hasattr(self, "_calib_win") and self._calib_win.winfo_exists():
            return  # Already open
        win = tk.Toplevel(self)
        win.title("Clics de calibración (1:1)")
        self._calib_win = win
        # Prepare Tk image from rectified (1:1 RGB)
        rgb = cv2.cvtColor(self.rectified, cv2.COLOR_BGR2RGB)
        self._calib_img_pil = Image.fromarray(rgb)
        self._calib_tk_img = ImageTk.PhotoImage(self._calib_img_pil)
        canvas = tk.Canvas(win, width=self._calib_img_pil.width, height=self._calib_img_pil.height, bg="black", scrollregion=(0,0,self._calib_img_pil.width,self._calib_img_pil.height))
        hbar = tk.Scrollbar(win, orient=tk.HORIZONTAL); hbar.pack(side=tk.BOTTOM, fill=tk.X)
        vbar = tk.Scrollbar(win, orient=tk.VERTICAL); vbar.pack(side=tk.RIGHT, fill=tk.Y)
        hbar.config(command=canvas.xview); vbar.config(command=canvas.yview)
        canvas.config(xscrollcommand=hbar.set, yscrollcommand=vbar.set)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        canvas.create_image(0,0, image=self._calib_tk_img, anchor=tk.NW)

        # Instructions label
        inst = tk.Label(win, text="Haz los clics según las instrucciones mostradas previamente.", justify="left")
        inst.pack(side=tk.TOP, anchor="w")

        # Bind click
        def on_click(ev):
            x, y = ev.x, ev.y
            self.calib_points.append((x,y))
            # Draw dot
            r = 5
            canvas.create_oval(x-r, y-r, x+r, y+r, outline="yellow", width=2)
            needed = 4 if self.calib_mode == "singlecol" else 5
            if len(self.calib_points) >= needed:
                win.destroy()
                self.finish_calibration()

        canvas.bind("<Button-1>", on_click)

    def finish_calibration(self):
        dpi = int(self.cfg["dpi"])
        pts = self.calib_points.copy()
        try:
            if self.calib_mode == "singlecol":
                if len(pts) < 4: raise ValueError("Faltan clics.")
                tl = pts[0]; br = pts[1]; right_center = pts[2]; nextrow_center = pts[3]
                # Compute params in mm
                w_mm = px_to_mm(abs(br[0]-tl[0]), dpi)
                h_mm = px_to_mm(abs(br[1]-tl[1]), dpi)
                # top-left mm equals tl pixel converted
                top_left_mm_x = px_to_mm(tl[0], dpi)
                top_left_mm_y = px_to_mm(tl[1], dpi)
                option_dx_mm = px_to_mm(abs(right_center[0] - (tl[0] + (br[0]-tl[0])/2.0)), dpi) * 2.0  # center-to-center
                row_dy_mm = px_to_mm(abs(nextrow_center[1] - (tl[1] + (br[1]-tl[1])/2.0)), dpi) * 2.0
                # Update cfg
                grid = self.cfg["grids"][0]
                grid["top_left_mm"] = [round(top_left_mm_x,2), round(top_left_mm_y,2)]
                grid["bubble_size_mm"] = [round(w_mm,2), round(h_mm,2)]
                grid["option_dx_mm"] = round(option_dx_mm,2)
                grid["row_dy_mm"] = round(row_dy_mm,2)
                grid["columns"] = 1
                self.status_var.set(f"Calibrado: top_left={grid['top_left_mm']} mm, bubble={grid['bubble_size_mm']} mm, option_dx={grid['option_dx_mm']} mm, row_dy={grid['row_dy_mm']} mm.")
            else:
                if len(pts) < 5: raise ValueError("Faltan clics.")
                tl = pts[0]; br = pts[1]; right_center = pts[2]; nextrow_center = pts[3]; nextcol_center = pts[4]
                w_mm = px_to_mm(abs(br[0]-tl[0]), dpi)
                h_mm = px_to_mm(abs(br[1]-tl[1]), dpi)
                top_left_mm_x = px_to_mm(tl[0], dpi)
                top_left_mm_y = px_to_mm(tl[1], dpi)
                option_dx_mm = px_to_mm(abs(right_center[0] - (tl[0] + (br[0]-tl[0])/2.0)), dpi) * 2.0
                row_dy_mm = px_to_mm(abs(nextrow_center[1] - (tl[1] + (br[1]-tl[1])/2.0)), dpi) * 2.0
                col_dx_mm = px_to_mm(abs(nextcol_center[0] - (tl[0] + (br[0]-tl[0])/2.0)), dpi)  # center-to-center to next column A
                grid = self.cfg["grids"][0]
                grid["top_left_mm"] = [round(top_left_mm_x,2), round(top_left_mm_y,2)]
                grid["bubble_size_mm"] = [round(w_mm,2), round(h_mm,2)]
                grid["option_dx_mm"] = round(option_dx_mm,2)
                grid["row_dy_mm"] = round(row_dy_mm,2)
                grid["columns"] = 2
                grid["col_dx_mm"] = round(col_dx_mm,2)
                self.status_var.set(f"Calibrado 2 col: top_left={grid['top_left_mm']} mm, bubble={grid['bubble_size_mm']} mm, option_dx={grid['option_dx_mm']} mm, row_dy={grid['row_dy_mm']} mm, col_dx={grid['col_dx_mm']} mm.")
        except Exception as e:
            messagebox.showerror("Calibración", f"No se pudo completar: {e}")
        finally:
            self.calibrating = False
            self.calib_points = []
            self.calib_mode = None
            self.refresh_cfg_text()

    # --------------- Grading ---------------
    def capture_and_grade(self):
        if self.rectified is None:
            messagebox.showwarning("OMR", "No hay rectificación disponible (¿se ven los 4 marcadores?).")
            return
        df, overlay = grade_rectified(self.rectified, self.cfg)
        self.last_results = df
        self.overlay = overlay
        self.show_on_canvas(self.overlay)
        self.show_results(df)

        if self.answer_key is not None:
            out = df.merge(self.answer_key, on="question", how="left")
            out["correct"] = out.apply(lambda r: (str(r.get("chosen_label","")).strip().upper()==str(r.get("answer","")).strip().upper()), axis=1)
            score = int(out["correct"].sum())
            total = int(out["question"].max()) if len(out)>0 else 0
            self.res_text.insert(tk.END, f"\n\nPuntaje: {score}/{total}\n")
        else:
            self.res_text.insert(tk.END, "\n\n(Clave no cargada; se muestran elecciones detectadas)\n")

    def show_results(self, df: pd.DataFrame):
        self.res_text.delete("1.0", tk.END)
        if df is None or df.empty:
            self.res_text.insert(tk.END, "Sin resultados.\n")
            return
        lines = []
        for _, r in df.iterrows():
            q = int(r["question"])
            ch = r["chosen_label"] if isinstance(r["chosen_label"], str) else ""
            st = r["status"]
            margin = float(r["margin"])
            best = float(r["best"])
            lines.append(f"Q{q:02d}: {ch or '-'}  (status={st}, fill_max={best:.2f}, margin={margin:.2f})")
        self.res_text.insert(tk.END, "\n".join(lines))

    # --------------- Key handling ---------------
    def load_key(self):
        path = filedialog.askopenfilename(filetypes=[("CSV","*.csv")])
        if not path: return
        try:
            key = pd.read_csv(path)
            if not {"question","answer"}.issubset(key.columns):
                raise ValueError("CSV debe tener columnas: question, answer")
            key["question"] = key["question"].astype(int)
            key["answer"] = key["answer"].astype(str).str.strip().str.upper()
            self.answer_key = key
            self.key_status.set(f"Clave cargada: {os.path.basename(path)} ({len(key)} preguntas)")
        except Exception as e:
            messagebox.showerror("Clave", f"No se pudo cargar la clave:\n{e}")

    # --------------- Export ---------------
    def export_results(self):
        if self.last_results is None or self.last_results.empty:
            messagebox.showwarning("Exportar", "No hay resultados para exportar.")
            return
        path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV","*.csv")])
        if not path: return
        df = self.last_results.copy()
        if self.answer_key is not None:
            df = df.merge(self.answer_key, on="question", how="left")
            df["correct"] = df.apply(lambda r: (str(r.get("chosen_label","")).strip().upper()==str(r.get("answer","")).strip().upper()), axis=1)
        df.insert(0, "timestamp", time.strftime("%Y-%m-%d %H:%M:%S"))
        df.to_csv(path, index=False, encoding="utf-8")
        messagebox.showinfo("Exportar", f"Resultados guardados en:\n{path}")

if __name__ == "__main__":
    app = OMRGUI()
    app.mainloop()
