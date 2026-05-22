"""
map_annotator.py — разметка зданий на карте с GUI-лончером.

Установка: pip install opencv-python numpy
           tkinter встроен в Python.

Запуск: python map_annotator.py
"""

import cv2
import numpy as np
import json
import os
import math
import tkinter as tk
from tkinter import filedialog, messagebox

# ── Конфиг ───────────────────────────────────────────────────────────────────

class Cfg:
    map_path    = ""
    output_path = ""
    world_x     = 180.0
    world_z     = 146.0
    center_wx   = 0.0
    center_wz   = 0.0
    buildings   = ["House_6", "House_29", "House_27"]
    snap_radius = 12

FONT      = cv2.FONT_HERSHEY_SIMPLEX
MODE_DRAW = 0
MODE_REF  = 1
MODE_PAN  = 2

C_DONE   = (0, 200, 0)
C_ACTIVE = (0, 180, 255)
C_DOT    = (0, 80, 255)
C_SNAP   = (0, 220, 220)
C_REF_A  = (60, 60, 255)
C_REF_B  = (255, 130, 30)
C_ARROW  = (200, 60, 220)
C_CLOSE  = (120, 80, 200)

PANEL_W = 170   # ширина боковой панели

# (label, bgr_color, command)
PANEL_BUTTONS = [
    ("Confirm  Enter",  (30, 160, 30),   "confirm"),
    ("Pan map    P",    (160, 160, 0),   "pan"),
    ("Ref arrow  R",    (160, 0, 160),   "ref"),
    ("Undo       Z",    (0, 120, 180),   "undo"),
    ("Save       S",    (30, 100, 200),  "save"),
    ("Exit       Q",    (30,  30, 160),  "quit"),
]
BTN_H    = 44
BTN_GAP  = 8
BTN_TOP  = 72   # y начала первой кнопки


# ── Координатные утилиты ──────────────────────────────────────────────────────

def building_name(idx):
    return Cfg.buildings[idx] if idx < len(Cfg.buildings) else f"Object_{idx + 1}"


def scale_mpp(img_shape):
    h, w = img_shape[:2]
    return Cfg.world_x / w, Cfg.world_z / h


def px_to_world(px, pz, img_w, img_h, world_x=None, world_z=None, cx=None, cz=None):
    wx  = world_x if world_x is not None else Cfg.world_x
    wz  = world_z if world_z is not None else Cfg.world_z
    ccx = cx      if cx      is not None else Cfg.center_wx
    ccz = cz      if cz      is not None else Cfg.center_wz
    return (px / img_w - 0.5) * wx + ccx, (pz / img_h - 0.5) * wz + ccz


def world_to_px(wx, wz, img_w, img_h):
    px = ((wx - Cfg.center_wx) / Cfg.world_x + 0.5) * img_w
    pz = ((wz - Cfg.center_wz) / Cfg.world_z + 0.5) * img_h
    return int(round(px)), int(round(pz))


def snap_points(done):
    return [(int(p[0]), int(p[1])) for b in done for p in b["polygon"]]


def snap(x, y, done):
    best_d = Cfg.snap_radius + 1
    best   = (x, y)
    for pt in snap_points(done):
        d = math.hypot(x - pt[0], y - pt[1])
        if d < best_d:
            best_d, best = d, pt
    return best, best_d <= Cfg.snap_radius


def text_outlined(img, text, pos, scale, color, thickness=1):
    cv2.putText(img, text, pos, FONT, scale, (255,255,255), thickness+2, cv2.LINE_AA)
    cv2.putText(img, text, pos, FONT, scale, color,         thickness,   cv2.LINE_AA)


# ── Боковая панель ────────────────────────────────────────────────────────────

def _btn_rect(i):
    """Возвращает (y0, y1) i-й кнопки в координатах панели."""
    y0 = BTN_TOP + i * (BTN_H + BTN_GAP)
    return y0, y0 + BTN_H


def draw_panel(panel, mode, current_idx, n_pts, done):
    panel[:] = (28, 28, 28)
    h = panel.shape[0]

    # Статус
    mode_str = ["DRAW", "REF ", "PAN "][mode]
    bname    = building_name(current_idx)
    cv2.putText(panel, f"Mode: {mode_str}", (10, 22), FONT, 0.55, (200,200,200), 1, cv2.LINE_AA)
    cv2.putText(panel, f"[{current_idx+1}] {bname}", (10, 44), FONT, 0.48, (180,220,180), 1, cv2.LINE_AA)
    cv2.putText(panel, f"{n_pts} pts", (10, 62), FONT, 0.45, (160,160,160), 1, cv2.LINE_AA)

    # Кнопки
    for i, (label, color, _cmd) in enumerate(PANEL_BUTTONS):
        y0, y1 = _btn_rect(i)
        if y1 > h - 4:
            break
        # Подсветка активного режима
        is_active = (mode == MODE_PAN and _cmd == "pan") or \
                    (mode == MODE_REF and _cmd == "ref")
        bg = tuple(min(255, c + 60) for c in color) if is_active else color
        cv2.rectangle(panel, (4, y0), (PANEL_W - 4, y1), bg, -1)
        cv2.rectangle(panel, (4, y0), (PANEL_W - 4, y1), (80,80,80), 1)
        cv2.putText(panel, label, (10, y0 + BTN_H // 2 + 6),
                    FONT, 0.48, (255,255,255), 1, cv2.LINE_AA)

    # Список зданий
    list_y = BTN_TOP + len(PANEL_BUTTONS) * (BTN_H + BTN_GAP) + 16
    cv2.putText(panel, "Done:", (10, list_y), FONT, 0.46, (140,140,140), 1, cv2.LINE_AA)
    for i, b in enumerate(done):
        ly = list_y + 18 + i * 18
        if ly > h - 6:
            break
        cv2.putText(panel, b["name"], (10, ly), FONT, 0.42, (80,200,80), 1, cv2.LINE_AA)


def panel_click(y):
    """Возвращает command-строку по y-клику в панели, или None."""
    for i, (_, _, cmd) in enumerate(PANEL_BUTTONS):
        y0, y1 = _btn_rect(i)
        if y0 <= y <= y1:
            return cmd
    return None


# ── Отрисовка ─────────────────────────────────────────────────────────────────

def draw_state(base_img, done, current_points, current_idx,
               mode, ref_pts, mx, my, img_offset=(0, 0)):
    odx, odz = img_offset
    if odx != 0 or odz != 0:
        M = np.float32([[1, 0, odx], [0, 1, odz]])
        map_img = cv2.warpAffine(base_img, M, (base_img.shape[1], base_img.shape[0]))
    else:
        map_img = base_img.copy()

    sx, sz = scale_mpp(base_img.shape)

    # Завершённые здания
    for b in done:
        poly = np.array(b["polygon"], dtype=np.int32)
        ol = map_img.copy(); cv2.fillPoly(ol, [poly], (0, 150, 0))
        cv2.addWeighted(ol, 0.18, map_img, 0.82, 0, map_img)
        cv2.polylines(map_img, [poly], True, C_DONE, 2)
        cx_ = int(b["center_px"][0]); cy_ = int(b["center_px"][1])
        text_outlined(map_img, b["name"], (cx_ - 30, cy_ - 5), 0.65, C_DONE)

    # Snap
    if mx is not None and mode == MODE_DRAW and mx < base_img.shape[1]:
        pt_s, is_s = snap(mx, my, done)
        if is_s:
            cv2.circle(map_img, pt_s, Cfg.snap_radius, C_SNAP, 2)

    # Текущий контур
    if current_points:
        pts_arr = np.array(current_points, dtype=np.int32)
        if len(current_points) >= 3:
            ol = map_img.copy(); cv2.fillPoly(ol, [pts_arr], (0, 90, 255))
            cv2.addWeighted(ol, 0.14, map_img, 0.86, 0, map_img)
        for i in range(len(current_points) - 1):
            cv2.line(map_img, current_points[i], current_points[i+1], C_ACTIVE, 1)
        if mx is not None and mode == MODE_DRAW:
            cv2.line(map_img, current_points[-1], (mx, my), C_ACTIVE, 1)
            if len(current_points) >= 2:
                cv2.line(map_img, (mx, my), current_points[0], C_CLOSE, 1)
        for pt in current_points:
            cv2.circle(map_img, pt, 4, C_DOT, -1)

    # Стрелка-референс
    if mode == MODE_REF:
        p0 = ref_pts[0] if ref_pts else None
        p1 = ref_pts[1] if len(ref_pts) >= 2 else ((mx, my) if mx is not None else None)
        if p0:
            cv2.circle(map_img, p0, 7, C_REF_A, -1); cv2.circle(map_img, p0, 7, (255,255,255), 1)
        if p0 and p1:
            cv2.arrowedLine(map_img, p0, p1, C_ARROW, 2, tipLength=0.04)
            ddx = p1[0]-p0[0]; ddz = p1[1]-p0[1]
            dist_m = math.hypot(ddx*sx, ddz*sz)
            ang    = math.degrees(math.atan2(ddz, ddx))
            mid    = ((p0[0]+p1[0])//2+6, (p0[1]+p1[1])//2-6)
            text_outlined(map_img, f"{dist_m:.1f}m  {ang:.1f}deg", mid, 0.55, C_ARROW)
        if len(ref_pts) >= 2:
            cv2.circle(map_img, ref_pts[1], 7, C_REF_B, -1); cv2.circle(map_img, ref_pts[1], 7, (255,255,255), 1)

    # Статус-бар карты
    cv2.rectangle(map_img, (0, 0), (map_img.shape[1], 30), (20, 20, 20), -1)
    if mode == MODE_PAN:
        s = f"PAN: drag to align  dx={odx:+d} dz={odz:+d}  |  Enter=commit  P=cancel"
        cv2.putText(map_img, s, (8, 21), FONT, 0.52, (0, 220, 255), 1, cv2.LINE_AA)
    elif mode == MODE_REF:
        if not ref_pts:       s = "REF: click anchor on existing building"
        elif len(ref_pts)==1: s = "REF: click start of new building (1st corner)"
        else:                 s = "REF locked.  Z = reset"
        cv2.putText(map_img, s, (8, 21), FONT, 0.56, (100, 200, 255), 1, cv2.LINE_AA)
    else:
        n = len(current_points); bname = building_name(current_idx)
        need = "Enter=confirm" if n >= 3 else f"need {3-n} more pts"
        cv2.putText(map_img, f"[{current_idx+1}] {bname}  {n} pts  |  {need}",
                    (8, 21), FONT, 0.56, (255, 255, 255), 1, cv2.LINE_AA)

    # Боковая панель
    panel = np.zeros((map_img.shape[0], PANEL_W, 3), dtype=np.uint8)
    draw_panel(panel, mode, current_idx, len(current_points), done)

    return np.hstack([map_img, panel])


# ── Вычисление ────────────────────────────────────────────────────────────────

def compute_building(name, points):
    pts_arr = np.array(points, dtype=np.float32)
    rect    = cv2.minAreaRect(pts_arr)
    (cx, cy), (w, h), angle = rect
    box = cv2.boxPoints(rect).astype(np.int32)
    return {
        "name":      name,
        "center_px": (cx, cy),
        "box_pts":   box,
        "polygon":   [list(p) for p in points],
        "width":     float(w),
        "length":    float(h),
        "angle":     float(angle),
    }


def _add_world_coords(entry, img_w, img_h):
    entry["polygon_world"] = [
        list(px_to_world(p[0], p[1], img_w, img_h))
        for p in entry["polygon"]
    ]


def _reproject_building(b, img_w, img_h):
    if "polygon_world" not in b:
        return
    pts = [world_to_px(p[0], p[1], img_w, img_h) for p in b["polygon_world"]]
    updated = compute_building(b["name"], pts)
    updated["polygon_world"] = b["polygon_world"]
    b.update(updated)


# ── JSON ──────────────────────────────────────────────────────────────────────

def build_json(img_shape, done):
    h, w = img_shape[:2]
    out = []
    for b in done:
        poly_world = b.get("polygon_world") or [
            list(px_to_world(p[0], p[1], w, h)) for p in b["polygon"]
        ]
        cx_w, cz_w = px_to_world(b["center_px"][0], b["center_px"][1], w, h)
        out.append({
            "name":             b["name"],
            "center":           {"x": round(b["center_px"][0], 2),
                                 "z": round(b["center_px"][1], 2)},
            "center_world":     {"x": round(cx_w, 4), "z": round(cz_w, 4)},
            "size":             {"width":  round(b["width"],  2),
                                 "length": round(b["length"], 2)},
            "rotation_degrees": round(b["angle"], 4),
            "polygon_px":       b["polygon"],
            "polygon_world":    [[round(p[0], 4), round(p[1], 4)] for p in poly_world],
        })
    return {
        "map_size":   {"width": w, "height": h},
        "world_meta": {"world_x":   Cfg.world_x, "world_z":   Cfg.world_z,
                       "center_wx": Cfg.center_wx, "center_wz": Cfg.center_wz},
        "buildings":  out,
    }


def load_existing(img_shape):
    if not os.path.exists(Cfg.output_path):
        return []
    try:
        with open(Cfg.output_path, encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"Cannot load {Cfg.output_path}: {e}")
        return []

    img_h, img_w = img_shape[:2]
    meta = data.get("world_meta")
    result = []

    for b in data.get("buildings", []):
        name = b["name"]
        poly_world = None

        if b.get("polygon_world"):
            poly_world = b["polygon_world"]
            points = [world_to_px(p[0], p[1], img_w, img_h) for p in poly_world]
        elif b.get("polygon_px") and meta:
            old_w = data["map_size"]["width"]; old_h = data["map_size"]["height"]
            poly_world = [list(px_to_world(p[0], p[1], old_w, old_h,
                               meta["world_x"], meta["world_z"],
                               meta["center_wx"], meta["center_wz"]))
                          for p in b["polygon_px"]]
            points = [world_to_px(p[0], p[1], img_w, img_h) for p in poly_world]
        elif b.get("polygon_px"):
            points = [tuple(p) for p in b["polygon_px"]]
        else:
            cx, cy = b["center"]["x"], b["center"]["z"]
            w2, h2, ang = b["size"]["width"], b["size"]["length"], b["rotation_degrees"]
            box = cv2.boxPoints(((cx, cy), (w2, h2), ang)).astype(np.int32)
            points = [tuple(p) for p in box]

        entry = compute_building(name, points)
        if poly_world:
            entry["polygon_world"] = poly_world
        result.append(entry)
        print(f"Loaded: {name}")
    return result


# ── Основной цикл аннотатора ──────────────────────────────────────────────────

def run_annotator():
    raw = np.fromfile(Cfg.map_path, dtype=np.uint8)
    base_img = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    if base_img is None:
        messagebox.showerror("Error", f"Cannot open:\n{Cfg.map_path}")
        return

    img_h, img_w = base_img.shape[:2]

    done           = load_existing(base_img.shape)
    current_points = []
    current_idx    = len(done)
    mode           = MODE_DRAW
    ref_pts        = []
    mouse          = [None, None]
    img_offset     = [0, 0]
    pan_dragging   = [False]
    pan_last       = [0, 0]
    panel_cmd      = [None]   # команда от клика по кнопке панели

    window = "map_annotator"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, min(img_w + PANEL_W, 1560), min(img_h, 900))

    def on_mouse(event, x, y, _flags, _param):
        nonlocal current_points, current_idx, mode, ref_pts
        mouse[0], mouse[1] = x, y

        # Клик по боковой панели
        if event == cv2.EVENT_LBUTTONDOWN and x >= img_w:
            cmd = panel_click(y)
            if cmd:
                panel_cmd[0] = cmd
            return

        if mode == MODE_PAN:
            if event == cv2.EVENT_LBUTTONDOWN:
                pan_dragging[0] = True; pan_last[0], pan_last[1] = x, y
            elif event == cv2.EVENT_MOUSEMOVE and pan_dragging[0]:
                img_offset[0] += x - pan_last[0]; img_offset[1] += y - pan_last[1]
                pan_last[0], pan_last[1] = x, y
            elif event == cv2.EVENT_LBUTTONUP:
                pan_dragging[0] = False
            return

        if event != cv2.EVENT_LBUTTONDOWN:
            return
        pt_s, is_s = snap(x, y, done)
        pt = pt_s if is_s else (x, y)
        if mode == MODE_REF:
            ref_pts.append(pt)
            if len(ref_pts) == 2:
                current_points = [ref_pts[1]]; mode = MODE_DRAW; ref_pts = []
            return
        current_points.append(pt)

    cv2.setMouseCallback(window, on_mouse)

    def commit_pan():
        nonlocal mode
        Cfg.center_wx -= img_offset[0] * (Cfg.world_x / img_w)
        Cfg.center_wz -= img_offset[1] * (Cfg.world_z / img_h)
        img_offset[0] = img_offset[1] = 0
        for b in done:
            _reproject_building(b, img_w, img_h)
        mode = MODE_DRAW

    def handle_action(action):
        nonlocal current_points, current_idx, mode, ref_pts
        if action == "confirm":
            if mode == MODE_PAN:
                commit_pan()
            elif mode == MODE_DRAW and len(current_points) >= 3:
                b = compute_building(building_name(current_idx), current_points)
                _add_world_coords(b, img_w, img_h)
                done.append(b); current_points = []; current_idx += 1
        elif action == "pan":
            if mode == MODE_DRAW:
                mode = MODE_PAN; img_offset[0] = img_offset[1] = 0
            elif mode == MODE_PAN:
                img_offset[0] = img_offset[1] = 0; mode = MODE_DRAW
        elif action == "ref":
            if mode != MODE_PAN:
                mode = MODE_REF if mode == MODE_DRAW else MODE_DRAW; ref_pts = []
        elif action == "undo":
            if mode == MODE_REF:
                if ref_pts: ref_pts.pop()
                else: mode = MODE_DRAW
            elif mode == MODE_DRAW:
                if current_points: current_points.pop()
                elif done:
                    last = done.pop(); current_idx -= 1
                    current_points = [list(p) for p in last["polygon"]]
                    print(f"Undone: {last['name']}")
        elif action == "save":
            if not done:
                print("Nothing to save.")
            else:
                data = build_json(base_img.shape, done)
                with open(Cfg.output_path, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
                print(f"Saved: {Cfg.output_path}")
        elif action == "quit":
            return True   # сигнал выхода
        return False

    while True:
        frame = draw_state(base_img, done, current_points, current_idx,
                           mode, ref_pts, mouse[0], mouse[1], tuple(img_offset))
        try:
            cv2.imshow(window, frame)
            key = cv2.waitKey(20) & 0xFF
            if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                break
        except cv2.error:
            break

        # Команда от кнопки панели
        if panel_cmd[0]:
            if handle_action(panel_cmd[0]):
                break
            panel_cmd[0] = None

        # Горячие клавиши
        if key in (ord('q'), 27):
            break
        elif key in (13, ord(' ')):
            handle_action("confirm")
        elif key in (ord('p'), ord('P')):
            handle_action("pan")
        elif key in (ord('r'), ord('R')):
            handle_action("ref")
        elif key in (ord('z'), ord('Z')):
            handle_action("undo")
        elif key in (ord('s'), ord('S')):
            handle_action("save")

    cv2.destroyAllWindows()
    cv2.waitKey(1)


# ── Tkinter-лончер ────────────────────────────────────────────────────────────

class Launcher:
    def __init__(self, root: tk.Tk):
        self.root = root
        root.title("Map Annotator")
        root.resizable(False, False)

        P = {"padx": 8, "pady": 4}

        tk.Label(root, text="Map:").grid(row=0, column=0, sticky="w", **P)
        self.img_var = tk.StringVar(value=self._default_map())
        tk.Entry(root, textvariable=self.img_var, width=46).grid(row=0, column=1, **P)
        tk.Button(root, text="...", command=self._pick_map).grid(row=0, column=2, **P)

        tk.Label(root, text="JSON:").grid(row=1, column=0, sticky="w", **P)
        self.out_var = tk.StringVar(value=self._default_out())
        tk.Entry(root, textvariable=self.out_var, width=46).grid(row=1, column=1, **P)
        tk.Button(root, text="...", command=self._pick_out).grid(row=1, column=2, **P)

        sf = tk.LabelFrame(root, text="Screenshot covers (meters) + world center")
        sf.grid(row=2, column=0, columnspan=3, sticky="ew", **P)
        for col, (lbl, var_attr, default) in enumerate([
            ("Width X:",  "wx_var", Cfg.world_x),
            ("Depth Z:",  "wz_var", Cfg.world_z),
        ]):
            tk.Label(sf, text=lbl).grid(row=0, column=col*2, **P)
            v = tk.StringVar(value=str(default)); setattr(self, var_attr, v)
            tk.Entry(sf, textvariable=v, width=8).grid(row=0, column=col*2+1, **P)
        for col, (lbl, var_attr, default) in enumerate([
            ("Center X:", "cx_var", Cfg.center_wx),
            ("Center Z:", "cz_var", Cfg.center_wz),
        ]):
            tk.Label(sf, text=lbl).grid(row=1, column=col*2, **P)
            v = tk.StringVar(value=str(default)); setattr(self, var_attr, v)
            tk.Entry(sf, textvariable=v, width=8).grid(row=1, column=col*2+1, **P)

        bf = tk.LabelFrame(root, text="Buildings (annotation order)")
        bf.grid(row=3, column=0, columnspan=3, sticky="ew", **P)
        self.listbox = tk.Listbox(bf, height=7, width=28, selectmode=tk.SINGLE)
        self.listbox.grid(row=0, column=0, rowspan=4, padx=8, pady=4)
        scroll = tk.Scrollbar(bf, command=self.listbox.yview)
        scroll.grid(row=0, column=1, rowspan=4, sticky="ns", pady=4)
        self.listbox.config(yscrollcommand=scroll.set)
        for b in Cfg.buildings:
            self.listbox.insert(tk.END, b)
        self.name_var = tk.StringVar()
        tk.Entry(bf, textvariable=self.name_var, width=18).grid(row=0, column=2, **P)
        tk.Button(bf, text="+ Add",    width=12, command=self._add).grid(row=1, column=2, **P)
        tk.Button(bf, text="- Remove", width=12, command=self._remove).grid(row=2, column=2, **P)
        tk.Button(bf, text="^ Up",     width=12, command=self._move_up).grid(row=3, column=2, **P)

        tk.Button(root, text="Start annotation",
                  bg="#2a6e2a", fg="white", font=("", 11, "bold"),
                  padx=16, pady=8, relief="flat",
                  command=self._launch).grid(row=4, column=0, columnspan=3, pady=14)

    def _default_map(self):
        p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "map.png")
        return p if os.path.exists(p) else ""

    def _default_out(self):
        return os.path.join(os.path.dirname(os.path.abspath(__file__)), "buildings_data.json")

    def _pick_map(self):
        path = filedialog.askopenfilename(
            title="Select map image",
            filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp"), ("All", "*.*")])
        if path:
            self.img_var.set(path)
            if not self.out_var.get():
                self.out_var.set(os.path.join(os.path.dirname(path), "buildings_data.json"))

    def _pick_out(self):
        path = filedialog.asksaveasfilename(
            title="Save JSON as", defaultextension=".json",
            filetypes=[("JSON", "*.json")])
        if path:
            self.out_var.set(path)

    def _add(self):
        name = self.name_var.get().strip()
        if name:
            self.listbox.insert(tk.END, name); self.name_var.set("")

    def _remove(self):
        sel = self.listbox.curselection()
        if sel: self.listbox.delete(sel[0])

    def _move_up(self):
        sel = self.listbox.curselection()
        if sel and sel[0] > 0:
            i = sel[0]; val = self.listbox.get(i)
            self.listbox.delete(i); self.listbox.insert(i-1, val)
            self.listbox.select_set(i-1)

    def _launch(self):
        map_path = self.img_var.get().strip()
        if not map_path or not os.path.exists(map_path):
            messagebox.showerror("Error", "Select a valid map file.")
            return
        try:
            wx = float(self.wx_var.get()); wz = float(self.wz_var.get())
            cx = float(self.cx_var.get()); cz = float(self.cz_var.get())
        except ValueError:
            messagebox.showerror("Error", "All numeric fields must be numbers.")
            return
        Cfg.map_path    = map_path
        Cfg.output_path = self.out_var.get().strip() or self._default_out()
        Cfg.world_x = wx; Cfg.world_z = wz
        Cfg.center_wx = cx; Cfg.center_wz = cz
        Cfg.buildings = list(self.listbox.get(0, tk.END))
        self.root.withdraw()
        run_annotator()
        self.root.destroy()


# ── Точка входа ───────────────────────────────────────────────────────────────

def main():
    root = tk.Tk()
    root.protocol("WM_DELETE_WINDOW", root.destroy)
    Launcher(root)
    root.mainloop()


if __name__ == "__main__":
    main()
