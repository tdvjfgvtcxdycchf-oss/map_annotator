"""
map_annotator.py — разметка зданий на карте с GUI-лончером.

Установка: pip install opencv-python numpy
           tkinter встроен в Python.

Запуск: python map_annotator.py
Горячие клавиши: Enter=confirm, A=align, R=ref, Z=undo, S=save, Q/Esc=quit

Система координат
-----------------
  World Space (пиксели) = Screen Space - img_offset
  Для первого скриншота offset=(0,0), поэтому world == screen pixel.
  При загрузке нового скриншота: режим Align двигает фон, здания — трафарет.
  После подтверждения Align img_offset задаёт соответствие screen↔world.
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
    world_x     = 180.0   # ширина карты в метрах (Godot X) — только для REF-дистанции
    world_z     = 146.0   # глубина карты в метрах (Godot Z)
    buildings   = ["House_6", "House_29", "House_27"]
    snap_radius = 12

FONT       = cv2.FONT_HERSHEY_SIMPLEX
MODE_DRAW  = 0
MODE_REF   = 1
MODE_ALIGN = 2

C_DONE    = (0, 200, 0)
C_ACTIVE  = (0, 180, 255)
C_DOT     = (0, 80, 255)
C_SNAP    = (0, 220, 220)
C_REF_A   = (60, 60, 255)
C_REF_B   = (255, 130, 30)
C_ARROW   = (200, 60, 220)
C_CLOSE   = (120, 80, 200)
C_STENCIL = (0, 200, 255)   # контуры-трафареты в режиме Align

PANEL_W = 170

PANEL_BUTTONS = [
    ("Confirm  Enter",  (30, 160, 30),   "confirm"),
    ("Align      A",    (160, 140, 0),   "align"),
    ("Ref arrow  R",    (160, 0, 160),   "ref"),
    ("Undo       Z",    (0, 120, 180),   "undo"),
    ("Save       S",    (30, 100, 200),  "save"),
    ("Exit       Q",    (30,  30, 160),  "quit"),
]
BTN_H   = 44
BTN_GAP = 8
BTN_TOP = 72


# ── Координатные утилиты ──────────────────────────────────────────────────────

def building_name(idx):
    return Cfg.buildings[idx] if idx < len(Cfg.buildings) else f"Object_{idx + 1}"


def scale_mpp(img_shape):
    """Метров на пиксель (для отображения дистанции в REF-стрелке)."""
    h, w = img_shape[:2]
    return Cfg.world_x / w, Cfg.world_z / h


def screen_to_world(sx, sy, ox, oy):
    """Экранные координаты → мировые пиксельные."""
    return sx - ox, sy - oy


def world_to_screen(wx, wy, ox, oy):
    """Мировые пиксельные → экранные."""
    return int(round(wx + ox)), int(round(wy + oy))


def snap(world_mx, world_my, done):
    """Притяжение к вершинам существующих полигонов (в мировых координатах)."""
    best_d = Cfg.snap_radius + 1
    best   = (world_mx, world_my)
    for b in done:
        for p in b["polygon"]:
            d = math.hypot(world_mx - p[0], world_my - p[1])
            if d < best_d:
                best_d, best = d, (p[0], p[1])
    return best, best_d <= Cfg.snap_radius


def text_outlined(img, text, pos, scale, color, thickness=1):
    cv2.putText(img, text, pos, FONT, scale, (255, 255, 255), thickness + 2, cv2.LINE_AA)
    cv2.putText(img, text, pos, FONT, scale, color,           thickness,     cv2.LINE_AA)


# ── Боковая панель ────────────────────────────────────────────────────────────

def _btn_rect(i):
    y0 = BTN_TOP + i * (BTN_H + BTN_GAP)
    return y0, y0 + BTN_H


def draw_panel(panel, mode, current_idx, n_pts, done):
    panel[:] = (28, 28, 28)
    h = panel.shape[0]

    mode_str = ["DRAW", "REF ", "ALIGN"][mode]
    bname    = building_name(current_idx)
    cv2.putText(panel, f"Mode: {mode_str}",   (10, 22), FONT, 0.55, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(panel, f"[{current_idx+1}] {bname}", (10, 44), FONT, 0.48, (180, 220, 180), 1, cv2.LINE_AA)
    cv2.putText(panel, f"{n_pts} pts",         (10, 62), FONT, 0.45, (160, 160, 160), 1, cv2.LINE_AA)

    for i, (label, color, cmd) in enumerate(PANEL_BUTTONS):
        y0, y1 = _btn_rect(i)
        if y1 > h - 4:
            break
        is_active = (mode == MODE_ALIGN and cmd == "align") or \
                    (mode == MODE_REF   and cmd == "ref")
        bg = tuple(min(255, c + 60) for c in color) if is_active else color
        cv2.rectangle(panel, (4, y0), (PANEL_W - 4, y1), bg, -1)
        cv2.rectangle(panel, (4, y0), (PANEL_W - 4, y1), (80, 80, 80), 1)
        cv2.putText(panel, label, (10, y0 + BTN_H // 2 + 6), FONT, 0.48, (255, 255, 255), 1, cv2.LINE_AA)

    list_y = BTN_TOP + len(PANEL_BUTTONS) * (BTN_H + BTN_GAP) + 16
    cv2.putText(panel, "Done:", (10, list_y), FONT, 0.46, (140, 140, 140), 1, cv2.LINE_AA)
    for i, b in enumerate(done):
        ly = list_y + 18 + i * 18
        if ly > h - 6:
            break
        cv2.putText(panel, b["name"], (10, ly), FONT, 0.42, (80, 200, 80), 1, cv2.LINE_AA)


def panel_click(y):
    for i, (_, _, cmd) in enumerate(PANEL_BUTTONS):
        y0, y1 = _btn_rect(i)
        if y0 <= y <= y1:
            return cmd
    return None


# ── Отрисовка ─────────────────────────────────────────────────────────────────

def draw_state(base_img, done, current_points, current_idx,
               mode, ref_pts, mx, my, img_offset=(0, 0)):
    ox, oy = img_offset

    # Фоновое изображение смещается на img_offset
    if ox != 0 or oy != 0:
        M = np.float32([[1, 0, ox], [0, 1, oy]])
        canvas = cv2.warpAffine(base_img, M, (base_img.shape[1], base_img.shape[0]))
    else:
        canvas = base_img.copy()

    sx_mpp, sz_mpp = scale_mpp(base_img.shape)

    # ── Существующие здания ──────────────────────────────────────────────────
    for b in done:
        if mode == MODE_ALIGN:
            # Режим выравнивания: здания — неподвижный трафарет (world coords = screen coords)
            pts = [(int(p[0]), int(p[1])) for p in b["polygon"]]
            lbl_x, lbl_y = int(b["center_px"][0]), int(b["center_px"][1])
            line_color = C_STENCIL
            fill_color = (0, 80, 160)
        else:
            # Обычный режим: world + offset → screen
            pts = [world_to_screen(p[0], p[1], ox, oy) for p in b["polygon"]]
            lbl_x, lbl_y = world_to_screen(b["center_px"][0], b["center_px"][1], ox, oy)
            line_color = C_DONE
            fill_color = (0, 150, 0)

        poly = np.array(pts, dtype=np.int32)
        ol = canvas.copy()
        cv2.fillPoly(ol, [poly], fill_color)
        cv2.addWeighted(ol, 0.18, canvas, 0.82, 0, canvas)
        cv2.polylines(canvas, [poly], True, line_color, 2)
        text_outlined(canvas, b["name"], (lbl_x - 30, lbl_y - 5), 0.65, line_color)

    # ── Snap-круг (только в режиме DRAW) ─────────────────────────────────────
    if mx is not None and mode == MODE_DRAW and mx < base_img.shape[1]:
        wmx, wmy = screen_to_world(mx, my, ox, oy)
        pt_w, is_s = snap(wmx, wmy, done)
        if is_s:
            cv2.circle(canvas, world_to_screen(pt_w[0], pt_w[1], ox, oy), Cfg.snap_radius, C_SNAP, 2)

    # ── Текущий контур (current_points в мировых координатах) ─────────────────
    if current_points:
        scr = [world_to_screen(p[0], p[1], ox, oy) for p in current_points]
        arr = np.array(scr, dtype=np.int32)
        if len(scr) >= 3:
            ol = canvas.copy()
            cv2.fillPoly(ol, [arr], (0, 90, 255))
            cv2.addWeighted(ol, 0.14, canvas, 0.86, 0, canvas)
        for i in range(len(scr) - 1):
            cv2.line(canvas, scr[i], scr[i + 1], C_ACTIVE, 1)
        if mx is not None and mode == MODE_DRAW:
            cv2.line(canvas, scr[-1], (mx, my), C_ACTIVE, 1)
            if len(scr) >= 2:
                cv2.line(canvas, (mx, my), scr[0], C_CLOSE, 1)
        for pt in scr:
            cv2.circle(canvas, pt, 4, C_DOT, -1)

    # ── Стрелка-референс (ref_pts в мировых координатах) ─────────────────────
    if mode == MODE_REF:
        p0_w = ref_pts[0] if ref_pts else None
        p1_w = ref_pts[1] if len(ref_pts) >= 2 else None
        p0 = world_to_screen(p0_w[0], p0_w[1], ox, oy) if p0_w else None
        if p1_w:
            p1 = world_to_screen(p1_w[0], p1_w[1], ox, oy)
        elif p0_w and mx is not None:
            p1 = (mx, my)
        else:
            p1 = None
        if p0:
            cv2.circle(canvas, p0, 7, C_REF_A, -1)
            cv2.circle(canvas, p0, 7, (255, 255, 255), 1)
        if p0 and p1:
            cv2.arrowedLine(canvas, p0, p1, C_ARROW, 2, tipLength=0.04)
            ddx, ddz = p1[0] - p0[0], p1[1] - p0[1]
            dist_m = math.hypot(ddx * sx_mpp, ddz * sz_mpp)
            ang    = math.degrees(math.atan2(ddz, ddx))
            mid    = ((p0[0] + p1[0]) // 2 + 6, (p0[1] + p1[1]) // 2 - 6)
            text_outlined(canvas, f"{dist_m:.1f}m  {ang:.1f}deg", mid, 0.55, C_ARROW)
        if p1_w:
            cv2.circle(canvas, p1, 7, C_REF_B, -1)
            cv2.circle(canvas, p1, 7, (255, 255, 255), 1)

    # ── Статус-бар ────────────────────────────────────────────────────────────
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 30), (20, 20, 20), -1)
    if mode == MODE_ALIGN:
        s = f"ALIGN: drag map to match stencil  dx={ox:+d} dy={oy:+d}  |  Enter=confirm  A=cancel"
        cv2.putText(canvas, s, (8, 21), FONT, 0.50, (0, 220, 255), 1, cv2.LINE_AA)
    elif mode == MODE_REF:
        if not ref_pts:        s = "REF: click anchor on existing building"
        elif len(ref_pts) == 1: s = "REF: click start of new building"
        else:                  s = "REF locked.  Z=reset"
        cv2.putText(canvas, s, (8, 21), FONT, 0.56, (100, 200, 255), 1, cv2.LINE_AA)
    else:
        n = len(current_points)
        need = "Enter=confirm" if n >= 3 else f"need {3 - n} more pts"
        cv2.putText(canvas, f"[{current_idx+1}] {building_name(current_idx)}  {n} pts  |  {need}",
                    (8, 21), FONT, 0.56, (255, 255, 255), 1, cv2.LINE_AA)

    panel = np.zeros((canvas.shape[0], PANEL_W, 3), dtype=np.uint8)
    draw_panel(panel, mode, current_idx, len(current_points), done)
    return np.hstack([canvas, panel])


# ── Вычисление здания ─────────────────────────────────────────────────────────

def compute_building(name, world_points):
    """Принимает точки в мировых пиксельных координатах."""
    pts_arr = np.array(world_points, dtype=np.float32)
    rect    = cv2.minAreaRect(pts_arr)
    (cx, cy), (w, h), angle = rect
    return {
        "name":      name,
        "center_px": (cx, cy),                          # мировые пиксельные
        "polygon":   [list(p) for p in world_points],   # мировые пиксельные
        "width":     float(w),
        "length":    float(h),
        "angle":     float(angle),
    }


# ── JSON ──────────────────────────────────────────────────────────────────────

def build_json(img_shape, done):
    h, w = img_shape[:2]
    buildings = []
    for b in done:
        buildings.append({
            "name":             b["name"],
            # center.x/z в мировых пикселях; json_to_scene.py использует эти значения
            # как пиксельные координаты на исходном изображении (совпадают при offset=0)
            "center":           {"x": round(b["center_px"][0], 2),
                                 "z": round(b["center_px"][1], 2)},
            "size":             {"width":  round(b["width"],  2),
                                 "length": round(b["length"], 2)},
            "rotation_degrees": round(b["angle"], 4),
            "polygon_world":    [[round(p[0], 2), round(p[1], 2)] for p in b["polygon"]],
        })
    return {
        "format":    "pixel_world_v2",
        "map_size":  {"width": w, "height": h},
        "world_meta": {"world_x": Cfg.world_x, "world_z": Cfg.world_z},
        "buildings": buildings,
    }


def load_existing(img_shape):
    """Загружает здания из JSON, конвертируя старые форматы в мировые пиксельные координаты."""
    if not os.path.exists(Cfg.output_path):
        return []
    try:
        with open(Cfg.output_path, encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"Cannot load {Cfg.output_path}: {e}")
        return []

    fmt    = data.get("format", "")
    img_h, img_w = img_shape[:2]
    result = []

    for b in data.get("buildings", []):
        name = b["name"]

        if fmt == "pixel_world_v2" and b.get("polygon_world"):
            # Новый формат: polygon_world уже в мировых пиксельных координатах
            points = [tuple(p) for p in b["polygon_world"]]

        elif b.get("polygon_world"):
            # Старый формат: polygon_world содержал метровые координаты Godot
            # Обратная конвертация: meter → pixel
            meta  = data.get("world_meta", {})
            wx_m  = meta.get("world_x",   Cfg.world_x)
            wz_m  = meta.get("world_z",   Cfg.world_z)
            cx_m  = meta.get("center_wx", 0.0)
            cz_m  = meta.get("center_wz", 0.0)
            points = []
            for p in b["polygon_world"]:
                px = ((p[0] - cx_m) / wx_m + 0.5) * img_w
                pz = ((p[1] - cz_m) / wz_m + 0.5) * img_h
                points.append((px, pz))

        elif b.get("polygon_px"):
            # Ещё более старый формат: пиксели оригинального изображения == мировые
            points = [tuple(p) for p in b["polygon_px"]]

        else:
            # Самый старый формат: только bbox
            cx, cy = b["center"]["x"], b["center"]["z"]
            w2, h2 = b["size"]["width"], b["size"]["length"]
            ang    = b["rotation_degrees"]
            box    = cv2.boxPoints(((cx, cy), (w2, h2), ang)).astype(np.float32)
            points = [tuple(p) for p in box]

        entry = compute_building(name, points)
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
    current_points = []   # мировые пиксельные координаты
    current_idx    = len(done)
    mode           = MODE_DRAW
    ref_pts        = []   # мировые пиксельные координаты
    mouse          = [None, None]
    img_offset     = [0, 0]
    saved_offset   = [0, 0]   # сохранённый offset при входе в ALIGN (для отмены)
    align_drag     = [False]
    align_last     = [0, 0]
    panel_cmd      = [None]

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

        # ── Режим выравнивания: drag двигает карту ──────────────────────────
        if mode == MODE_ALIGN:
            if event == cv2.EVENT_LBUTTONDOWN:
                align_drag[0] = True
                align_last[0], align_last[1] = x, y
            elif event == cv2.EVENT_MOUSEMOVE and align_drag[0]:
                img_offset[0] += x - align_last[0]
                img_offset[1] += y - align_last[1]
                align_last[0], align_last[1] = x, y
            elif event == cv2.EVENT_LBUTTONUP:
                align_drag[0] = False
            return

        if event != cv2.EVENT_LBUTTONDOWN:
            return

        # Конвертируем экран → мировые координаты
        wmx, wmy = screen_to_world(x, y, img_offset[0], img_offset[1])
        pt_w, is_s = snap(wmx, wmy, done)
        pt = pt_w if is_s else (wmx, wmy)

        if mode == MODE_REF:
            ref_pts.append(pt)
            if len(ref_pts) == 2:
                current_points = [ref_pts[1]]
                mode = MODE_DRAW
                ref_pts = []
            return
        current_points.append(pt)

    cv2.setMouseCallback(window, on_mouse)

    def handle_action(action):
        nonlocal current_points, current_idx, mode, ref_pts
        if action == "confirm":
            if mode == MODE_ALIGN:
                # Подтверждаем выравнивание: img_offset уже корректный, просто выходим
                mode = MODE_DRAW
            elif mode == MODE_DRAW and len(current_points) >= 3:
                b = compute_building(building_name(current_idx), current_points)
                done.append(b)
                current_points = []
                current_idx += 1

        elif action == "align":
            if mode == MODE_DRAW:
                # Входим в режим выравнивания, сохраняем текущий offset для отмены
                saved_offset[0], saved_offset[1] = img_offset[0], img_offset[1]
                align_drag[0] = False
                mode = MODE_ALIGN
            elif mode == MODE_ALIGN:
                # Отмена: восстанавливаем offset
                img_offset[0], img_offset[1] = saved_offset[0], saved_offset[1]
                mode = MODE_DRAW

        elif action == "ref":
            if mode != MODE_ALIGN:
                mode = MODE_REF if mode == MODE_DRAW else MODE_DRAW
                ref_pts = []

        elif action == "undo":
            if mode == MODE_REF:
                if ref_pts: ref_pts.pop()
                else: mode = MODE_DRAW
            elif mode == MODE_DRAW:
                if current_points:
                    current_points.pop()
                elif done:
                    last = done.pop()
                    current_idx -= 1
                    current_points = [tuple(p) for p in last["polygon"]]
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
            return True

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

        if panel_cmd[0]:
            if handle_action(panel_cmd[0]):
                break
            panel_cmd[0] = None

        if key in (ord('q'), 27):
            break
        elif key in (13,):          # Enter
            handle_action("confirm")
        elif key in (ord('a'), ord('A')):
            handle_action("align")
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

        sf = tk.LabelFrame(root, text="Screenshot covers (meters)")
        sf.grid(row=2, column=0, columnspan=3, sticky="ew", **P)
        for col, (lbl, attr, default) in enumerate([
            ("Width X:", "wx_var", Cfg.world_x),
            ("Depth Z:", "wz_var", Cfg.world_z),
        ]):
            tk.Label(sf, text=lbl).grid(row=0, column=col * 2, **P)
            v = tk.StringVar(value=str(default)); setattr(self, attr, v)
            tk.Entry(sf, textvariable=v, width=8).grid(row=0, column=col * 2 + 1, **P)

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
            self.listbox.insert(tk.END, name)
            self.name_var.set("")

    def _remove(self):
        sel = self.listbox.curselection()
        if sel:
            self.listbox.delete(sel[0])

    def _move_up(self):
        sel = self.listbox.curselection()
        if sel and sel[0] > 0:
            i = sel[0]; val = self.listbox.get(i)
            self.listbox.delete(i); self.listbox.insert(i - 1, val)
            self.listbox.select_set(i - 1)

    def _launch(self):
        map_path = self.img_var.get().strip()
        if not map_path or not os.path.exists(map_path):
            messagebox.showerror("Error", "Select a valid map file.")
            return
        try:
            wx = float(self.wx_var.get())
            wz = float(self.wz_var.get())
        except ValueError:
            messagebox.showerror("Error", "Width and Depth must be numbers.")
            return
        Cfg.map_path    = map_path
        Cfg.output_path = self.out_var.get().strip() or self._default_out()
        Cfg.world_x     = wx
        Cfg.world_z     = wz
        Cfg.buildings   = list(self.listbox.get(0, tk.END))
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
