"""
map_annotator.py — разметка зданий на карте (Tkinter Canvas).

Установка: pip install opencv-python numpy pillow
Запуск:    python map_annotator.py

Горячие клавиши: Enter=confirm, A=align, R=ref, Z=undo, S=save, Q/Esc=quit

Canvas = World Space.
  - Карта лежит на холсте в позиции (img_offset_x, img_offset_y).
  - Здания рисуются строго по мировым (canvas) координатам — без offset.
  - Клик в DRAW: world_x = event.x, world_y = event.y (напрямую).
  - ALIGN: drag двигает только карту; контуры зданий — неподвижный трафарет.
"""

import cv2
import numpy as np
import json
import os
import math
import io
import tkinter as tk
from tkinter import filedialog, messagebox

try:
    from PIL import Image, ImageTk
except ImportError:
    raise SystemExit("Установи Pillow: pip install pillow")


# ── Конфиг ───────────────────────────────────────────────────────────────────

class Cfg:
    map_path    = ""
    output_path = ""
    world_x     = 180.0   # метров по X (только для REF-дистанции)
    world_z     = 146.0   # метров по Z
    buildings   = ["House_6", "House_29", "House_27"]
    snap_radius = 12


MODE_DRAW  = 0
MODE_REF   = 1
MODE_ALIGN = 2

PANEL_W  = 180

# (label, normal_bg, active_bg, cmd)
BTN_CFG = [
    ("Confirm  Enter", "#1a7a1a", "#3aaa3a", "confirm"),
    ("Align      A",   "#707000", "#a0a020", "align"),
    ("Ref arrow  R",   "#6a006a", "#9a309a", "ref"),
    ("Undo       Z",   "#005870", "#1088a0", "undo"),
    ("Save       S",   "#0a4080", "#2a70c0", "save"),
    ("Exit       Q",   "#0a0a50", "#303080", "quit"),
]


# ── Бизнес-логика: здания и JSON ──────────────────────────────────────────────

def building_name(idx):
    return Cfg.buildings[idx] if idx < len(Cfg.buildings) else f"Object_{idx + 1}"


def compute_building(name, world_points):
    """Принимает мировые (canvas) координаты, возвращает запись здания."""
    pts = np.array(world_points, dtype=np.float32)
    rect = cv2.minAreaRect(pts)
    (cx, cy), (w, h), angle = rect
    return {
        "name":      name,
        "center_px": (float(cx), float(cy)),
        "polygon":   [list(map(float, p)) for p in world_points],
        "width":     float(w),
        "length":    float(h),
        "angle":     float(angle),
    }


def build_json(img_w, img_h, done):
    buildings = []
    for b in done:
        buildings.append({
            "name":             b["name"],
            "center":           {"x": round(b["center_px"][0], 2),
                                 "z": round(b["center_px"][1], 2)},
            "size":             {"width":  round(b["width"],  2),
                                 "length": round(b["length"], 2)},
            "rotation_degrees": round(b["angle"], 4),
            "polygon_world":    [[round(p[0], 2), round(p[1], 2)] for p in b["polygon"]],
        })
    return {
        "format":     "pixel_world_v2",
        "map_size":   {"width": img_w, "height": img_h},
        "world_meta": {"world_x": Cfg.world_x, "world_z": Cfg.world_z},
        "buildings":  buildings,
    }


def load_existing(img_w, img_h):
    if not os.path.exists(Cfg.output_path):
        return []
    try:
        with open(Cfg.output_path, encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"Cannot load JSON: {e}")
        return []

    fmt    = data.get("format", "")
    result = []

    for b in data.get("buildings", []):
        name = b["name"]

        if fmt == "pixel_world_v2" and b.get("polygon_world"):
            points = [tuple(p) for p in b["polygon_world"]]

        elif b.get("polygon_world"):
            # Старый формат: метровые координаты Godot → canvas-пиксели
            meta = data.get("world_meta", {})
            wx_m = meta.get("world_x",   Cfg.world_x)
            wz_m = meta.get("world_z",   Cfg.world_z)
            cx_m = meta.get("center_wx", 0.0)
            cz_m = meta.get("center_wz", 0.0)
            points = [
                (((p[0] - cx_m) / wx_m + 0.5) * img_w,
                 ((p[1] - cz_m) / wz_m + 0.5) * img_h)
                for p in b["polygon_world"]
            ]

        elif b.get("polygon_px"):
            points = [tuple(p) for p in b["polygon_px"]]

        else:
            cx_b, cy_b = b["center"]["x"], b["center"]["z"]
            w2, h2 = b["size"]["width"], b["size"]["length"]
            ang = b["rotation_degrees"]
            box = cv2.boxPoints(((cx_b, cy_b), (w2, h2), ang)).astype(np.float32)
            points = [tuple(p) for p in box]

        result.append(compute_building(name, points))
        print(f"Loaded: {name}")

    return result


# ── Аннотатор ─────────────────────────────────────────────────────────────────

class AnnotatorWindow:
    def __init__(self, root: tk.Tk):
        self.root = root
        root.title("map_annotator")
        root.configure(bg="#111111")

        # Загрузка карты
        with open(Cfg.map_path, "rb") as fh:
            raw = fh.read()
        self.pil_img = Image.open(io.BytesIO(raw))
        self.pil_img.load()
        self.img_w, self.img_h = self.pil_img.size
        self.map_photo = ImageTk.PhotoImage(self.pil_img)

        # Состояние
        self.done           = load_existing(self.img_w, self.img_h)
        self.current_points = []   # мировые (canvas) координаты
        self.current_idx    = len(self.done)
        self.mode           = MODE_DRAW
        self.ref_pts        = []

        # Смещение карты
        self.img_offset_x = 0
        self.img_offset_y = 0
        self._saved_ox    = 0
        self._saved_oy    = 0

        # Drag-состояние
        self._dragging  = False
        self._drag_lx   = 0
        self._drag_ly   = 0

        self._build_ui()
        self._bind()
        self.redraw_static()

    # ── UI ────────────────────────────────────────────────────────────────────

    def _build_ui(self):
        sw = self.root.winfo_screenwidth()
        sh = self.root.winfo_screenheight()
        cw = min(self.img_w, sw - PANEL_W - 16)
        ch = min(self.img_h, sh - 80)
        self.root.geometry(f"{cw + PANEL_W}x{ch + 28}")
        self.root.resizable(True, True)

        wrap = tk.Frame(self.root, bg="#111111")
        wrap.pack(fill="both", expand=True)

        self.canvas = tk.Canvas(wrap, bg="#1a1a1a",
                                 highlightthickness=0, cursor="crosshair")
        self.canvas.pack(side="left", fill="both", expand=True)

        panel = tk.Frame(wrap, width=PANEL_W, bg="#1c1c1c")
        panel.pack(side="right", fill="y")
        panel.pack_propagate(False)
        self._build_panel(panel)

        self.status_var = tk.StringVar()
        tk.Label(self.root, textvariable=self.status_var,
                  bg="#141414", fg="#d8d8d8", font=("Courier", 9),
                  anchor="w", padx=8).pack(side="bottom", fill="x")

    def _build_panel(self, panel):
        tk.Label(panel, bg="#1c1c1c").pack(pady=4)

        self.lbl_mode  = tk.Label(panel, text="Mode: DRAW",
                                   bg="#1c1c1c", fg="#c0c0c0",
                                   font=("Courier", 10, "bold"), anchor="w")
        self.lbl_mode.pack(fill="x", padx=8)

        self.lbl_bname = tk.Label(panel, text="",
                                   bg="#1c1c1c", fg="#90cc90",
                                   font=("Courier", 9), anchor="w")
        self.lbl_bname.pack(fill="x", padx=8)

        self.lbl_pts   = tk.Label(panel, text="0 pts",
                                   bg="#1c1c1c", fg="#707070",
                                   font=("Courier", 9), anchor="w")
        self.lbl_pts.pack(fill="x", padx=8, pady=(0, 8))

        self.btn_refs = {}
        for label, norm_bg, _, cmd in BTN_CFG:
            b = tk.Button(panel, text=label,
                           bg=norm_bg, fg="white",
                           font=("Courier", 9), relief="flat",
                           padx=6, pady=5,
                           activebackground="#606060",
                           command=lambda c=cmd: self.action(c))
            b.pack(fill="x", padx=6, pady=2)
            self.btn_refs[cmd] = b

        tk.Frame(panel, bg="#404040", height=1).pack(fill="x", padx=6, pady=6)
        tk.Label(panel, text="Done:", bg="#1c1c1c", fg="#888888",
                  font=("Courier", 9), anchor="w").pack(fill="x", padx=8)

        self.done_frame = tk.Frame(panel, bg="#1c1c1c")
        self.done_frame.pack(fill="both", padx=8)

    # ── Отрисовка ─────────────────────────────────────────────────────────────

    def redraw_static(self):
        """Полная перерисовка: карта + здания + зафиксированные точки."""
        self.canvas.delete("map", "building", "current", "dynamic")

        # Карта — смещаемый слой
        self.canvas.create_image(self.img_offset_x, self.img_offset_y,
                                   anchor="nw", image=self.map_photo,
                                   tags="map")

        # Здания — строго по мировым координатам (без offset)
        for b in self.done:
            flat = [c for p in b["polygon"] for c in (float(p[0]), float(p[1]))]
            if len(flat) >= 6:
                self.canvas.create_polygon(
                    *flat,
                    fill="#003a00", stipple="gray25",
                    outline="#00c800", width=2,
                    tags="building")
            cx, cy = b["center_px"]
            self.canvas.create_text(
                int(cx), int(cy),
                text=b["name"],
                fill="#00c800", font=("Helvetica", 9, "bold"),
                tags="building")

        # Зафиксированные точки текущего полигона
        if len(self.current_points) >= 2:
            flat = [c for p in self.current_points for c in (float(p[0]), float(p[1]))]
            self.canvas.create_line(*flat, fill="#00b4ff", width=1, tags="current")
        for p in self.current_points:
            r = 4
            self.canvas.create_oval(p[0]-r, p[1]-r, p[0]+r, p[1]+r,
                                     fill="#0050ff", outline="",
                                     tags="current")

        self._refresh_panel()

    def redraw_dynamic(self, mx, my):
        """Быстрое частичное обновление: живая линия, snap, REF-стрелка."""
        self.canvas.delete("dynamic")

        if self.mode == MODE_DRAW and self.current_points:
            lx, ly = self.current_points[-1]
            self.canvas.create_line(lx, ly, mx, my,
                                     fill="#00b4ff", width=1, tags="dynamic")
            if len(self.current_points) >= 2:
                fx, fy = self.current_points[0]
                self.canvas.create_line(fx, fy, mx, my,
                                         fill="#7850c8", width=1,
                                         dash=(4, 4), tags="dynamic")

        if self.mode == MODE_DRAW:
            sp, snapped = self._snap(mx, my)
            if snapped:
                r = Cfg.snap_radius
                self.canvas.create_oval(sp[0]-r, sp[1]-r, sp[0]+r, sp[1]+r,
                                         outline="#00dcdc", width=2,
                                         tags="dynamic")

        if self.mode == MODE_REF and self.ref_pts:
            p0 = self.ref_pts[0]
            p1 = self.ref_pts[1] if len(self.ref_pts) >= 2 else (mx, my)
            self.canvas.create_line(p0[0], p0[1], p1[0], p1[1],
                                     fill="#c830e0", width=2,
                                     arrow="last", arrowshape=(12, 16, 4),
                                     tags="dynamic")
            self.canvas.create_oval(p0[0]-6, p0[1]-6, p0[0]+6, p0[1]+6,
                                     fill="#3c3cff", outline="white",
                                     tags="dynamic")
            if len(self.ref_pts) >= 2:
                self.canvas.create_oval(p1[0]-6, p1[1]-6, p1[0]+6, p1[1]+6,
                                         fill="#ff8200", outline="white",
                                         tags="dynamic")
                sx = Cfg.world_x / self.img_w
                sz = Cfg.world_z / self.img_h
                dist_m = math.hypot((p1[0]-p0[0])*sx, (p1[1]-p0[1])*sz)
                ang    = math.degrees(math.atan2(p1[1]-p0[1], p1[0]-p0[0]))
                self.canvas.create_text(
                    (p0[0]+p1[0])//2 + 8, (p0[1]+p1[1])//2 - 12,
                    text=f"{dist_m:.1f}m  {ang:.1f}°",
                    fill="#c830e0", font=("Courier", 9),
                    tags="dynamic")

    # ── Snap ─────────────────────────────────────────────────────────────────

    def _snap(self, wx, wy):
        best_d, best = Cfg.snap_radius + 1, (wx, wy)
        for b in self.done:
            for p in b["polygon"]:
                d = math.hypot(wx - p[0], wy - p[1])
                if d < best_d:
                    best_d, best = d, (p[0], p[1])
        return best, best_d <= Cfg.snap_radius

    # ── События ───────────────────────────────────────────────────────────────

    def on_press(self, event):
        x, y = event.x, event.y

        if self.mode == MODE_ALIGN:
            self._dragging = True
            self._drag_lx, self._drag_ly = x, y
            self.canvas.config(cursor="fleur")
            return

        # Координата клика = мировая координата (Canvas IS world space)
        sp, snapped = self._snap(x, y)
        pt = sp if snapped else (float(x), float(y))

        if self.mode == MODE_REF:
            self.ref_pts.append(pt)
            if len(self.ref_pts) == 2:
                self.current_points = [self.ref_pts[1]]
                self.mode  = MODE_DRAW
                self.ref_pts = []
                self.redraw_static()
            else:
                self.redraw_dynamic(x, y)
            return

        self.current_points.append(pt)
        self.redraw_static()
        self.redraw_dynamic(x, y)

    def on_drag(self, event):
        x, y = event.x, event.y
        if self.mode == MODE_ALIGN and self._dragging:
            dx = x - self._drag_lx
            dy = y - self._drag_ly
            self.img_offset_x += dx
            self.img_offset_y += dy
            self._drag_lx, self._drag_ly = x, y
            # Двигаем ТОЛЬКО карту; здания ("building") остаются неподвижными
            self.canvas.move("map", dx, dy)
            self._update_status()
        else:
            self.redraw_dynamic(x, y)

    def on_release(self, *_):
        if self.mode == MODE_ALIGN:
            self._dragging = False
            self.canvas.config(cursor="crosshair")

    def on_motion(self, event):
        if self.mode != MODE_ALIGN:
            self.redraw_dynamic(event.x, event.y)

    # ── Команды ───────────────────────────────────────────────────────────────

    def action(self, cmd):
        if cmd == "confirm":
            if self.mode == MODE_ALIGN:
                self.mode = MODE_DRAW
                self.canvas.config(cursor="crosshair")
                self.redraw_static()
            elif self.mode == MODE_DRAW and len(self.current_points) >= 3:
                b = compute_building(building_name(self.current_idx),
                                     self.current_points)
                self.done.append(b)
                self.current_points = []
                self.current_idx   += 1
                self.redraw_static()

        elif cmd == "align":
            if self.mode == MODE_DRAW:
                self._saved_ox, self._saved_oy = self.img_offset_x, self.img_offset_y
                self.mode = MODE_ALIGN
                self._dragging = False
                self.canvas.config(cursor="fleur")
                self._refresh_panel()
                self._update_status()
            elif self.mode == MODE_ALIGN:
                # Отмена — восстанавливаем offset
                dx = self._saved_ox - self.img_offset_x
                dy = self._saved_oy - self.img_offset_y
                self.img_offset_x = self._saved_ox
                self.img_offset_y = self._saved_oy
                self.canvas.move("map", dx, dy)
                self.mode = MODE_DRAW
                self.canvas.config(cursor="crosshair")
                self._refresh_panel()
                self._update_status()

        elif cmd == "ref":
            if self.mode != MODE_ALIGN:
                self.mode    = MODE_REF if self.mode == MODE_DRAW else MODE_DRAW
                self.ref_pts = []
                self._refresh_panel()
                self._update_status()

        elif cmd == "undo":
            if self.mode == MODE_REF:
                if self.ref_pts: self.ref_pts.pop()
                else: self.mode = MODE_DRAW
            elif self.mode == MODE_DRAW:
                if self.current_points:
                    self.current_points.pop()
                elif self.done:
                    last = self.done.pop()
                    self.current_idx  -= 1
                    self.current_points = [tuple(p) for p in last["polygon"]]
                    print(f"Undone: {last['name']}")
            self.redraw_static()

        elif cmd == "save":
            if not self.done:
                print("Nothing to save.")
                return
            data = build_json(self.img_w, self.img_h, self.done)
            with open(Cfg.output_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            print(f"Saved → {Cfg.output_path}")

        elif cmd == "quit":
            self.root.destroy()

    # ── Обновление панели / статуса ───────────────────────────────────────────

    def _refresh_panel(self):
        mode_str = ["DRAW", "REF", "ALIGN"][self.mode]
        self.lbl_mode.config(text=f"Mode: {mode_str}")
        self.lbl_bname.config(text=f"[{self.current_idx+1}] {building_name(self.current_idx)}")
        self.lbl_pts.config(text=f"{len(self.current_points)} pts")

        for _, norm_bg, act_bg, cmd in BTN_CFG:
            is_active = (cmd == "align" and self.mode == MODE_ALIGN) or \
                        (cmd == "ref"   and self.mode == MODE_REF)
            self.btn_refs[cmd].config(bg=act_bg if is_active else norm_bg)

        # Done-список
        for w in self.done_frame.winfo_children():
            w.destroy()
        for b in self.done:
            tk.Label(self.done_frame, text=b["name"],
                      bg="#1c1c1c", fg="#48c048",
                      font=("Courier", 9), anchor="w").pack(fill="x")

        self._update_status()

    def _update_status(self):
        if self.mode == MODE_ALIGN:
            ox, oy = self.img_offset_x, self.img_offset_y
            s = f"ALIGN  img_offset=({ox:+d}, {oy:+d})  |  drag=move map  Enter=confirm  A=cancel"
        elif self.mode == MODE_REF:
            steps = ["click anchor on existing building",
                     "click start of new building",
                     "locked — Z to reset"]
            s = "REF: " + steps[min(len(self.ref_pts), 2)]
        else:
            n     = len(self.current_points)
            hint  = "Enter=confirm" if n >= 3 else f"need {3-n} more pts"
            s = f"DRAW  [{self.current_idx+1}] {building_name(self.current_idx)}  {n} pts  |  {hint}"
        self.status_var.set(s)

    # ── Бинды ─────────────────────────────────────────────────────────────────

    def _bind(self):
        c = self.canvas
        c.bind("<ButtonPress-1>",   self.on_press)
        c.bind("<B1-Motion>",       self.on_drag)
        c.bind("<ButtonRelease-1>", self.on_release)
        c.bind("<Motion>",          self.on_motion)

        for key, act in [("<Return>", "confirm"),
                          ("<KeyPress-a>", "align"), ("<KeyPress-A>", "align"),
                          ("<KeyPress-r>", "ref"),   ("<KeyPress-R>", "ref"),
                          ("<KeyPress-z>", "undo"),  ("<KeyPress-Z>", "undo"),
                          ("<KeyPress-s>", "save"),  ("<KeyPress-S>", "save"),
                          ("<KeyPress-q>", "quit"),  ("<Escape>",     "quit")]:
            self.root.bind(key, lambda _, a=act: self.action(a))


# ── Лончер ────────────────────────────────────────────────────────────────────

class Launcher:
    def __init__(self, root: tk.Tk):
        self.root = root
        root.title("Map Annotator — Launcher")
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
            tk.Label(sf, text=lbl).grid(row=0, column=col*2, **P)
            v = tk.StringVar(value=str(default)); setattr(self, attr, v)
            tk.Entry(sf, textvariable=v, width=8).grid(row=0, column=col*2+1, **P)

        bf = tk.LabelFrame(root, text="Buildings (annotation order)")
        bf.grid(row=3, column=0, columnspan=3, sticky="ew", **P)
        self.listbox = tk.Listbox(bf, height=7, width=28, selectmode=tk.SINGLE)
        self.listbox.grid(row=0, column=0, rowspan=4, padx=8, pady=4)
        sb = tk.Scrollbar(bf, command=self.listbox.yview)
        sb.grid(row=0, column=1, rowspan=4, sticky="ns", pady=4)
        self.listbox.config(yscrollcommand=sb.set)
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
            i = sel[0]; v = self.listbox.get(i)
            self.listbox.delete(i); self.listbox.insert(i-1, v)
            self.listbox.select_set(i-1)

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
        # Морфируем лончер → аннотатор в том же окне
        for w in self.root.winfo_children():
            w.destroy()
        self.root.resizable(True, True)
        AnnotatorWindow(self.root)


# ── Точка входа ───────────────────────────────────────────────────────────────

def main():
    root = tk.Tk()
    root.protocol("WM_DELETE_WINDOW", root.destroy)
    Launcher(root)
    root.mainloop()


if __name__ == "__main__":
    main()
