import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
import os
import random
from PIL import Image, ImageTk

# ---------------------- Utility functions ----------------------

def load_image(image_path):
    """Load image with Pillow and return RGB numpy array."""
    try:
        img = Image.open(image_path).convert("RGB")
        return np.array(img)
    except Exception as e:
        print(f"Ошибка при загрузке изображения {image_path}: {e}")
        return None


def save_image_rgb(path, arr):
    """Save RGB numpy array using OpenCV (convert to BGR)."""
    try:
        bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        cv2.imwrite(path, bgr)
        return True
    except Exception as e:
        print(f"Ошибка сохранения {path}: {e}")
        return False


def extract_tiles_from_image(img, grid_n):
    """Возвращает список плиток (numpy arrays) для данного изображения, разбитого на grid_n x grid_n."""
    h, w = img.shape[:2]
    ys = [0] + [ (h * i) // grid_n for i in range(1, grid_n) ] + [h]
    xs = [0] + [ (w * i) // grid_n for i in range(1, grid_n) ] + [w]
    tiles = []
    for ry in range(grid_n):
        for rx in range(grid_n):
            y0, y1 = ys[ry], ys[ry+1]
            x0, x1 = xs[rx], xs[rx+1]
            tile = img[y0:y1, x0:x1].copy()
            tiles.append(tile)
    return tiles


def create_mosaic(image, grid_n=2, iterations=1, pool_images=None):
    """
    Создаёт мозаику для заданного image:
    - если pool_images пустой/None -> плитки берутся из самого изображения (как раньше)
    - если pool_images задан -> собирается пул плиток со всех изображений в pool_images и плитки для результирующей картинки берутся случайно из этого пула
    iterations — сколько проходов перемешивания (последовательные перестановки плиток)
    Возвращает RGB numpy array размером как image.
    """
    if image is None:
        return None
    h, w = image.shape[:2]
    if grid_n <= 1:
        return image.copy()

    # Подготовим координаты целевых плиток (по целевой картинке)
    ys = [0] + [ (h * i) // grid_n for i in range(1, grid_n) ] + [h]
    xs = [0] + [ (w * i) // grid_n for i in range(1, grid_n) ] + [w]

    # Если есть пул — подготовим плитки из пула
    pool_tiles = None
    if pool_images:
        pool_tiles = []
        for pimg in pool_images:
            try:
                tiles = extract_tiles_from_image(pimg, grid_n)
                pool_tiles.extend(tiles)
            except Exception:
                continue
        # Если пул пуст — fallback к обычному режиму
        if not pool_tiles:
            pool_tiles = None

    # Если пул есть — будем для каждого целевого положения брать случайную плитку из пула
    out = image.copy()

    for _ in range(max(1, iterations)):
        new_img = np.zeros_like(image)
        for ry in range(grid_n):
            for rx in range(grid_n):
                y0, y1 = ys[ry], ys[ry+1]
                x0, x1 = xs[rx], xs[rx+1]
                th, tw = y1 - y0, x1 - x0
                if pool_tiles is None:
                    # классический режим: берем плитки из самой картинки и перемешиваем их местами
                    # сначала разрежем исходный out на плитки и перемешаем
                    # реализуем простым способом: соберём все плитки, перемешаем индексный список и возьмём соответствующую
                    pass
        # Если pool отсутствует — используем старую логику (перетасовка плиток внутри одной картинки)
        if pool_tiles is None:
            # разбиваем текущую картинку на плитки
            tiles = extract_tiles_from_image(out, grid_n)
            perm = list(range(len(tiles)))
            random.shuffle(perm)
            tgt_idx = 0
            for ry in range(grid_n):
                for rx in range(grid_n):
                    y0, y1 = ys[ry], ys[ry+1]
                    x0, x1 = xs[rx], xs[rx+1]
                    th, tw = y1 - y0, x1 - x0
                    src_tile = tiles[perm[tgt_idx]]
                    if src_tile.shape[0] != th or src_tile.shape[1] != tw:
                        src_tile = cv2.resize(src_tile, (tw, th), interpolation=cv2.INTER_LINEAR)
                    new_img[y0:y1, x0:x1] = src_tile
                    tgt_idx += 1
            out = new_img
        else:
            # Пул-режим: для каждого места берём случайную плитку из pool_tiles
            for ry in range(grid_n):
                for rx in range(grid_n):
                    y0, y1 = ys[ry], ys[ry+1]
                    x0, x1 = xs[rx], xs[rx+1]
                    th, tw = y1 - y0, x1 - x0
                    src_tile = random.choice(pool_tiles)
                    if src_tile.shape[0] != th or src_tile.shape[1] != tw:
                        src_tile = cv2.resize(src_tile, (tw, th), interpolation=cv2.INTER_LINEAR)
                    new_img[y0:y1, x0:x1] = src_tile
            out = new_img

    return out


# ---------------------- Tkinter application ----------------------

class MosaicApp:
    def __init__(self, root):
        self.root = root
        root.title("Image Mosaic Shuffler — Dashboard")
        root.geometry("720x520")

        # Data
        self.image_paths = []
        self.image_arrays = {}  # path -> numpy array
        self.preview_images = {}  # path -> PhotoImage (to keep references)
        self.mosaic_results = {}  # path -> mosaic numpy array

        # Pool management
        self.pool_paths = []  # список путей, включённых в пул плиток
        self.pool_mode_var = tk.BooleanVar(value=False)

        # Auto state
        self.auto_after_id = None
        self.auto_running = False

        # --- Left control frame ---
        ctrl = tk.Frame(root, padx=8, pady=8)
        ctrl.pack(side=tk.LEFT, fill=tk.Y)

        tk.Button(ctrl, text="Добавить изображения", width=20, command=self.add_images).pack(pady=4)
        tk.Button(ctrl, text="Удалить выделенные", width=20, command=self.remove_selected).pack(pady=4)
        tk.Button(ctrl, text="Очистить всё", width=20, command=self.clear_all).pack(pady=4)

        tk.Label(ctrl, text="Список изображений:", anchor="w").pack(anchor="w", pady=(10,0))
        self.listbox = tk.Listbox(ctrl, selectmode=tk.EXTENDED, width=30, height=8)
        self.listbox.pack()

        # Pool controls
        tk.Label(ctrl, text="Пул плиток (Use pool mode):", anchor="w").pack(anchor="w", pady=(8,0))
        self.pool_checkbox = tk.Checkbutton(ctrl, text="Включить пул плиток (использовать плитки из нескольких изображений)", variable=self.pool_mode_var)
        self.pool_checkbox.pack(anchor="w")

        # Pool listbox
        self.pool_listbox = tk.Listbox(ctrl, selectmode=tk.EXTENDED, width=30, height=6)
        self.pool_listbox.pack(pady=(4,0))
        pool_btns = tk.Frame(ctrl)
        pool_btns.pack(pady=(4,4))
        tk.Button(pool_btns, text="Добавить в пул", width=10, command=self.add_selected_to_pool).pack(side=tk.LEFT, padx=4)
        tk.Button(pool_btns, text="Убрать из пула", width=10, command=self.remove_selected_from_pool).pack(side=tk.LEFT, padx=4)
        tk.Button(pool_btns, text="Очистить пул", width=10, command=self.clear_pool).pack(side=tk.LEFT, padx=4)

        # Parameters
        tk.Label(ctrl, text="Grid (n x n):").pack(anchor="w", pady=(8,0))
        self.entry_grid = tk.Entry(ctrl, width=8)
        self.entry_grid.insert(0, "2")
        self.entry_grid.pack()

        tk.Label(ctrl, text="Iterations (shuffles):").pack(anchor="w", pady=(8,0))
        self.entry_iters = tk.Entry(ctrl, width=8)
        self.entry_iters.insert(0, "1")
        self.entry_iters.pack()

        tk.Label(ctrl, text="Auto interval (ms):").pack(anchor="w", pady=(8,0))
        self.entry_interval = tk.Entry(ctrl, width=8)
        self.entry_interval.insert(0, "1000")
        self.entry_interval.pack()

        btn_frame = tk.Frame(ctrl)
        btn_frame.pack(pady=(10,4))
        tk.Button(btn_frame, text="Генерировать сейчас", width=14, command=self.generate_now).pack(side=tk.LEFT, padx=4)
        self.auto_btn = tk.Button(btn_frame, text="Start Auto", width=14, command=self.toggle_auto)
        self.auto_btn.pack(side=tk.LEFT, padx=4)

        tk.Button(ctrl, text="Открыть окно мозаик", width=38, command=self.open_mosaic_window).pack(pady=(8,4))
        tk.Button(ctrl, text="Сохранить все мозаики (Рабочий стол)", width=38, command=self.save_all).pack(pady=(4,4))

        # Status
        self.status_var = tk.StringVar(value="Ready")
        tk.Label(ctrl, textvariable=self.status_var, fg="blue").pack(pady=(8,0))

        # --- Right: preview ---
        right = tk.Frame(root, padx=8, pady=8)
        right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        tk.Label(right, text="Превью (выбранное):").pack(anchor="nw")
        self.preview_canvas = tk.Canvas(right, width=360, height=360, bg="#eee")
        self.preview_canvas.pack(fill=tk.BOTH, expand=True)

        # Bind selection
        self.listbox.bind("<<ListboxSelect>>", self.on_listbox_select)

        # Mosaic window reference
        self.mosaic_window = None

    # ---------- Image & pool management ----------
    def add_images(self):
        paths = filedialog.askopenfilenames(title="Выберите изображения",
                                            filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.bmp;*.gif")])
        if not paths:
            return
        for p in paths:
            if p in self.image_paths:
                continue
            arr = load_image(p)
            if arr is None:
                continue
            self.image_paths.append(p)
            self.image_arrays[p] = arr
            self.listbox.insert(tk.END, os.path.basename(p))
            # по умолчанию не добавляем в пул — пользователь решает
        self.status_var.set(f"Загружено {len(self.image_paths)} изображений")

    def add_selected_to_pool(self):
        sel = self.listbox.curselection()
        if not sel:
            return
        for i in sel:
            p = self.image_paths[i]
            if p not in self.pool_paths:
                self.pool_paths.append(p)
                self.pool_listbox.insert(tk.END, os.path.basename(p))
        self.status_var.set(f"В пуле {len(self.pool_paths)} изображений")

    def remove_selected_from_pool(self):
        sel = list(self.pool_listbox.curselection())
        if not sel:
            return
        for i in reversed(sel):
            name = self.pool_listbox.get(i)
            full = None
            for p in self.pool_paths:
                if os.path.basename(p) == name:
                    full = p
                    break
            if full:
                self.pool_paths.remove(full)
            self.pool_listbox.delete(i)
        self.status_var.set(f"В пуле {len(self.pool_paths)} изображений")

    def clear_pool(self):
        self.pool_paths.clear()
        self.pool_listbox.delete(0, tk.END)
        self.status_var.set("Пул очищен")

    def remove_selected(self):
        sel = list(self.listbox.curselection())
        if not sel:
            return
        for i in reversed(sel):
            name = self.listbox.get(i)
            # find full path by basename match (first match)
            full = None
            for p in self.image_paths:
                if os.path.basename(p) == name:
                    full = p
                    break
            if full:
                self.image_paths.remove(full)
                self.image_arrays.pop(full, None)
                self.mosaic_results.pop(full, None)
                self.preview_images.pop(full, None)
                # also remove from pool if present
                if full in self.pool_paths:
                    idx = self.pool_paths.index(full)
                    self.pool_paths.remove(full)
                    try:
                        self.pool_listbox.delete(idx)
                    except Exception:
                        pass
            self.listbox.delete(i)
        self.preview_canvas.delete("all")
        self.status_var.set(f"{len(self.image_paths)} изображений осталось")

    def clear_all(self):
        self.image_paths.clear()
        self.image_arrays.clear()
        self.mosaic_results.clear()
        self.preview_images.clear()
        self.pool_paths.clear()
        self.listbox.delete(0, tk.END)
        self.pool_listbox.delete(0, tk.END)
        self.preview_canvas.delete("all")
        self.status_var.set("Очищено")

    # ---------- Generation ----------
    def generate_now(self):
        sel_indices = self.listbox.curselection()
        if not sel_indices:
            messagebox.showwarning("Внимание", "Выберите хотя бы одно изображение в списке.")
            return
        try:
            grid_n = max(1, int(self.entry_grid.get()))
        except:
            messagebox.showerror("Ошибка", "Grid должен быть целым числом >=1")
            return
        try:
            iters = max(1, int(self.entry_iters.get()))
        except:
            iters = 1

        # Подготовим пул массивов, если пул включён
        pool_arrays = None
        if self.pool_mode_var.get() and self.pool_paths:
            pool_arrays = [self.image_arrays[p] for p in self.pool_paths if p in self.image_arrays]
            if not pool_arrays:
                pool_arrays = None

        selected_paths = [self.image_paths[i] for i in sel_indices]
        for p in selected_paths:
            arr = self.image_arrays.get(p)
            mosaic = create_mosaic(arr, grid_n=grid_n, iterations=iters, pool_images=pool_arrays)
            self.mosaic_results[p] = mosaic
        self.status_var.set(f"Сгенерировано мозаик для {len(selected_paths)} изображений")
        # Update preview of first selected and mosaic window if opened
        self.update_preview(selected_paths[0])
        if self.mosaic_window:
            self.update_mosaic_window()

    # ---------- Auto-generation ----------
    def toggle_auto(self):
        if self.auto_running:
            self.stop_auto()
        else:
            self.start_auto()

    def start_auto(self):
        if not self.image_paths:
            messagebox.showwarning("Внимание", "Добавьте изображения перед автогенерацией.")
            return
        try:
            interval = int(self.entry_interval.get())
        except:
            messagebox.showerror("Ошибка", "Интервал должен быть целым миллисекунд.")
            return
        if interval < 100:
            messagebox.showerror("Ошибка", "Интервал должен быть >=100 ms.")
            return
        self.auto_running = True
        self.auto_btn.config(text="Stop Auto")
        self._auto_step()

    def _auto_step(self):
        if not self.auto_running:
            return
        # Для авто генерируем для всех выделенных, а если выделения нет — для всех в списке
        sel = self.listbox.curselection()
        if sel:
            to_process = [self.image_paths[i] for i in sel]
        else:
            to_process = list(self.image_paths)
        try:
            grid_n = max(1, int(self.entry_grid.get()))
        except:
            grid_n = 2
        try:
            iters = max(1, int(self.entry_iters.get()))
        except:
            iters = 1
        pool_arrays = None
        if self.pool_mode_var.get() and self.pool_paths:
            pool_arrays = [self.image_arrays[p] for p in self.pool_paths if p in self.image_arrays]
            if not pool_arrays:
                pool_arrays = None
        for p in to_process:
            arr = self.image_arrays.get(p)
            mosaic = create_mosaic(arr, grid_n=grid_n, iterations=iters, pool_images=pool_arrays)
            self.mosaic_results[p] = mosaic
        self.status_var.set(f"Auto-generated {len(to_process)} mosaics")
        if to_process:
            self.update_preview(to_process[0])
        if self.mosaic_window:
            self.update_mosaic_window()
        try:
            interval = max(100, int(self.entry_interval.get()))
        except:
            interval = 1000
        self.auto_after_id = self.root.after(interval, self._auto_step)

    def stop_auto(self):
        if self.auto_after_id:
            try:
                self.root.after_cancel(self.auto_after_id)
            except Exception:
                pass
            self.auto_after_id = None
        self.auto_running = False
        self.auto_btn.config(text="Start Auto")
        self.status_var.set("Auto stopped")

    # ---------- Preview and Mosaic window ----------
    def on_listbox_select(self, event=None):
        sel = self.listbox.curselection()
        if not sel:
            return
        p = self.image_paths[sel[0]]
        # show mosaic if exists else original
        if p in self.mosaic_results:
            self.update_preview(p)
        else:
            self.update_preview(p, show_mosaic=False)

    def update_preview(self, path, show_mosaic=True):
        arr = None
        if show_mosaic and path in self.mosaic_results:
            arr = self.mosaic_results[path]
        else:
            arr = self.image_arrays.get(path)
        if arr is None:
            return
        pil = Image.fromarray(arr.copy())
        # scale to fit preview canvas
        cw = self.preview_canvas.winfo_width() or 360
        ch = self.preview_canvas.winfo_height() or 360
        scale = min(1.0, min(cw / pil.width, ch / pil.height))
        if scale < 1.0:
            pil = pil.resize((int(pil.width * scale), int(pil.height * scale)), Image.LANCZOS)
        imgtk = ImageTk.PhotoImage(pil)
        self.preview_canvas.delete("all")
        self.preview_canvas.create_image(0, 0, anchor="nw", image=imgtk)
        # keep reference
        self.preview_images[path] = imgtk

    def open_mosaic_window(self):
        if self.mosaic_window and tk.Toplevel.winfo_exists(self.mosaic_window):
            self.mosaic_window.lift()
            return
        self.mosaic_window = tk.Toplevel(self.root)
        self.mosaic_window.title("Mosaic Preview")
        self.mosaic_window.geometry("900x600")

        # Scrollable canvas
        canvas = tk.Canvas(self.mosaic_window, bg="#fff")
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        vbar = tk.Scrollbar(self.mosaic_window, orient=tk.VERTICAL, command=canvas.yview)
        vbar.pack(side=tk.RIGHT, fill=tk.Y)
        canvas.configure(yscrollcommand=vbar.set)
        self.mosaic_frame = tk.Frame(canvas)
        canvas.create_window((0, 0), window=self.mosaic_frame, anchor='nw')
        self.mosaic_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox('all')))

        # Populate
        self.update_mosaic_window()

        # When window closed, drop reference
        self.mosaic_window.protocol("WM_DELETE_WINDOW", self.on_mosaic_window_close)

    def on_mosaic_window_close(self):
        try:
            self.mosaic_window.destroy()
        except:
            pass
        self.mosaic_window = None

    def update_mosaic_window(self):
        if not (self.mosaic_window and tk.Toplevel.winfo_exists(self.mosaic_window)):
            return
        # Clear frame
        for w in self.mosaic_frame.winfo_children():
            w.destroy()
        # Build grid of mosaics for images that have results (or show original if none)
        items = list(self.image_paths)
        if not items:
            tk.Label(self.mosaic_frame, text="No images loaded").pack()
            return
        cols = 3
        thumb_w = 260
        thumb_h = 180
        col = 0
        row = 0
        for p in items:
            arr = self.mosaic_results.get(p, self.image_arrays.get(p))
            if arr is None:
                continue
            pil = Image.fromarray(arr.copy())
            # Fit into thumbnail box
            scale = min(1.0, min(thumb_w / pil.width, thumb_h / pil.height))
            if scale < 1.0:
                pil = pil.resize((int(pil.width * scale), int(pil.height * scale)), Image.LANCZOS)
            imgtk = ImageTk.PhotoImage(pil)
            frm = tk.Frame(self.mosaic_frame, bd=1, relief=tk.RIDGE)
            lbl = tk.Label(frm, image=imgtk)
            lbl.image = imgtk  # keep reference
            lbl.pack()
            caption = tk.Label(frm, text=os.path.basename(p), wraplength=thumb_w)
            caption.pack()
            frm.grid(row=row, column=col, padx=6, pady=6)
            col += 1
            if col >= cols:
                col = 0
                row += 1

    # ---------- Save ----------
    def save_all(self):
        if not self.mosaic_results:
            messagebox.showwarning("Внимание", "Нет результатов для сохранения. Сначала сгенерируйте мозаики.")
            return
        desktop = os.path.join(os.path.expanduser("~"), "Desktop")
        saved = 0
        for p, arr in self.mosaic_results.items():
            base = os.path.splitext(os.path.basename(p))[0]
            path = os.path.join(desktop, f"{base}_mosaic.png")
            ok = save_image_rgb(path, arr)
            if ok:
                saved += 1
        messagebox.showinfo("Сохранено", f"Сохранено {saved} изображений на рабочий стол.")


# ---------------------- Main ----------------------

def main():
    root = tk.Tk()
    app = MosaicApp(root)
    root.mainloop()


if __name__ == '__main__':
    main()
