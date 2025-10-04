import argparse
import os
import json
import numpy as np
from PIL import Image, ImageFilter
import random
import math

CACHE_FILENAME = "tiles_cache.json"


def average_color(img: Image.Image):
    """Возвращает кортеж трёх обычных int (R,G,B)."""
    arr = np.array(img)
    if arr.ndim == 2:  # grayscale
        v = int(arr.mean())
        return (v, v, v)
    vals = arr[:, :, :3].reshape(-1, 3).mean(axis=0)
    return tuple(int(v) for v in vals)


def compute_gradient_mean(img: Image.Image):
    """Вычисляет среднюю величину градиента для изображения (в оттенках серого).
    Возвращает float (приблизительно в том же масштабе, что и 0..255).
    """
    try:
        g = np.array(img.convert("L"), dtype=float)
        # simple finite difference
        dy = np.diff(g, axis=0)
        dx = np.diff(g, axis=1)
        dx_pad = np.zeros_like(g)
        dy_pad = np.zeros_like(g)
        dx_pad[:, :-1] = dx
        dy_pad[:-1, :] = dy
        mag = np.hypot(dx_pad, dy_pad)
        return float(mag.mean())
    except Exception:
        return 0.0


def load_tile_features(tiles_path: str, grid_size: int, cache_filename: str = CACHE_FILENAME):
    """
    Возвращает dict {path: {"color": (r,g,b), "grad": float}}.
    Кэш хранится в файле tiles_cache.json внутри папки tiles_path, структура:
    { "<full_path>": {"mtime": 123456789.0, "color": [r,g,b], "grad": 12.34} }
    """
    cache_path = os.path.join(tiles_path, cache_filename)
    cache = {}
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                cache = json.load(f)
        except Exception:
            print("Warning: cache is corrupted or unreadable — rebuilding cache.")

    tile_features = {}
    updated_cache = {}

    for fname in os.listdir(tiles_path):
        if not fname.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".webp")):
            continue
        full = os.path.join(tiles_path, fname)
        try:
            mtime = os.path.getmtime(full)
        except OSError:
            continue

        cached = cache.get(full)
        if cached and float(cached.get("mtime", 0.0)) == mtime:
            col_raw = cached.get("color", [0, 0, 0])
            color = tuple(int(c) for c in col_raw)
            grad = float(cached.get("grad", 0.0))
        else:
            try:
                img = Image.open(full).convert("RGB").resize((grid_size, grid_size), Image.Resampling.LANCZOS)
                color = average_color(img)
                grad = compute_gradient_mean(img)
            except Exception as e:
                print(f"Failed to process tile {full}: {e}")
                continue

        tile_features[full] = {"color": color, "grad": float(grad)}
        updated_cache[full] = {"mtime": mtime, "color": [int(c) for c in color], "grad": float(grad)}

    try:
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(updated_cache, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"Could not write cache file '{cache_path}': {e}")

    return tile_features


def find_best_tile_for_target(target, tile_features, usage_count=None, max_usage=0, metric='color', grad_weight=0.5):
    """
    Возвращает путь к лучшему тайлу по выбранной метрике.
    - target: dict {"color":(r,g,b), "grad":float}
    - tile_features: {path: {"color":..., "grad":...}}
    - metric: 'color' или 'color+grad'
    - grad_weight: относительный вес градиента (0..1)

    Если max_usage>0, сначала ищет среди доступных (usage < max_usage).
    Если среди доступных нет — вернёт лучший вообще (игнорируя лимит).
    """
    t_color = np.array(target.get("color", (0, 0, 0)), dtype=float)
    t_grad = float(target.get("grad", 0.0))

    best_tile = None
    best_score = float("inf")

    def score_for(path, feat):
        c = np.array(feat.get("color", (0, 0, 0)), dtype=float)
        # normalized color distance 0..1
        color_dist_sq = float(np.sum((t_color - c) ** 2))
        color_norm = color_dist_sq / (3 * (255.0 ** 2))
        if metric == 'color':
            return color_norm
        # color + grad
        g = float(feat.get("grad", 0.0))
        # normalize gradient difference
        grad_diff = abs(t_grad - g) / 255.0
        return color_norm + float(grad_weight) * (grad_diff ** 2)

    # first pass: honour max_usage
    for path, feat in tile_features.items():
        if max_usage and usage_count is not None and usage_count.get(path, 0) >= max_usage:
            continue
        s = score_for(path, feat)
        if s < best_score:
            best_score = s
            best_tile = path

    if best_tile is not None:
        return best_tile

    # fallback: ignore max usage
    for path, feat in tile_features.items():
        s = score_for(path, feat)
        if s < best_score:
            best_score = s
            best_tile = path

    return best_tile


def create_mosaic(
    input_path,
    tiles_path,
    output_path,
    grid_size,
    stride=None,
    allow_rotate=False,
    max_usage=0,
    color_correction_strength=0.0,
    metric='color',
    grad_weight=0.5,
    seam_smoothing=0.0,
    output_size=None,
    blend=0.0,
    progress_callback=None
):
    """
    Создаёт мозаику.
    - grid_size: размер тайла (в пикселях).
    - stride: шаг сетки (если None => равен grid_size).
    - allow_rotate: bool, разрешить случайные повороты (0,90,180,270).
    - max_usage: int, 0 = без ограничений.
    - color_correction_strength: 0..1 — сила коррекции среднего цвета тайла.
    - metric: 'color' или 'color+grad'
    - grad_weight: вес градиента в комбинированной метрике.
    - seam_smoothing: 0..1 — степень пост-усреднения для сглаживания швов.
    - blend: 0..1 — альфа-смешивание тайла с оригинальным блоком (tile * (1-blend) + block * blend).
    - output_size: (w,h) или None — если задано, исходник масштабируется до этого размера перед обработкой.
    - progress_callback(done, total): вызывается при каждом обработанном блоке.
    """

    # валидация
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input not found: {input_path}")
    if not os.path.isdir(tiles_path):
        raise FileNotFoundError(f"Tiles folder not found: {tiles_path}")

    stride = grid_size if stride is None else int(stride)

    # подготовка тайлов (считает средний цвет + градиенты и кеширует)
    tile_features = load_tile_features(tiles_path, grid_size)
    if not tile_features:
        raise RuntimeError("No valid tiles found in tiles folder.")

    # загрузка исходника и масштабирование (если нужно)
    input_img = Image.open(input_path).convert("RGB")
    if output_size is not None:
        input_img = input_img.resize((int(output_size[0]), int(output_size[1])), Image.Resampling.LANCZOS)

    w, h = input_img.size
    mosaic = Image.new("RGB", (w, h))

    blocks_positions = []
    for y in range(0, h, stride):
        for x in range(0, w, stride):
            blocks_positions.append((x, y))
    total_blocks = len(blocks_positions)
    done_blocks = 0

    usage_count = {}  # path -> count

    for (x, y) in blocks_positions:
        # блок берём размера grid_size (могут быть усечённые на краях)
        block = input_img.crop((x, y, min(x + grid_size, w), min(y + grid_size, h)))

        # для вычислений удобнее привести блок к grid_size
        block_for_feat = block.resize((grid_size, grid_size), Image.Resampling.LANCZOS)
        avg = average_color(block_for_feat)
        grad = compute_gradient_mean(block_for_feat)

        target = {"color": avg, "grad": grad}

        # найти лучший тайл с учетом лимита использования
        best_tile_path = find_best_tile_for_target(target, tile_features, usage_count=usage_count, max_usage=max_usage, metric=metric, grad_weight=grad_weight)

        if best_tile_path is None:
            # на крайний случай брать случайный
            best_tile_path = random.choice(list(tile_features.keys()))

        try:
            tile_img = Image.open(best_tile_path).convert("RGB")
        except Exception:
            tile_img = Image.new("RGB", (block.width, block.height), (0, 0, 0))

        # приводим тайл к размерам блока (учёт усечённых краёв)
        tile_img = tile_img.resize((block.width, block.height), Image.Resampling.LANCZOS)

        # поворот, если разрешён
        if allow_rotate:
            angle = random.choice([0, 90, 180, 270])
            if angle != 0:
                tile_img = tile_img.rotate(angle, expand=False)

        # простая цветовая коррекция (слабая подстройка среднего цвета)
        if color_correction_strength and color_correction_strength > 0.0:
            try:
                t_arr = np.array(tile_img).astype(np.int16)
                b_arr = np.array(block.resize((tile_img.width, tile_img.height))).astype(np.int16)
                # compute delta
                t_mean = t_arr[:, :, :3].reshape(-1, 3).mean(axis=0)
                b_mean = b_arr[:, :, :3].reshape(-1, 3).mean(axis=0)
                delta = (b_mean - t_mean) * float(color_correction_strength)
                # apply
                t_arr[:, :, :3] = np.clip(t_arr[:, :, :3] + delta, 0, 255)
                tile_img = Image.fromarray(t_arr.astype('uint8'), 'RGB')
            except Exception:
                pass  # на случай ошибок — не ломаем процесс

        # alpha-blend с оригинальным блоком (для мягкого встраивания деталей)
        if blend and blend > 0.0:
            try:
                resized_block = block.resize((tile_img.width, tile_img.height), Image.Resampling.LANCZOS)
                tile_img = Image.blend(tile_img, resized_block, float(blend))
            except Exception:
                pass

        # вставляем тайл
        mosaic.paste(tile_img, (x, y))

        # обновляем счётчик использования
        usage_count[best_tile_path] = usage_count.get(best_tile_path, 0) + 1

        # обновляем прогресс
        done_blocks += 1
        if progress_callback:
            try:
                progress_callback(done_blocks, total_blocks)
            except Exception:
                pass

    # постобработка: сглаживание швов (простая реализация — смешиваем с размытием)
    if seam_smoothing and seam_smoothing > 0.0:
        try:
            radius = max(1.0, float(seam_smoothing) * max(1.0, grid_size / 4.0))
            blurred = mosaic.filter(ImageFilter.GaussianBlur(radius=radius))
            mosaic = Image.blend(mosaic, blurred, float(seam_smoothing))
        except Exception:
            pass

    # сохраняем результат в подходящем формате
    ext = os.path.splitext(output_path)[1].lower()
    if ext in [".jpg", ".jpeg"]:
        mosaic.save(output_path, "JPEG", quality=90)
    elif ext == ".bmp":
        mosaic.save(output_path, "BMP")
    elif ext == ".webp":
        mosaic.save(output_path, "WEBP", quality=90)
    else:
        mosaic.save(output_path, "PNG")

    print(f"Mosaic saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--tiles", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--grid", type=int, default=30)
    parser.add_argument("--stride", type=int, default=None)
    parser.add_argument("--rotate", action="store_true")
    parser.add_argument("--max-usage", type=int, default=0)
    parser.add_argument("--color-correction", type=float, default=0.0, help="0..1 strength of tile color correction")
    parser.add_argument("--blend", type=float, default=0.0, help="0..1 alpha blend tile with source block")
    parser.add_argument("--metric", choices=["color", "color+grad"], default="color", help="matching metric")
    parser.add_argument("--grad-weight", type=float, default=0.5, help="relative weight for gradient when metric includes it")
    parser.add_argument("--seam-smoothing", type=float, default=0.0, help="0..1 seam smoothing strength")
    args = parser.parse_args()

    create_mosaic(
        args.input,
        args.tiles,
        args.output,
        args.grid,
        stride=args.stride,
        allow_rotate=args.rotate,
        max_usage=args.max_usage,
        color_correction_strength=args.color_correction,
        metric=args.metric,
        grad_weight=args.grad_weight,
        seam_smoothing=args.seam_smoothing,
        blend=args.blend
    )

