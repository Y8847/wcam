"""Чтение изображений, включая HEIC.

OpenCV формат HEIC не поддерживает вовсе, а именно в нём iPhone снимает по
умолчанию. `read` сперва пробует cv2, а для .heic и .heif обращается к
pillow-heif, если он установлен, либо к внешнему heif-dec из libheif.
"""
import os
import shutil
import subprocess
import tempfile

import cv2

HEIF_SUFFIXES = ('.heic', '.heif')
CONVERTERS = ('heif-dec', 'heif-convert')


def _via_pillow(path, flags):
    try:
        import numpy as np
        import pillow_heif
    except ImportError:
        return None
    heif = pillow_heif.read_heif(path)
    import numpy as np
    data = np.asarray(heif.to_pillow().convert('RGB'))
    img = cv2.cvtColor(data, cv2.COLOR_RGB2BGR)
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if flags == cv2.IMREAD_GRAYSCALE else img


def _via_tool(path, flags):
    tool = next((t for t in CONVERTERS if shutil.which(t)), None)
    if tool is None:
        return None
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, 'frame.png')
        done = subprocess.run([tool, path, out], capture_output=True)
        if done.returncode != 0 or not os.path.exists(out):
            return None
        return cv2.imread(out, flags)


def read(path, flags=cv2.IMREAD_COLOR):
    """Изображение по пути `path`, или None если прочитать не удалось."""
    img = cv2.imread(path, flags)
    if img is not None:
        return img
    if not os.path.exists(path):
        return None
    if os.path.splitext(path)[1].lower() not in HEIF_SUFFIXES:
        return None
    return _via_pillow(path, flags) or _via_tool(path, flags)


def require(path, flags=cv2.IMREAD_COLOR):
    """То же, но с внятной ошибкой вместо None."""
    img = read(path, flags)
    if img is not None:
        return img
    if not os.path.exists(path):
        raise SystemExit(f"нет такого файла: {path}")
    if os.path.splitext(path)[1].lower() in HEIF_SUFFIXES:
        raise SystemExit(
            f"не прочитать {path}: OpenCV не умеет HEIC.\n"
            "Поставьте pillow-heif (pip install pillow-heif) либо libheif,\n"
            "которая даёт heif-dec, либо переведите снимок в PNG или JPEG.")
    raise SystemExit(f"не прочитать {path}: формат не поддерживается или файл повреждён")
