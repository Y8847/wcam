"""Run a focal stack captured by the phone through the Python pipeline."""
import os
import re
import struct
import sys

import cv2
import numpy as np

import blend
import codec
import extractor
import rectify
import reconfigure
import register
import sharpness
import substack


def load(directory):
    """(frames, dioptres, notes) for one archived capture."""
    with open(os.path.join(directory, 'stack.bin'), 'rb') as handle:
        blob = handle.read()
    count, width, height, scale = struct.unpack('<4i', blob[:16])
    planes = np.frombuffer(blob[16:], dtype='<u2').reshape(count, height, width)
    frames = [(p.astype(np.float64) / scale).astype(np.uint8) for p in planes]
    notes = open(os.path.join(directory, 'notes.txt'), encoding='utf-8').read()
    distances = [float(m) for m in
                 re.findall(r'^frame \d+ lens [0-9.]+ distance ([0-9.]+)', notes, re.M)]
    dioptres = [1.0 / d if d > 0 else 0.0 for d in distances] or [0.0] * count
    return frames, dioptres, notes


def run(directory, denom=2):
    frames, dioptres, notes = load(directory)
    print(f"{len(frames)} кадров {frames[0].shape[1]}x{frames[0].shape[0]}")

    maps = sharpness.sharpness_maps(frames)
    selected, foreground, t, valuable, occurrence = substack.select_substack(maps)
    print(f"подстек отобрал {selected} (порог t={t:.3f}, полезных {valuable})")

    saved = reconfigure.SCALE_DENOM
    reconfigure.SCALE_DENOM = denom
    try:
        descreened, feats, _ = reconfigure.reconfigure_selected(frames, dioptres, selected)
    finally:
        reconfigure.SCALE_DENOM = saved

    feats = [cv2.cvtColor(np.clip(f, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
             if f.ndim == 2 else f for f in feats]

    matrices = register.register_stack(feats)
    placed = sum(1 for m in matrices if m is not None)
    print(f"совмещено {placed} из {len(feats)}")
    registered = blend.warp_registered(feats, matrices)
    if not registered:
        return print("совмещать нечего")
    composite = blend.blend_stack(registered)

    scale = composite.shape[0] / frames[0].shape[0]
    if composite.ndim == 2:                      # find_frame_quad wants colour
        composite = cv2.cvtColor(np.clip(composite, 0, 255).astype(np.uint8),
                                 cv2.COLOR_GRAY2BGR)
    quad = rectify.find_frame_quad(
        composite,
        window=max(round(rectify.THRESH_WINDOW * scale) | 1, 3),
        close_size=max(round(rectify.CLOSE_ELEMENT_SIZE * scale), 1))
    if quad is None:
        return print("рамка не найдена в склейке")
    print(f"рамка найдена, склейка {composite.shape[1]}x{composite.shape[0]}")

    size = rectify.CORE_SIZE + 2 * rectify.MARGIN
    o = rectify.FRAME_OUTER_INSET
    dst = np.array([[o, o], [size - 1 - o, o],
                    [size - 1 - o, size - 1 - o], [o, size - 1 - o]], dtype=np.float32)
    warped = cv2.warpPerspective(descreened[0],
                                 cv2.getPerspectiveTransform(quad * denom, dst), (size, size))
    core = warped[rectify.MARGIN:rectify.MARGIN + rectify.CORE_SIZE,
                  rectify.MARGIN:rectify.MARGIN + rectify.CORE_SIZE]

    for rotation in range(4):
        angles = []
        turned = np.rot90(core, rotation).copy()
        for gy in range(extractor.GRID):
            for gx in range(extractor.GRID):
                block = turned[gy * extractor.BLOCK:(gy + 1) * extractor.BLOCK,
                               gx * extractor.BLOCK:(gx + 1) * extractor.BLOCK]
                measured, _, _ = extractor.block_angle(block.astype(np.float64))
                angles.append(measured)
        sync = angles[0] if angles[0] is not None else 0.0
        corrected = [extractor.correct_for_sync(a, sync) for a in angles[1:]]
        checked = codec.decode_verified(corrected)
        if checked:
            payload, disagreements = checked
            return print(f"декодировано при повороте {rotation * 90}°: {payload.hex()}"
                         f" (расхождений {disagreements})")
        loose = codec.decode_message(corrected)
        if loose:
            print(f"  поворот {rotation * 90}°: RS выдал {loose.hex()},"
                  f" но проверка пересборкой его отвергла")
    print("сообщение не собралось ни в одном из четырёх поворотов")


USAGE = """Прогнать фокус-стек через конвейер разделов 3.1-3.8.

  python _run_real_stack.py <каталог съёмки>

Каталог съёмки -- папка, внутри которой лежат stack.bin (кадры) и
notes.txt (параметры съёмки). Такие каталоги записывало наше приложение
для iPhone; в этот репозиторий оно не входит, формат stack.bin описан в
README. Пример съёмки лежит в корне репозитория:

  python _run_real_stack.py .
"""


if __name__ == '__main__':
    if len(sys.argv) != 2 or sys.argv[1] in ('-h', '--help'):
        print(USAGE)
        raise SystemExit(0 if len(sys.argv) > 1 else 2)
    target = sys.argv[1]
    if not os.path.isdir(target):
        raise SystemExit(f"нет такого каталога: {target}")
    if not os.path.exists(os.path.join(target, 'stack.bin')):
        raise SystemExit(f"в {target} нет stack.bin -- это точно каталог съёмки?")
    run(target)
