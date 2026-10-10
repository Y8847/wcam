"""Достать кадры из stack.bin отдельными картинками.

Приложение хранит фокус-стек одним двоичным файлом: заголовок из четырёх
32-битных целых (число кадров, ширина, высота, множитель яркости), затем
подряд кадры по ширина x высота беззнаковых 16-битных целых, построчно.
Значение пикселя -- яркость, умноженная на множитель; канал один.

Здесь он разбирается обратно в обычные PNG, по файлу на кадр.
"""
import os
import re
import struct
import sys

import cv2
import numpy as np

import paths

USAGE = """Разобрать фокус-стек на отдельные кадры.

  python dump_stack.py <каталог съёмки или stack.bin> [куда положить]

Каталог съёмки -- папка из архива приложения, внутри stack.bin и notes.txt.
Если второе имя не указано, кадры лягут в out/<имя съёмки>/.

Пример:

  python dump_stack.py captures/2026-09-13T03-22-59Z
"""


def read_stack(path):
    """(кадры, множитель) из stack.bin; кадры -- массив (N, высота, ширина)."""
    with open(path, 'rb') as handle:
        blob = handle.read()
    if len(blob) < 16:
        raise SystemExit(f"{path} слишком мал для заголовка")
    count, width, height, scale = struct.unpack('<4i', blob[:16])
    expected = 16 + count * width * height * 2
    if not (0 < count < 10000 and 0 < width < 20000 and 0 < height < 20000):
        raise SystemExit(f"{path}: заголовок не похож на стек "
                         f"({count} кадров {width}x{height})")
    if len(blob) != expected:
        raise SystemExit(f"{path}: ожидалось {expected} байт, а в файле {len(blob)}")
    frames = np.frombuffer(blob[16:], dtype='<u2').reshape(count, height, width)
    return frames, (scale or 1)


def read_planes(notes_path):
    """Положение линзы и дистанция для каждого кадра, если notes.txt рядом."""
    if not os.path.exists(notes_path):
        return {}
    text = open(notes_path, encoding='utf-8').read()
    out = {}
    for index, lens, distance in re.findall(
            r'^frame (\d+) lens ([0-9.]+) distance ([0-9.]+)', text, re.M):
        out[int(index)] = (float(lens), float(distance))
    return out


def dump(target, destination=None):
    if os.path.isdir(target):
        stack_path = os.path.join(target, 'stack.bin')
        notes_path = os.path.join(target, 'notes.txt')
        name = os.path.basename(os.path.normpath(target))
    else:
        stack_path = target
        notes_path = os.path.join(os.path.dirname(target), 'notes.txt')
        name = os.path.splitext(os.path.basename(target))[0]
    if not os.path.exists(stack_path):
        raise SystemExit(f"нет файла {stack_path}")

    frames, scale = read_stack(stack_path)
    planes = read_planes(notes_path)
    destination = destination or paths.resolve(name)
    os.makedirs(destination, exist_ok=True)

    print(f"{len(frames)} кадров {frames.shape[2]}x{frames.shape[1]}, "
          f"множитель яркости {scale}")
    print(f"{'кадр':>5} {'линза':>8} {'дистанция':>11}   файл")
    for index, plane in enumerate(frames):
        image = np.clip(plane.astype(np.float64) / scale, 0, 255).astype(np.uint8)
        out_path = os.path.join(destination, f'frame_{index:02d}.png')
        cv2.imwrite(out_path, image)
        lens, distance = planes.get(index, (None, None))
        print(f"{index:5d} {lens if lens is not None else '—':>8} "
              f"{f'{distance * 100:.1f} см' if distance else '—':>11}   {out_path}")
    return destination


if __name__ == '__main__':
    if len(sys.argv) < 2 or sys.argv[1] in ('-h', '--help'):
        print(USAGE)
        raise SystemExit(0 if len(sys.argv) > 1 else 2)
    dump(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
