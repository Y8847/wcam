"""End-to-end pipeline for a real printed+photographed test image (Phase D):"""
import json
import paths
import os
import sys

import cv2
import numpy as np
import imread
from skimage.metrics import structural_similarity as ssim

import rectify
import extractor
import codec

DEFAULT_COVER = "sample_images/peppers.png"


def _extract_core(core, truth_payload=None, truth_angles=None):
    """Чтение блоков и декодирование по уже выпрямленному ядру.

    Возвращает (есть ли синхроблок, сколько уровней совпало, нагрузка).
    Совпадение уровней считается только если переданы эталонные углы:
    само чтение эталона не требует, в том и смысл метода.
    """
    ycrcb = cv2.cvtColor(core, cv2.COLOR_BGR2YCrCb).astype(np.float64)
    Y = extractor.descreen(ycrcb[:, :, 0])

    raw_angles = []
    sync_present = False
    idx = 0
    for gy in range(extractor.GRID):
        for gx in range(extractor.GRID):
            y0, x0 = gy * extractor.BLOCK, gx * extractor.BLOCK
            block = Y[y0:y0 + extractor.BLOCK, x0:x0 + extractor.BLOCK]
            measured, _dbg, present = extractor.block_angle(block)
            raw_angles.append(measured)
            if idx == 0:
                sync_present = present
            idx += 1

    sync_measured = raw_angles[0] if raw_angles[0] is not None else 0.0
    corrected = [extractor.correct_for_sync(m, sync_measured) for m in raw_angles[1:]]
    levels_correct = None if truth_angles is None else sum(
        1 for c, t in zip(corrected, truth_angles)
        if codec.angle_to_level(c) == codec.angle_to_level(t))
    checked = codec.decode_verified(corrected)
    decoded_payload = checked[0] if checked else None
    return sync_present, levels_correct, decoded_payload


def process(photo_path, meta_path="meta.json", wm_path=None, cover_path=DEFAULT_COVER):
    core = rectify.rectify(photo_path, out_path="_photo_rectified.png",
                            debug_path="_photo_rectify_debug.png")

    truth_payload = truth_angles = None
    if meta_path is not None:
        with open(paths.resolve(meta_path)) as f:
            meta = json.load(f)
        truth_payload = bytes.fromhex(meta["payload_hex"])
        truth_angles = meta["angles"]

    print(f"frame found: yes (see _photo_rectify_debug.png to sanity-check alignment)")

    best = None  # (rotation_deg, sync_present, levels_correct, decoded_payload)
    for k in range(4):
        rot = np.rot90(core, k)
        sync_present, levels_correct, decoded_payload = _extract_core(rot, truth_payload, truth_angles)
        if decoded_payload is not None:
            best = (k * 90, sync_present, levels_correct, decoded_payload)
            break
        if best is None or (levels_correct if levels_correct is not None else -1) > (
            best[2] if best[2] is not None else -1):
            best = (k * 90, sync_present, levels_correct, decoded_payload)

    rotation_deg, sync_present, levels_correct, decoded_payload = best
    print(f"watermark detected (sync block, Eq. 19): {sync_present} (best rotation tried: {rotation_deg}deg)")

    print(f"decoded payload: {decoded_payload.hex() if decoded_payload else None}")
    decoded_ok = None
    if truth_angles is not None:
        decoded_ok = decoded_payload == truth_payload
        print(f"angle levels correct (pre-ECC): {levels_correct}/{len(truth_angles)} "
              f"({100.0 * levels_correct / len(truth_angles):.1f}%)")
        print(f"BER (pre-ECC): {100.0 * (len(truth_angles) - levels_correct) / len(truth_angles):.1f}%")
        print(f"truth payload:   {truth_payload.hex()}")
        print(f"message recovered correctly: {decoded_ok}")
    elif decoded_payload is None:
        print("сообщение не собралось: ошибок больше, чем исправляет RS(15,8)")
    else:
        print("(метаданные не переданы -- сверить не с чем; "
              "нагрузка прошла проверку пересборкой)")

    core_oriented = np.rot90(core, rotation_deg // 90)
    core_y = cv2.cvtColor(core_oriented, cv2.COLOR_BGR2YCrCb)[:, :, 0]
    results = {"decoded_ok": decoded_ok, "levels_correct": levels_correct, "rotation_deg": rotation_deg}

    cover = imread.read(cover_path)
    if cover is not None:
        cover = cv2.resize(cover, (rectify.CORE_SIZE, rectify.CORE_SIZE), interpolation=cv2.INTER_AREA)
        cover_y = cv2.cvtColor(cover, cv2.COLOR_BGR2YCrCb)[:, :, 0]
        mssim_vs_cover = ssim(cover_y, core_y, data_range=255)
        results["mssim_vs_cover"] = mssim_vs_cover
        print(f"\nMSSIM vs original cover (overall photo-vs-source, NOT comparable to "
              f"paper's Table 1): {mssim_vs_cover:.3f}")
    else:
        print(f"\n(could not read cover image {cover_path} -- skipping cover MSSIM)")

    if wm_path is not None:
        wm = imread.read(wm_path)
        if wm is None:
            print(f"(could not read watermarked-digital image {wm_path} -- skipping paper-exact MSSIM)")
        else:
            wm = cv2.resize(wm, (rectify.CORE_SIZE, rectify.CORE_SIZE), interpolation=cv2.INTER_AREA)
            wm_y = cv2.cvtColor(wm, cv2.COLOR_BGR2YCrCb)[:, :, 0]
            mssim_vs_watermarked = ssim(wm_y, core_y, data_range=255)
            results["mssim_vs_watermarked"] = mssim_vs_watermarked
            print(f"MSSIM vs digital watermarked file (paper's Table 1 methodology -- "
                  f"print/scan/descreen channel fidelity): {mssim_vs_watermarked:.3f}")
    else:
        print("(no wm_path given -- skipping paper-exact MSSIM; pass the digital "
              "watermarked core, e.g. watermarked.png, to compute it)")

    print("(caveat: both numbers also reflect residual rectification/color/lighting "
          "differences, not purely watermark visibility or purely channel fidelity)")

    return results


USAGE = """Прочитать сообщение с одной фотографии отпечатка.

  python process_photo.py <фотография> [метаданные.json] [ядро.png] [обложка.png]

Метаданные необязательны: сообщение читается из самой фотографии, эталон
нужен только чтобы сверить ответ и посчитать долю ошибочных блоков. Если
второй аргумент не указан, файл ищется сам -- рядом с фотографией по её имени
с расширением .json, затем такой же в каталоге out/. Не нашёлся -- чтение всё
равно состоится, а прочитанная нагрузка проверяется пересборкой сообщения.

Третий и четвёртый аргументы необязательны и нужны только для оценки качества:
ядро без рамки и исходная обложка без знака.

Пример:

  python embed_cli.py sample_images/peppers.png --payload deadbeef
  python process_photo.py out/peppers_wm.png
"""


def find_meta(photo_path):
    """Метаданные рядом с фотографией, иначе одноимённые в out/."""
    stem = os.path.splitext(photo_path)[0]
    for candidate in (stem + '.json', paths.resolve(
            os.path.basename(stem) + '.json', create=False)):
        if os.path.exists(candidate):
            return candidate
    return None


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] in ('-h', '--help'):
        print(USAGE)
        raise SystemExit(0 if len(sys.argv) > 1 else 2)
    photo = sys.argv[1]
    if not os.path.exists(photo):
        raise SystemExit(f"нет такого файла: {photo}")
    meta = sys.argv[2] if len(sys.argv) > 2 else find_meta(photo)
    if meta is not None and not os.path.exists(paths.resolve(meta)):
        raise SystemExit(f"нет такого файла метаданных: {meta}")
    if meta is None:
        print("метаданные не найдены -- читаю без сверки "
              f"(искал {os.path.splitext(photo)[0]}.json и такой же в out/)")
    wm = sys.argv[3] if len(sys.argv) > 3 else None
    cover = sys.argv[4] if len(sys.argv) > 4 else DEFAULT_COVER
    process(photo, meta, wm, cover)
