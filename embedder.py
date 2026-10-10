"""Literal WCAM embedder, Pramila et al. 2018 (Section 2, Fig. 2), falling
back to Pramila et al. 2012 only where 2018 explicitly defers to it.
"""
import json
import sys

import cv2
import numpy as np

from pattern import make_pattern
import codec
import imread
import jnd as jnd_module
import paths
import preprocess

BLOCK = 128
GRID = 4  # 4x4 = 16 blocks

DELTA1 = 100.0
DELTA2 = 10.0


def embed(image_path, out_path, meta_path, payload=None, seed=1234, delta1=None, delta2=None,
          use_preprocess=True, use_global_jnd=False, preprocess_mysize=3, pattern_rng_kind="PCG64",
          correct_jnd_kernels=True, jnd_beta_coef=jnd_module.BETA_COEF_CHOU_LI, jnd_floor=0.0,
          sync_angle=0.0, pattern_dc=0.0, pattern_balanced=True, pattern_noise=0.0,
          gamut_safe=False):
    delta1 = DELTA1 if delta1 is None else delta1
    delta2 = DELTA2 if delta2 is None else delta2
    if payload is None:
        payload = bytes(np.random.default_rng(seed).integers(0, 256, size=codec.PAYLOAD_BYTES, dtype=np.uint8))
    message_angles = codec.encode_message(payload)  # 15 thetas, one per non-sync block

    img = imread.require(image_path)
    img = cv2.resize(img, (BLOCK * GRID, BLOCK * GRID), interpolation=cv2.INTER_AREA)

    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb).astype(np.float64)
    Y = ycrcb[:, :, 0]

    X = preprocess.preprocess(Y, seed=seed, mysize=preprocess_mysize) if use_preprocess else Y.copy()

    jnd_full = jnd_module.compute(X, global_norm=use_global_jnd,
                                   correct_kernels=correct_jnd_kernels,
                                   beta_coef=jnd_beta_coef, floor=jnd_floor)

    if gamut_safe:
        cr, cb = ycrcb[:, :, 1] - 128.0, ycrcb[:, :, 2] - 128.0
        terms = np.stack([1.403 * cr, -0.714 * cr - 0.344 * cb, 1.773 * cb])
        gamut_hi = np.minimum(255.0 - terms.max(0), 255.0)
        gamut_lo = np.maximum(-terms.min(0), 0.0)

    idx = 0
    msg_idx = 0
    for gy in range(GRID):
        for gx in range(GRID):
            y0, x0 = gy * BLOCK, gx * BLOCK
            block = X[y0:y0 + BLOCK, x0:x0 + BLOCK]
            block_jnd = jnd_full[y0:y0 + BLOCK, x0:x0 + BLOCK]

            if idx == 0:
                theta = sync_angle
            else:
                theta = (message_angles[msg_idx] + sync_angle) % 180.0
                msg_idx += 1

            pattern = make_pattern(BLOCK, theta, seed=seed, rng_kind=pattern_rng_kind,
                                    dc=pattern_dc, balanced=pattern_balanced,
                                    noise=pattern_noise)
            strength = delta1 * block_jnd + delta2 * (1.0 - block_jnd)
            if gamut_safe:
                signal = pattern * strength
                offset = pattern.mean() * strength
                swing = signal - offset
                hi = gamut_hi[y0:y0 + BLOCK, x0:x0 + BLOCK] - block
                lo = gamut_lo[y0:y0 + BLOCK, x0:x0 + BLOCK] - block
                scale = np.ones_like(swing)
                over = swing > hi
                scale[over] = hi[over] / swing[over]
                under = swing < lo
                scale[under] = lo[under] / swing[under]
                swing = swing * np.clip(scale, 0.0, 1.0)
                signal = swing + np.clip(offset, lo - swing, hi - swing)
            else:
                signal = pattern * strength
            X[y0:y0 + BLOCK, x0:x0 + BLOCK] = block + signal

            idx += 1

    Y = np.clip(X, 0, 255)
    ycrcb[:, :, 0] = Y
    out = cv2.cvtColor(ycrcb.astype(np.uint8), cv2.COLOR_YCrCb2BGR)
    out_path = paths.resolve(out_path)
    cv2.imwrite(out_path, out)

    with open(paths.resolve(meta_path), "w") as f:
        json.dump({"grid": GRID, "block": BLOCK, "seed": seed,
                    "sync_index": 0, "payload_hex": payload.hex(),
                    "angles": message_angles, "delta1": delta1, "delta2": delta2,
                    "use_global_jnd": use_global_jnd,
                    "preprocess_mysize": preprocess_mysize if use_preprocess else None,
                    "correct_jnd_kernels": correct_jnd_kernels,
                    "jnd_beta_coef": jnd_beta_coef,
                    "jnd_floor": jnd_floor,
                    "sync_angle": sync_angle,
                    "pattern_dc": pattern_dc,
                    "pattern_balanced": pattern_balanced,
                    "pattern_noise": pattern_noise,
                    "gamut_safe": gamut_safe}, f, indent=2)

    print(f"wrote {out_path}")
    print(f"wrote {meta_path}: payload={payload.hex()}, angles={[round(a, 2) for a in message_angles]}")
    return out


MARGIN = 80
FRAME_THICKNESS = 6


FRAME_COLOR_BLUE = (255, 0, 0)   # BGR
FRAME_COLOR_BLACK = (0, 0, 0)


def add_frame(core_img, frame_path, color=FRAME_COLOR_BLUE):
    """Wrap a core in the printed frame and write it."""
    if isinstance(core_img, str):
        loaded = cv2.imread(paths.resolve(core_img), cv2.IMREAD_COLOR)
        if loaded is None:
            raise SystemExit(f"не прочитать ядро: {core_img}")
        core_img = loaded
    core_size = core_img.shape[0]
    canvas_size = core_size + 2 * MARGIN
    canvas = np.full((canvas_size, canvas_size, 3), 255, dtype=np.uint8)

    fx0 = MARGIN - FRAME_THICKNESS
    fy0 = MARGIN - FRAME_THICKNESS
    fx1 = MARGIN + core_size + FRAME_THICKNESS
    fy1 = MARGIN + core_size + FRAME_THICKNESS
    cv2.rectangle(canvas, (fx0, fy0), (fx1, fy1), color, FRAME_THICKNESS)

    canvas[MARGIN:MARGIN + core_size, MARGIN:MARGIN + core_size] = core_img
    frame_path = paths.resolve(frame_path)
    cv2.imwrite(frame_path, canvas)
    print(f"wrote {frame_path} (core={core_size}, margin={MARGIN}) -- display/print this one")


if __name__ == "__main__":
    src = sys.argv[1] if len(sys.argv) > 1 else "sample_images/peppers.png"
    core = embed(src, "watermarked.png", "meta.json")
    add_frame(core, "for_display.png")
