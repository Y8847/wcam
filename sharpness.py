"""Pramila et al. 2018, Section 3.2 (Sharpness map acquisition)."""
import cv2
import numpy as np

MAP_W, MAP_H = 96, 72


def _sharpness_map_raw(scaled_gray_or_bgr):
    """Eq. 3's block-averaged abs-Laplacian, NOT yet normalized."""
    if scaled_gray_or_bgr.ndim == 3:
        gray = cv2.cvtColor(scaled_gray_or_bgr, cv2.COLOR_BGR2GRAY)
    else:
        gray = scaled_gray_or_bgr
    L = np.abs(cv2.Laplacian(gray.astype(np.float64), cv2.CV_64F, ksize=3))
    return cv2.resize(L, (MAP_W, MAP_H), interpolation=cv2.INTER_AREA)


def sharpness_map(scaled_gray_or_bgr):
    """Eq. 3 for a SINGLE scaled (816x612) image, normalized to its own
    [0,1] range. Kept for standalone/one-off use -- NOT what
    `sharpness_maps` uses for a real stack (see module docstring: a
    single-image-only normalization can't be compared across a stack).
    """
    small = _sharpness_map_raw(scaled_gray_or_bgr)
    lo, hi = small.min(), small.max()
    if hi - lo < 1e-9:
        return np.zeros_like(small)
    return (small - lo) / (hi - lo)


def sharpness_maps(scaled_frames):
    """Eq. 3 for the whole stack, normalized with a single min/max shared
    across all N frames (see module docstring) so that Si(u,v) values
    stay comparable across the stack index i, as Section 3.3's selection
    algorithm requires. Returns array shape (N, 72, 96).
    """
    raws = np.stack([_sharpness_map_raw(f) for f in scaled_frames])
    lo, hi = raws.min(), raws.max()
    if hi - lo < 1e-9:
        return np.zeros_like(raws)
    return (raws - lo) / (hi - lo)


if __name__ == "__main__":
    import sys
    import capture

    path = sys.argv[1] if len(sys.argv) > 1 else "for_display_baboon_100.png"
    angle = float(sys.argv[2]) if len(sys.argv) > 2 else 45.0

    canvas = cv2.imread(path, cv2.IMREAD_COLOR)
    if canvas is None:
        raise SystemExit(f"could not read {path}")
    full_frames, scaled_frames, dioptres = capture.build_focal_stack(canvas, angle, seed=0)
    maps = sharpness_maps(scaled_frames)
    print(f"sharpness maps shape: {maps.shape}")
    for i in [0, 3, 6, 9]:
        m = maps[i]
        left_mean = m[:, :MAP_W // 2].mean()
        right_mean = m[:, MAP_W // 2:].mean()
        print(f"  frame {i} (focus={dioptres[i]:.2f}D): left_mean={left_mean:.3f} right_mean={right_mean:.3f}")
        cv2.imwrite(f"_debug_smap_{i}.png", (m * 255).astype(np.uint8))
    print("wrote _debug_smap_0/3/6/9.png")
