"""Synthetic print-cam degradation pipeline (Phase B)."""
import cv2
import numpy as np


def simulate_tilt(img, angle_deg, axis="yaw", distance=2.6, pad_frac=0.5):
    """Warp `img` as if photographed at `angle_deg` off fronto-parallel."""
    h, w = img.shape[:2]
    pad = int(round(max(h, w) * pad_frac))
    canvas_h, canvas_w = h + 2 * pad, w + 2 * pad
    fill = 255 if img.ndim == 2 else (255,) * img.shape[2]

    theta = np.radians(angle_deg)
    half_w, half_h = 1.0, 1.0
    corners = np.array([
        [-half_w, -half_h, 0.0],
        [half_w, -half_h, 0.0],
        [half_w, half_h, 0.0],
        [-half_w, half_h, 0.0],
    ])

    if axis == "yaw":
        Xr = corners[:, 0] * np.cos(theta)
        Zr = -corners[:, 0] * np.sin(theta)
        Yr = corners[:, 1]
    elif axis == "pitch":
        Yr = corners[:, 1] * np.cos(theta)
        Zr = corners[:, 1] * np.sin(theta)
        Xr = corners[:, 0]
    else:
        raise ValueError(f"unknown axis {axis!r}")

    f = distance
    proj_x = f * Xr / (distance + Zr)
    proj_y = f * Yr / (distance + Zr)

    # normalized [-1,1] projected corners -> pixel coords of the *original*
    # (unpadded) rectangle's footprint within the padded canvas
    dst = np.stack([
        (proj_x * 0.5 + 0.5) * w + pad,
        (proj_y * 0.5 + 0.5) * h + pad,
    ], axis=1).astype(np.float32)

    src = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(src, dst)

    warped = cv2.warpPerspective(img, M, (canvas_w, canvas_h),
                                  flags=cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_CONSTANT, borderValue=fill)
    return warped


DEFOCUS_K = 0.25

#: What to use when modelling a single shot with no focal stack behind it.
DEFOCUS_K_NO_STACK = 3.5


def apply_degradation(img, angle_deg, axis="yaw", distance=3.4,
                       halftone_blur_sigma=0.6, defocus_k=DEFOCUS_K, noise_std=4.5,
                       jpeg_quality=85, brightness=0.0, contrast=1.0,
                       illumination=0.15, seed=None):
    x = img.astype(np.float64)

    if illumination > 0:
        ramp = np.linspace(1 - illumination / 2, 1 + illumination / 2, x.shape[1])
        x = x * (ramp[None, :, None] if x.ndim == 3 else ramp[None, :])
        # Clipped here and not later: the tilt step round-trips through uint8,
        # where anything the ramp pushed past 255 would wrap to black.
        x = np.clip(x, 0, 255)

    total_sigma = halftone_blur_sigma + defocus_k * np.sin(np.radians(angle_deg)) ** 2
    if total_sigma > 0:
        x = cv2.GaussianBlur(x, (0, 0), total_sigma)

    x = simulate_tilt(x.astype(np.uint8).astype(np.float64), angle_deg, axis=axis, distance=distance)

    if noise_std > 0:
        rng = np.random.default_rng(seed)
        x = x + rng.normal(0, noise_std, x.shape)

    x = x * contrast + brightness
    x = np.clip(x, 0, 255).astype(np.uint8)

    if jpeg_quality is not None:
        ok, enc = cv2.imencode(".jpg", x, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
        color_flag = cv2.IMREAD_COLOR if x.ndim == 3 else cv2.IMREAD_GRAYSCALE
        x = cv2.imdecode(enc, color_flag)

    return x


if __name__ == "__main__":
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else "for_display.png"
    angle = float(sys.argv[2]) if len(sys.argv) > 2 else 30.0

    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise SystemExit(f"could not read {path}")

    # sanity: angle=0 must be the identity warp (no perspective change)
    identity = simulate_tilt(img, 0.0)
    core = identity[identity.shape[0] // 2 - 5:identity.shape[0] // 2 + 5,
                     identity.shape[1] // 2 - 5:identity.shape[1] // 2 + 5]
    print(f"identity-warp sanity check: output shape includes padding "
          f"({identity.shape} vs input {img.shape}); center patch mean={core.mean():.1f}")

    out = apply_degradation(img, angle)
    cv2.imwrite("debug_degraded.png", out)
    print(f"wrote debug_degraded.png (angle={angle} deg)")
