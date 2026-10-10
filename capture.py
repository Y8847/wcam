"""Pramila et al. 2018, Section 3.1 (Capturing process)."""
import cv2
import numpy as np

import degrade

N_FRAMES = 10
FSTEP = 1.0 / 100.0
F_ORIGINAL = 1.0 / 0.15  # dioptres; ~15cm focus distance, matching the papers' own ~12.5-24cm test range

# Real half-megapixel target the paper gives explicitly (816x612 = 499,392px).
SCALED_W, SCALED_H = 816, 612

FOCUS_DIOPTER_TO_SIGMA = 50.0


def focus_points(f_original=F_ORIGINAL, n=N_FRAMES, fstep=FSTEP):
    """Literal Eq. 2: the N dioptre values the (real) autofocus-stepped
    capture would use, i=1..n.
    """
    points = []
    for i in range(1, n + 1):
        denom = (1.0 / f_original) - (3.0 * fstep) + (fstep * (i - 1))
        points.append(1.0 / denom)
    return points


def _depth_at_u(u, theta):
    """Same depth model as degrade.simulate_tilt/wcam's focal_stack.py:
    for a page tilted by theta about the yaw axis, a point at normalized
    horizontal position u in [-1,1] sits at relative depth -u*sin(theta).
    """
    return -u * np.sin(theta)


PAGE_HALF_WIDTH_CM = 13.5 / 2.0  # papers' own printed size, 13.5x13.5cm

SIGMA_BASE = 50.0     # safe value at range->0 (matches the original fixed default)
SIGMA_MAX = 100.0     # value at range >= SIGMA_REF_RANGE
SIGMA_REF_RANGE = 6.0  # dioptres; roughly this synthetic setup's own 60deg range


def _scene_dioptre_range(angle_deg, f_original=F_ORIGINAL, axis="yaw"):
    """The scene's actual real dioptre spread across the tilted page at
    this angle (u=-1 to u=+1), computed directly from the same physical
    model as synth_defocus_frame -- not estimated, exact for this
    synthetic setup.
    """
    theta = np.radians(angle_deg)
    dist_cm = 100.0 / f_original
    d1 = _depth_at_u(-1.0, theta) * PAGE_HALF_WIDTH_CM
    d2 = _depth_at_u(1.0, theta) * PAGE_HALF_WIDTH_CM
    dist1 = max(dist_cm + d1, 1.0)
    dist2 = max(dist_cm + d2, 1.0)
    return abs(100.0 / dist1 - 100.0 / dist2)


def adaptive_focus_sigma_coef(angle_deg, f_original=F_ORIGINAL):
    """SIGMA_BASE at angle->0, ramping linearly to SIGMA_MAX as the
    scene's real dioptre range approaches SIGMA_REF_RANGE.
    """
    rng = _scene_dioptre_range(angle_deg, f_original=f_original)
    frac = min(rng / SIGMA_REF_RANGE, 1.0)
    return SIGMA_BASE + (SIGMA_MAX - SIGMA_BASE) * frac


def synth_defocus_frame(core_img, angle_deg, focus_dioptre, f_original=F_ORIGINAL,
                         axis="yaw", base_sigma=0.4, strip=16, sigma_coef=None):
    """Render core_img as it would look captured with the lens focused at
    `focus_dioptre`, given the page is tilted by angle_deg (so different
    horizontal strips sit at different simulated depths/dioptres, same
    space-variant-blur idea as wcam/focal_stack.py's synth_focus_frame,
    but the depth-to-dioptre conversion is now physically grounded (real
    cm, via the papers' own 13.5x13.5cm print size and the viewing
    distance implied by f_original) instead of an arbitrary scale factor
    -- the arbitrary version put the scene's actual depth range far
    outside the Eq.2 focus points' range, so the whole stack just blurred
    uniformly instead of the sharp region genuinely shifting across
    frames (found by checking left-vs-right Laplacian sharpness across
    the stack: ratio should invert from >1 to <1, it instead collapsed
    to near-zero absolute sharpness everywhere by frame 9).
    """
    h, w = core_img.shape[:2]
    theta = np.radians(angle_deg)
    dist_cm = 100.0 / f_original  # f_original is in dioptres -> distance in m -> cm
    if sigma_coef is None:
        sigma_coef = adaptive_focus_sigma_coef(angle_deg, f_original=f_original)
    out = core_img.astype(np.float64).copy()
    for x0 in range(0, w, strip):
        x1 = min(x0 + strip, w)
        xc = (x0 + x1) / 2.0
        u = (xc / (w - 1)) * 2 - 1
        depth_cm = _depth_at_u(u, theta) * PAGE_HALF_WIDTH_CM
        point_dist_cm = max(dist_cm + depth_cm, 1.0)  # avoid non-physical <=0 distance
        depth_dioptre = 100.0 / point_dist_cm
        sigma = base_sigma + sigma_coef * abs(1.0 / depth_dioptre - 1.0 / focus_dioptre)
        if sigma > 0.05:
            pad = int(np.ceil(3 * sigma)) + 1
            xa, xb = max(0, x0 - pad), min(w, x1 + pad)
            region = out[:, xa:xb]
            blurred = cv2.GaussianBlur(region, (0, 0), sigma)
            out[:, x0:x1] = blurred[:, x0 - xa:x1 - xa]
    return out


def build_focal_stack(core_img, angle_deg, axis="yaw", noise_std=3.0, jpeg_quality=85, seed=None):
    """Simulate the N=10-frame captured focal stack (Section 3.1): for
    each Eq.2 focus dioptre, defocus + tilt + noise + JPEG. Returns two
    parallel lists -- full-resolution frames (kept "for later processing",
    i.e. the real 3.8 extraction) and half-megapixel-scaled frames (used
    for the fast 3.2-3.5 pipeline) -- plus the focus dioptre list.
    """
    rng = np.random.default_rng(seed)
    dioptres = focus_points()
    full_frames = []
    scaled_frames = []
    for fd in dioptres:
        defocused = synth_defocus_frame(core_img, angle_deg, fd, axis=axis)
        tilted = degrade.simulate_tilt(defocused.astype(np.uint8).astype(np.float64), angle_deg, axis=axis)
        x = tilted
        if noise_std > 0:
            x = x + rng.normal(0, noise_std, x.shape)
        x = np.clip(x, 0, 255).astype(np.uint8)
        if jpeg_quality is not None:
            ok, enc = cv2.imencode(".jpg", x, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
            x = cv2.imdecode(enc, cv2.IMREAD_COLOR)
        full_frames.append(x)
        scaled_frames.append(cv2.resize(x, (SCALED_W, SCALED_H), interpolation=cv2.INTER_AREA))
    return full_frames, scaled_frames, dioptres


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "for_display_baboon_100.png"
    angle = float(sys.argv[2]) if len(sys.argv) > 2 else 45.0

    print("Eq. 2 focus dioptres (i=1..10):", [round(f, 4) for f in focus_points()])

    canvas = cv2.imread(path, cv2.IMREAD_COLOR)
    if canvas is None:
        raise SystemExit(f"could not read {path}")
    full_frames, scaled_frames, dioptres = build_focal_stack(canvas, angle, seed=0)
    print(f"built {len(full_frames)} full-res frames {full_frames[0].shape} "
          f"and {len(scaled_frames)} scaled frames {scaled_frames[0].shape}")
    for i, f in enumerate(full_frames):
        cv2.imwrite(f"_debug_stack_frame_{i}.png", f)
    print("wrote _debug_stack_frame_0..9.png")
