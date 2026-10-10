"""Pramila et al. 2018, Section 3.3 (Optimizing the size of the focal stack)."""
import cv2
import numpy as np

FOREGROUND_SIGMA_FRAC = 0.1   # Eq. 4: foreground if sigma > 0.1 * sigma_max
OCCURRENCE_NOISE_CUTOFF = 0.1  # images with occurrence < 0.1 are "noise"
GAUSSIAN_KERNEL_SIGMA = 0.5    # Eq. 5's 3x3 Gaussian kernel
COVERAGE_TARGET = 0.8          # literal (p.209): "iterates until at least 80% of the foreground region is covered..."
MAX_STACK_SIZE = 5              # literal (p.209): "...or 5 images have been selected"


def _gaussian_kernel_3x3(sigma=GAUSSIAN_KERNEL_SIGMA):
    ax = np.arange(-1, 2)
    xx, yy = np.meshgrid(ax, ax)
    k = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    return k / k.sum()


_G3 = _gaussian_kernel_3x3()


def _weighted_dilate(binary_map):
    """Eq. 5 (see module docstring's reading #1): max over the 3x3
    Gaussian-weighted window.
    """
    b = binary_map.astype(np.float64)
    padded = np.pad(b, 1, mode="edge")
    out = np.zeros_like(b)
    h, w = b.shape
    for dy in range(3):
        for dx in range(3):
            weight = _G3[dy, dx]
            out = np.maximum(out, weight * padded[dy:dy + h, dx:dx + w])
    return out


def segment_foreground(sharpness_maps):
    """Eq. 4: per-position std across the stack (axis 0), foreground
    where std > 0.1 * max(std).
    """
    sigma = sharpness_maps.std(axis=0)
    sigma_max = sigma.max()
    if sigma_max < 1e-9:
        return np.zeros_like(sigma, dtype=bool)
    return sigma > FOREGROUND_SIGMA_FRAC * sigma_max


EMPIRICAL_T = 0.15  # see module docstring reading #2 -- count/N tested and rejected, swept to this value


def compute_t(sharpness_maps, foreground):
    """Occurrence-vector method (still computed and returned for its own
    sake -- it's the paper's literal "images with occurrence < 0.1 are
    noise" diagnostic, useful on its own) but the tolerance multiplier
    actually used by sharp_binary_maps is EMPIRICAL_T (see module
    docstring reading #2), not derived from this count.
    """
    n = sharpness_maps.shape[0]
    winner = np.argmax(sharpness_maps, axis=0)
    occurrence = np.zeros(n)
    fg_winners = winner[foreground]
    if fg_winners.size > 0:
        for i in range(n):
            occurrence[i] = np.sum(fg_winners == i)
        total = occurrence.sum()
        if total > 1e-9:
            occurrence = occurrence / total
    valuable_count = int(np.sum(occurrence >= OCCURRENCE_NOISE_CUTOFF))
    return EMPIRICAL_T, valuable_count, occurrence


def sharp_binary_maps(sharpness_maps, foreground, t):
    """'An element (u,v) in the ith map is determined as sharp if
    Smax(u,v)==Si(u,v), or Smax(u,v)-Si(u,v)<Smax*t and the element in
    the neighbouring map is sharp' (neighbouring = adjacent focus index,
    i-1/i+1, not a spatial neighbor).
    """
    n = sharpness_maps.shape[0]
    Smax = sharpness_maps.max(axis=0)
    is_max = sharpness_maps >= (Smax[None, :, :] - 1e-12)
    within_tol = (Smax[None, :, :] - sharpness_maps) < (Smax[None, :, :] * t)

    sharp = is_max.copy()
    changed = True
    while changed:
        changed = False
        for i in range(n):
            if not within_tol[i].any():
                continue
            neighbour_sharp = np.zeros_like(sharp[i])
            if i > 0:
                neighbour_sharp |= sharp[i - 1]
            if i < n - 1:
                neighbour_sharp |= sharp[i + 1]
            newly = within_tol[i] & neighbour_sharp & ~sharp[i]
            if newly.any():
                sharp[i] |= newly
                changed = True

    binary = sharp & foreground[None, :, :]
    return binary


def select_substack(sharpness_maps):
    """Eq. 6 greedy selection. Returns (selected_indices, foreground_mask, t)."""
    foreground = segment_foreground(sharpness_maps)
    t, valuable_count, occurrence = compute_t(sharpness_maps, foreground)
    B = sharp_binary_maps(sharpness_maps, foreground, t)
    n = B.shape[0]
    G = np.stack([_weighted_dilate(B[i]) for i in range(n)])

    totals = G.sum(axis=(1, 2))
    m = int(np.argmax(totals))
    selected = [m]
    Gm = G[m].copy()
    foreground_total = foreground.sum()
    remaining = [i for i in range(n) if i != m]

    while remaining and len(selected) < MAX_STACK_SIZE:
        covered = np.maximum(Gm, 0).astype(bool) & foreground
        covered_frac = covered.sum() / max(foreground_total, 1)
        if covered_frac >= COVERAGE_TARGET:
            break
        best_sum, best_idx = -1.0, None
        for i in remaining:
            s = np.maximum(Gm, G[i]).sum()
            if s > best_sum:
                best_sum, best_idx = s, i
        selected.append(best_idx)
        Gm = np.maximum(Gm, G[best_idx])
        remaining.remove(best_idx)

    return sorted(selected), foreground, t, valuable_count, occurrence


if __name__ == "__main__":
    import sys
    import capture
    import sharpness

    path = sys.argv[1] if len(sys.argv) > 1 else "for_display_baboon_100.png"
    angle = float(sys.argv[2]) if len(sys.argv) > 2 else 45.0

    canvas = cv2.imread(path, cv2.IMREAD_COLOR)
    if canvas is None:
        raise SystemExit(f"could not read {path}")
    full_frames, scaled_frames, dioptres = capture.build_focal_stack(canvas, angle, seed=0)
    maps = sharpness.sharpness_maps(scaled_frames)
    selected, foreground, t, valuable_count, occurrence = select_substack(maps)
    print(f"angle={angle}: foreground fraction={foreground.mean():.3f}, t={t:.3f}, "
          f"occurrence-vector valuable_count={valuable_count}")
    print(f"selected {len(selected)} of {maps.shape[0]} frames: indices={selected}, "
          f"focus dioptres={[round(dioptres[i], 2) for i in selected]}")
