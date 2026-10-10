"""Pramila et al. 2018, Section 3.6 (Image blending)."""
import cv2
import numpy as np

L0_SOFTEN_SIGMA = 1.0   # unspecified by the paper; small, keeps L0 (detail) blend close to Q
G1_SOFTEN_SIGMA = 4.0   # unspecified by the paper; larger, smooths the coarse-layer seams more


def sharpness_mask(registered_gray_list):
    """Eq. 7: per-pixel argmax of the 5x5-averaged abs-Laplacian across
    the registered images.
    """
    scores = []
    for g in registered_gray_list:
        L = np.abs(cv2.Laplacian(g.astype(np.float64), cv2.CV_64F, ksize=3))
        smoothed = cv2.blur(L, (5, 5))
        scores.append(smoothed)
    stacked = np.stack(scores)
    return np.argmax(stacked, axis=0)


def _one_hot_weights(Q, m, sigma):
    """Soft one-hot weight maps (see module docstring): (Q==j) blurred by
    sigma, then renormalized across j to sum to 1 at every pixel.
    """
    weights = []
    for j in range(m):
        w = (Q == j).astype(np.float64)
        if sigma > 0:
            w = cv2.GaussianBlur(w, (0, 0), sigma)
        weights.append(w)
    stacked = np.stack(weights)
    total = stacked.sum(axis=0, keepdims=True)
    total = np.maximum(total, 1e-9)
    return stacked / total


def blend_stack(registered_bgr_list):
    """One-level Laplacian pyramid blend of M registered (same-geometry)
    BGR images, using the Eq. 7 sharpness mask as alpha. Returns the
    blended BGR image.
    """
    m = len(registered_bgr_list)
    if m == 1:
        return registered_bgr_list[0].copy()

    grays = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in registered_bgr_list]
    Q = sharpness_mask(grays)

    h, w = registered_bgr_list[0].shape[:2]
    W0 = _one_hot_weights(Q, m, L0_SOFTEN_SIGMA)                     # (m, h, w)

    Q_small = cv2.resize(Q.astype(np.uint8), (w // 2, h // 2), interpolation=cv2.INTER_NEAREST)
    W1 = _one_hot_weights(Q_small.astype(np.int64), m, G1_SOFTEN_SIGMA)  # (m, h//2, w//2)

    blended_L0 = np.zeros((h, w, 3), dtype=np.float64)
    blended_G1 = np.zeros((h // 2, w // 2, 3), dtype=np.float64)

    for j, img in enumerate(registered_bgr_list):
        img_f = img.astype(np.float64)
        G0 = img_f
        G1 = cv2.pyrDown(G0)
        G1_up = cv2.resize(G1, (w, h), interpolation=cv2.INTER_LINEAR)
        L0 = G0 - G1_up

        blended_L0 += L0 * W0[j][:, :, None]
        blended_G1 += G1 * W1[j][:, :, None]

    blended_G1_up = cv2.resize(blended_G1, (w, h), interpolation=cv2.INTER_LINEAR)
    blended = blended_L0 + blended_G1_up
    return np.clip(blended, 0, 255).astype(np.uint8)


def warp_registered(scaled_frames, matrices):
    """Apply register.py's homographies to actually produce the Î_j^r
    registered images (register.py itself only returns the matrices).
    """
    ref_shape = scaled_frames[0].shape[:2]
    out = [scaled_frames[0]]
    for img, M in zip(scaled_frames[1:], matrices[1:]):
        if M is None:
            continue
        warped = cv2.warpPerspective(img, M, (ref_shape[1], ref_shape[0]), borderMode=cv2.BORDER_REPLICATE)
        out.append(warped)
    return out


if __name__ == "__main__":
    import sys
    import capture
    import sharpness
    import substack
    import reconfigure
    import register

    path = sys.argv[1] if len(sys.argv) > 1 else "for_display_baboon_100.png"
    angle = float(sys.argv[2]) if len(sys.argv) > 2 else 45.0

    canvas = cv2.imread(path, cv2.IMREAD_COLOR)
    if canvas is None:
        raise SystemExit(f"could not read {path}")
    full_frames, scaled_frames_hm, dioptres = capture.build_focal_stack(canvas, angle, seed=0)
    maps = sharpness.sharpness_maps(scaled_frames_hm)
    selected, foreground, t, valuable_count, occurrence = substack.select_substack(maps)
    descreened_list, feat_scaled_list, denom = reconfigure.reconfigure_selected(full_frames, dioptres, selected)
    matrices = register.register_stack(feat_scaled_list)

    registered = warp_registered(feat_scaled_list, matrices)
    print(f"angle={angle}: blending {len(registered)} registered frames (of {len(selected)} selected)")
    blended = blend_stack(registered)
    cv2.imwrite("_debug_blended.png", blended)
    print(f"wrote _debug_blended.png, shape={blended.shape}")
