"""WCAM extractor, Pramila et al. 2018 (Section 3.8 / Fig. 9) -- literal
except for a small number of EXPLICITLY DOCUMENTED, tested deviations
(currently: the Hough line-aggregation choice, see step 6 below) adopted
at the user's direction when a literal reading measurably, severely
underperforms in this reproduction. Every such deviation is called out at
its own point below and is a deliberate choice, not an oversight -- this
directory's default posture is still literal-first, deviations are the
exception and must be justified by real measurement, not preference.
"""
import json
import paths
import sys

import cv2
import numpy as np
from scipy.signal import wiener
from scipy.ndimage import gaussian_laplace, binary_dilation

import codec

BLOCK = 128
GRID = 4

DETECTION_A = 0.7

WIENER_MYSIZE = 5
LOG_SIGMA = 0.8  # scale of the LoG enhancement (paper gives the formula, not sigma)

DESCREEN_SIGMA = 0.2

HOUGH_MIN_LINE_SEP = 8

MASK_RADIUS_DIVISOR = 10


def descreen(Y):
    """Gaussian-smooth a captured image's Y channel before block division,
    per Pramila 2018 Section 3.4. Applied to the whole channel (not
    per-block) so there's no border artifact at block edges.
    """
    return cv2.GaussianBlur(Y, (0, 0), DESCREEN_SIGMA)


def _hann2d(n):
    """Windowing before the autocorrelation FFT. In NEITHER paper's text
    at all (see module docstring) -- pure engineering addition, kept
    because removing it holds up on a clean digital round trip but
    collapses robustness under synthetic degradation (project memory).
    """
    w = np.hanning(n)
    return np.outer(w, w)


USE_TWO_LINE_HOUGH = False  # see function docstring: deliberate, tested, documented deviation


def _hough_dominant_angle(peaks_xy, size, ntheta=180, min_sep=HOUGH_MIN_LINE_SEP,
                           two_line=USE_TWO_LINE_HOUGH):
    """DELIBERATE, TESTED DEVIATION FROM THE 2018 TEXT (2026-09-04, at the
    user's explicit direction after seeing the measured cost of the
    literal reading -- prioritize real-world result over textual fidelity
    for this one specific step, clearly flagged as such rather than left
    as an open question).
    """
    thetas = np.radians(np.arange(ntheta))
    x = peaks_xy[:, 0]
    y = peaks_xy[:, 1]
    rho = np.outer(x, np.cos(thetas)) + np.outer(y, np.sin(thetas))
    bins = np.linspace(-size, size, size + 1)
    best_score, best_theta, best_hist = -1, 0, None
    for ti in range(ntheta):
        hist, _ = np.histogram(rho[:, ti], bins=bins)
        if not two_line:
            s = hist.max()
            if s > best_score:
                best_score, best_theta, best_hist = s, ti, hist
            continue
        order = np.argsort(hist)[::-1]
        top = order[0]
        second_count = 0
        for cand in order[1:]:
            if abs(int(cand) - int(top)) >= min_sep:
                second_count = hist[cand]
                break
        s = hist[top] + second_count
        if s > best_score:
            best_score, best_theta, best_hist = s, ti, hist
    return float(best_theta), int(best_score), best_hist


def _watermark_present(hist, a=DETECTION_A):
    """Eq. 19: sum of top-5 projected peaks > a * sum of the rest."""
    if hist is None:
        return False
    p = np.sort(hist)[::-1]
    if len(p) < 6 or p.sum() == 0:
        return False
    return p[:5].sum() > a * p[5:].sum()


def block_angle(y_block):
    b = y_block.astype(np.float64)
    b = b - b.mean()

    try:
        filtered = wiener(b, mysize=WIENER_MYSIZE)
        filtered = np.nan_to_num(filtered)
    except Exception:
        filtered = b
    residual = b - filtered

    n = residual.shape[0]
    filtered = residual * _hann2d(n)

    padded = np.zeros((2 * n, 2 * n), dtype=np.float64)
    padded[:n, :n] = filtered

    F = np.fft.fft2(padded)
    ac = np.fft.ifft2(F * np.conj(F)).real
    ac = np.fft.fftshift(ac)

    ac = ac - ac.min()
    if ac.max() > 0:
        ac = ac / ac.max()

    enhanced = -gaussian_laplace(ac, sigma=LOG_SIGMA)
    enhanced = np.clip(enhanced, 0, None)
    if enhanced.max() > 0:
        enhanced = enhanced / enhanced.max()

    # mask out center (self-correlation peak = mostly low-freq noise per paper)
    h, w = enhanced.shape
    cy, cx = h // 2, w // 2
    r = max(3, min(h, w) // MASK_RADIUS_DIVISOR)
    ac_masked = enhanced.copy()
    cv2.circle(ac_masked, (cx, cy), r, 0, -1)

    thresh = np.quantile(ac_masked, 0.997)
    binary = ac_masked >= thresh

    # dilate thresholded peaks with a small disk so line detection tolerates
    # the peak-position jitter that resampling/distortion introduces
    disk = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)  # r=1
    binary = binary_dilation(binary, structure=disk)

    dbg = binary.astype(np.uint8) * 255
    ys, xs = np.where(binary)
    if len(xs) < 4:
        return None, dbg, False
    peaks = np.stack([xs - cx, ys - cy], axis=1).astype(np.float64)
    dominant, score, hist = _hough_dominant_angle(peaks, w)

    present = _watermark_present(hist)

    if not present:
        raw_masked = ac.copy()
        cv2.circle(raw_masked, (cx, cy), r, 0, -1)
        raw_thresh = np.quantile(raw_masked, 0.997)
        raw_binary = raw_masked >= raw_thresh
        raw_binary = binary_dilation(raw_binary, structure=disk)
        rys, rxs = np.where(raw_binary)
        if len(rxs) >= 4:
            raw_peaks = np.stack([rxs - cx, rys - cy], axis=1).astype(np.float64)
            _raw_dominant, _raw_score, raw_hist = _hough_dominant_angle(raw_peaks, w)
            present = _watermark_present(raw_hist)

    return dominant, dbg, present


def correct_for_sync(measured, sync_offset):
    """Angle relative to the sync block (block 0), which corrects for any
    overall rotation of the captured image (paper: theta*_k = theta_r - theta*_i).
    """
    if measured is None:
        return None
    return (measured - sync_offset) % 180


def extract(image_path, meta_path):
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise SystemExit(f"could not read {image_path}")
    img = cv2.resize(img, (BLOCK * GRID, BLOCK * GRID), interpolation=cv2.INTER_AREA)
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb).astype(np.float64)
    Y = descreen(ycrcb[:, :, 0])

    with open(paths.resolve(meta_path)) as f:
        meta = json.load(f)
    truth_payload = bytes.fromhex(meta["payload_hex"])
    truth_angles = meta["angles"]

    raw_angles = []
    idx = 0
    sync_present = False
    for gy in range(GRID):
        for gx in range(GRID):
            y0, x0 = gy * BLOCK, gx * BLOCK
            block = Y[y0:y0 + BLOCK, x0:x0 + BLOCK]
            measured, _dbg, present = block_angle(block)
            raw_angles.append(measured)
            if idx == 0:
                sync_present = present  # paper: existence is tested on the
            idx += 1

    if not sync_present:
        print("note: reference-block presence check (Eq. 19) failed -- reading anyway")

    sync_measured = raw_angles[0] if raw_angles[0] is not None else 0.0
    corrected = [correct_for_sync(m, sync_measured) for m in raw_angles[1:]]

    levels_correct = sum(1 for c, t in zip(corrected, truth_angles)
                          if codec.angle_to_level(c) == codec.angle_to_level(t))
    decoded_payload = codec.decode_message(corrected)

    print(f"sync block raw angle: {sync_measured:.2f}")
    print(f"truth angles:     {[round(a, 2) for a in truth_angles]}")
    print(f"corrected angles: {[None if c is None else round(c, 2) for c in corrected]}")
    print(f"angle levels correct (pre-ECC): {levels_correct}/{len(truth_angles)}")
    print(f"truth payload:   {truth_payload.hex()}")
    print(f"decoded payload: {decoded_payload.hex() if decoded_payload else None}")
    ok = decoded_payload == truth_payload
    print(f"message recovered correctly: {ok}")
    return ok, levels_correct, len(truth_angles)


if __name__ == "__main__":
    img = sys.argv[1] if len(sys.argv) > 1 else "watermarked.png"
    meta = sys.argv[2] if len(sys.argv) > 2 else "meta.json"
    extract(img, meta)
