"""Frame detection + perspective rectification (paper Section 3.7 / Fig. 8),
without the focal-stack/registration machinery -- just enough to turn a
single real photo (screen or print, at an angle) into a fronto-parallel
512x512 crop that extractor.py's block grid assumes.
"""
import sys

import cv2
import numpy as np
import imread
import paths

from embedder import MARGIN, FRAME_THICKNESS

CORE_SIZE = 512

FRAME_OUTER_INSET = MARGIN - FRAME_THICKNESS - FRAME_THICKNESS // 2


def order_corners(pts):
    """Corners as tl, tr, br, bl -- by polar angle about the centroid, not by
    sums and differences of coordinates.
    """
    pts = pts.reshape(4, 2).astype(np.float64)
    cx, cy = pts[:, 0].mean(), pts[:, 1].mean()
    # y grows downward, so increasing atan2 walks the corners clockwise on
    # screen, which is the order tl, tr, br, bl is written in.
    angles = np.arctan2(pts[:, 1] - cy, pts[:, 0] - cx)
    ring = pts[np.argsort(angles, kind="stable")]
    start = int(np.argmin(ring.sum(axis=1)))
    return np.roll(ring, -start, axis=0).astype(np.float32)


THRESH_WINDOW = 51
THRESH_SIGMA = 8.0
THRESH_C = 10  # offset below local mean to trigger foreground; not given by the paper
CLOSE_ELEMENT_SIZE = 11

# "far off the image centre" (Section 3.7) has no number attached -- our
# own fill-in, as a fraction of min(h,w). See find_frame_quad.
CENTER_CUTOFF_FRAC = 0.30


def _adaptive_threshold_paper(gray, window=THRESH_WINDOW, sigma=THRESH_SIGMA):
    local_mean = cv2.GaussianBlur(gray.astype(np.float64), (window, window),
                                   sigmaX=sigma)
    return ((gray.astype(np.float64) < (local_mean - THRESH_C)).astype(np.uint8)) * 255


def find_frame_quad(img, window=THRESH_WINDOW, close_size=CLOSE_ELEMENT_SIZE):
    """window/close_size default to the paper-literal constants above,
    which is exactly what every existing call site (rectify()/
    rectify_affine(), both single-shot, real-photo-resolution) still
    gets -- unchanged behavior.
    """
    window = max(3, window | 1)  # cv2.GaussianBlur requires an odd kernel >= 1
    close_size = max(1, close_size)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = _adaptive_threshold_paper(gray, window=window, sigma=THRESH_SIGMA)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE,
                               cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                                          (close_size, close_size)))

    # Suzuki & Abe (1985) is exactly the algorithm cv2.findContours implements.
    contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    h, w = img.shape[:2]
    center = np.array([w / 2.0, h / 2.0])
    img_area = h * w
    candidates = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < 0.05 * img_area or area > 0.98 * img_area:
            continue
        # Douglas-Peucker simplification (paper: Heckbert and Garland, 1997).
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            centroid = approx.reshape(4, 2).mean(axis=0)
            dist = np.linalg.norm(centroid - center)
            candidates.append((dist, approx))

    if not candidates:
        return None
    dists = np.array([d for d, _ in candidates])
    cutoff = CENTER_CUTOFF_FRAC * min(h, w)
    near_center = [c for c in candidates if c[0] <= cutoff]
    if not near_center:
        # nothing within the cutoff -- fall back to the single closest
        # candidate rather than returning no frame at all.
        near_center = [min(candidates, key=lambda t: t[0])]
    largest = max(near_center, key=lambda t: cv2.contourArea(t[1]))
    return order_corners(largest[1])


def rectify_core(img, quad, use_affine=False):
    """Shared by rectify()/rectify_affine(): map quad -> the inset canvas
    position and crop the core. use_affine=True uses a least-squares
    affine fit (cv2.estimateAffine2D over the 4 corners -- affine has
    only 6 DOF so 4 correspondences over-determine it) instead of a full
    homography, per the paper's literal "assuming affine transformation"
    (Section 3.7) -- see rectify.py's module-level note on this being a
    genuine, not-yet-defaulted-to discrepancy.
    """
    canvas_size = CORE_SIZE + 2 * MARGIN
    o = FRAME_OUTER_INSET
    dst_full = np.array([[o, o], [canvas_size - 1 - o, o],
                          [canvas_size - 1 - o, canvas_size - 1 - o], [o, canvas_size - 1 - o]], dtype=np.float32)
    if use_affine:
        M, _inliers = cv2.estimateAffine2D(quad, dst_full)
        if M is None:
            return None
        warped = cv2.warpAffine(img, M, (canvas_size, canvas_size))
    else:
        M = cv2.getPerspectiveTransform(quad, dst_full)
        warped = cv2.warpPerspective(img, M, (canvas_size, canvas_size))
    return warped[MARGIN:MARGIN + CORE_SIZE, MARGIN:MARGIN + CORE_SIZE]


def rectify(photo_path, out_path="rectified.png", debug_path="rectify_debug.png"):
    img = imread.require(photo_path)

    quad = find_frame_quad(img)
    if quad is None:
        raise SystemExit("no frame quadrilateral found -- check lighting/contrast, "
                          "make sure the whole black frame is visible in the photo")

    dbg = img.copy()
    cv2.polylines(dbg, [quad.astype(np.int32)], True, (0, 0, 255), 4)
    cv2.imwrite(paths.resolve(debug_path), dbg)
    print(f"wrote {debug_path} (sanity-check the red quad hugs the frame)")

    core = rectify_core(img, quad, use_affine=False)

    out_path = paths.resolve(out_path)
    cv2.imwrite(out_path, core)
    print(f"wrote {out_path}")
    return core


def rectify_affine(photo_path, out_path="rectified.png", debug_path="rectify_debug.png"):
    """Same as rectify() but literal-per-Section-3.7: affine, not
    perspective/homography. See rectify_core()'s docstring.
    """
    img = imread.require(photo_path)
    quad = find_frame_quad(img)
    if quad is None:
        raise SystemExit("no frame quadrilateral found")
    core = rectify_core(img, quad, use_affine=True)
    if core is None:
        raise SystemExit("affine estimation failed")
    out_path = paths.resolve(out_path)
    cv2.imwrite(out_path, core)
    return core


if __name__ == "__main__":
    photo = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else "rectified.png"
    rectify(photo, out)
