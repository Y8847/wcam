"""Pramila et al. 2018, Section 3.5 (Image registration)."""
import cv2
import numpy as np

N_OCTAVES = 4
N_OCTAVE_LAYERS = 2
AKAZE_THRESHOLD = 0.002

RATIO_TEST = 0.75          # unspecified by the paper, standard Lowe's-ratio value
RANSAC_REPROJ_THRESH = 5.0  # unspecified by the paper, standard default
MIN_INLIERS = 8
MIN_INLIER_RATIO = 0.4


def make_akaze():
    return cv2.xfeatures2d.AKAZE_create(
        descriptor_type=cv2.xfeatures2d.AKAZE_DESCRIPTOR_MLDB,
        nOctaves=N_OCTAVES, nOctaveLayers=N_OCTAVE_LAYERS, threshold=AKAZE_THRESHOLD)


def register_to_reference(ref_bgr_or_gray, frame_bgr_or_gray):
    """AKAZE detect+describe on both images, brute-force Hamming match
    with a ratio test, RANSAC homography. Returns the 3x3 matrix mapping
    frame -> ref coordinates, or None if registration failed.
    """
    def to_gray(x):
        return cv2.cvtColor(x, cv2.COLOR_BGR2GRAY) if x.ndim == 3 else x

    ref_gray = to_gray(ref_bgr_or_gray)
    frame_gray = to_gray(frame_bgr_or_gray)

    akaze = make_akaze()
    k1, d1 = akaze.detectAndCompute(ref_gray, None)
    k2, d2 = akaze.detectAndCompute(frame_gray, None)
    if d1 is None or d2 is None or len(k1) < 4 or len(k2) < 4:
        return None, 0

    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    knn = bf.knnMatch(d2, d1, k=2)
    good = [m for pair in knn if len(pair) == 2 for m, n in [pair] if m.distance < RATIO_TEST * n.distance]
    if len(good) < 4:
        return None, len(good)

    src = np.float32([k2[m.queryIdx].pt for m in good])
    dst = np.float32([k1[m.trainIdx].pt for m in good])
    M, mask = cv2.findHomography(src, dst, cv2.RANSAC, RANSAC_REPROJ_THRESH)
    if M is None or mask is None:
        return None, len(good)

    inliers = int(mask.sum())
    if inliers < MIN_INLIERS or inliers / len(good) < MIN_INLIER_RATIO:
        return None, inliers
    return M, inliers


def register_stack(scaled_frames):
    """Register frames[1:] to frames[0] (the reference). Returns list of
    3x3 matrices (None for frame 0 itself / for failed registrations).
    """
    ref = scaled_frames[0]
    matrices = [np.eye(3)]
    for frame in scaled_frames[1:]:
        M, inliers = register_to_reference(ref, frame)
        matrices.append(M)
    return matrices


if __name__ == "__main__":
    import sys
    import capture
    import sharpness
    import substack
    import reconfigure

    path = sys.argv[1] if len(sys.argv) > 1 else "for_display_baboon_100.png"
    angle = float(sys.argv[2]) if len(sys.argv) > 2 else 45.0

    canvas = cv2.imread(path, cv2.IMREAD_COLOR)
    if canvas is None:
        raise SystemExit(f"could not read {path}")
    full_frames, scaled_frames_hm, dioptres = capture.build_focal_stack(canvas, angle, seed=0)
    maps = sharpness.sharpness_maps(scaled_frames_hm)
    selected, foreground, t, valuable_count, occurrence = substack.select_substack(maps)
    descreened_list, feat_scaled_list, denom = reconfigure.reconfigure_selected(full_frames, dioptres, selected)

    print(f"angle={angle}: registering {len(feat_scaled_list)} selected frames "
          f"(indices {selected}) at 1/{denom} scale")
    matrices = register_stack(feat_scaled_list)
    for i, M in enumerate(matrices):
        if M is None:
            print(f"  frame {selected[i]}: registration FAILED")
        else:
            print(f"  frame {selected[i]}: registered, translation~=({M[0,2]:.1f},{M[1,2]:.1f})")
