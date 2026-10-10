"""Pramila et al. 2018, Section 3.4 (Reconfigure images) -- the part not
already covered by extractor.descreen()/rectify.py.
"""
import cv2
import numpy as np

SCALE_DENOM = 8  # "scaled to 1/8th of the size"


def descreen_sigma_for_dioptre(dioptre):
    """Literal coefficient: 1/8 of the focus distance in dioptres."""
    return dioptre / 8.0


def reconfigure(full_frame_bgr, dioptre):
    """Gaussian-blur (halftone descreen, literal per-frame coefficient)
    then scale to 1/8th size for feature computation. Returns
    (descreened_full_res, scaled_for_features).
    """
    sigma = descreen_sigma_for_dioptre(dioptre)
    descreened = cv2.GaussianBlur(full_frame_bgr, (0, 0), sigma)
    h, w = full_frame_bgr.shape[:2]
    scaled = cv2.resize(descreened, (w // SCALE_DENOM, h // SCALE_DENOM), interpolation=cv2.INTER_AREA)
    return descreened, scaled


def reconfigure_selected(full_frames, dioptres, selected_indices):
    """Section 3.4 applied to the M selected frames from 3.3 (I_j).
    Returns (descreened_full_res_list, scaled_list, scale_denom).
    """
    descreened_list = []
    scaled_list = []
    for i in selected_indices:
        d, s = reconfigure(full_frames[i], dioptres[i])
        descreened_list.append(d)
        scaled_list.append(s)
    return descreened_list, scaled_list, SCALE_DENOM


if __name__ == "__main__":
    import sys
    import capture
    import sharpness
    import substack

    path = sys.argv[1] if len(sys.argv) > 1 else "for_display_baboon_100.png"
    angle = float(sys.argv[2]) if len(sys.argv) > 2 else 45.0

    canvas = cv2.imread(path, cv2.IMREAD_COLOR)
    if canvas is None:
        raise SystemExit(f"could not read {path}")
    full_frames, scaled_frames, dioptres = capture.build_focal_stack(canvas, angle, seed=0)
    maps = sharpness.sharpness_maps(scaled_frames)
    selected, foreground, t, valuable_count, occurrence = substack.select_substack(maps)

    descreened_list, feat_scaled_list, denom = reconfigure_selected(full_frames, dioptres, selected)
    print(f"angle={angle}: selected={selected}, dioptres={[round(dioptres[i],2) for i in selected]}")
    for idx, i in enumerate(selected):
        sigma = descreen_sigma_for_dioptre(dioptres[i])
        print(f"  frame {i}: descreen_sigma={sigma:.3f}  full_res={full_frames[i].shape}  "
              f"scaled(1/{denom})={feat_scaled_list[idx].shape}")
