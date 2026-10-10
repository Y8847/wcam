"""Just-Noticeable-Difference (JND) map."""
import cv2
import numpy as np

LAMBDA1 = 2.0
LAMBDA2 = 3.0
T0 = 17.0
GAMMA = 3.0 / 128.0
CHOU_LI_LAMBDA = 0.5

GLOBAL_LO = 7.48
GLOBAL_HI = 63.66

_G1 = np.array([
    [0, 0, 0, 0, 0],
    [1, 3, 8, 3, 1],
    [0, 0, 0, 0, 0],
    [-1, -3, -8, -3, -1],
    [0, 0, 0, 0, 0],
], dtype=np.float64)


_G2 = np.array([
    [0, 0, 1, 0, 0],
    [0, 8, 3, 8, 0],
    [1, 3, 0, -3, -1],
    [0, 0, -3, 0, 0],
    [0, 0, -1, 0, 0],
], dtype=np.float64)

_G3 = _G2.T
_G4 = _G1.T
_GRAD_KERNELS = (_G1, _G2, _G3, _G4)

_G2_CHOU_LI = np.array([
    [0, 0, 1, 0, 0],
    [0, 8, 3, 0, 0],
    [1, 3, 0, -3, -1],
    [0, 0, -3, -8, 0],
    [0, 0, -1, 0, 0],
], dtype=np.float64)

_GRAD_KERNELS_CHOU_LI = (_G1, _G2_CHOU_LI, np.fliplr(_G2_CHOU_LI), _G1.T)

BETA_COEF_PAPER = 0.1
BETA_COEF_CHOU_LI = 0.01

# Weighted low-pass operator for background luminance (Eq. 9/10).
_B = np.array([
    [1, 1, 1, 1, 1],
    [1, 2, 2, 2, 1],
    [1, 2, 0, 2, 1],
    [1, 2, 2, 2, 1],
    [1, 1, 1, 1, 1],
], dtype=np.float64)


def _corr(p, kernel):
    """5x5 correlation matching the paper's sum_{i,j=1..5} p(x-3+i, y-3+j) *
    K(i,j) convention (i.e. NOT a flipped convolution -- cv2.filter2D
    already computes correlation, not convolution, so the kernel is used
    as-is).
    """
    return cv2.filter2D(p, -1, kernel, borderType=cv2.BORDER_REFLECT)


def compute(y_channel, global_norm=False, correct_kernels=True,
            beta_coef=BETA_COEF_CHOU_LI, floor=0.0):
    """Full-resolution JND map, in [0, 1], same shape as y_channel."""
    p = y_channel.astype(np.float64)

    mg = np.zeros_like(p)
    for kernel in (_GRAD_KERNELS_CHOU_LI if correct_kernels else _GRAD_KERNELS):
        grad = np.abs(_corr(p, kernel)) / 16.0
        mg = np.maximum(mg, grad)

    bg = _corr(p, _B) / 32.0

    alpha = bg * 0.0001 + 0.115
    beta = CHOU_LI_LAMBDA - bg * beta_coef
    f1 = mg * alpha + beta

    f2 = np.where(
        bg <= 127.0,
        T0 * (1.0 - np.sqrt(np.clip(bg / 127.0, 0.0, None))) + 3.0,
        GAMMA * (bg - 127.0) + 3.0,
    )

    jnd = LAMBDA1 * (f1 + LAMBDA2) + f2

    if global_norm:
        lo, hi = GLOBAL_LO, GLOBAL_HI
        return _apply_floor(np.clip((jnd - lo) / (hi - lo), 0.0, 1.0), floor)

    lo, hi = jnd.min(), jnd.max()
    if hi - lo < 1e-9:
        return np.zeros_like(jnd)
    return _apply_floor((jnd - lo) / (hi - lo), floor)


def _apply_floor(j, floor):
    """Push the bottom `floor` of the normalized JND down to exactly 0."""
    return j if floor <= 0 else np.clip((j - floor) / (1.0 - floor), 0.0, 1.0)


if __name__ == "__main__":
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else "test_src.png"
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    Y = ycrcb[:, :, 0]
    j = compute(Y)
    print(f"JND map: shape={j.shape}, min={j.min():.4f}, max={j.max():.4f}, mean={j.mean():.4f}")
    cv2.imwrite("debug_jnd.png", (j * 255).astype(np.uint8))
    print("wrote debug_jnd.png")
