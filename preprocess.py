"""Cover-image preprocessing before watermark embedding."""
import numpy as np
from scipy.signal import wiener

MASK_FRACTION = 0.8  # fraction of pixels eligible for the substitution (Eq. 1: 80%)

DEFAULT_MYSIZE = 3


def preprocess(y_channel, seed=None, mysize=DEFAULT_MYSIZE):
    y = y_channel.astype(np.float64)

    try:
        y_wiener = wiener(y, mysize=mysize)
        y_wiener = np.nan_to_num(y_wiener, nan=0.0, posinf=255.0, neginf=0.0)
    except Exception:
        y_wiener = y.copy()

    rng = np.random.default_rng(seed)
    r = rng.random(y.shape) < MASK_FRACTION  # ~80% True, matching r(x,y) in Eq. 1

    substitute = r & ((y - y_wiener) > 0)
    return np.where(substitute, y_wiener, y)


if __name__ == "__main__":
    import sys

    import cv2

    path = sys.argv[1] if len(sys.argv) > 1 else "test_src.png"
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    Y = ycrcb[:, :, 0]
    X = preprocess(Y, seed=1234)
    diff = X - Y.astype(np.float64)
    print(f"preprocessed: {(diff != 0).mean() * 100:.1f}% of pixels changed, "
          f"mean |change|={np.abs(diff).mean():.3f}, max |change|={np.abs(diff).max():.3f}")
    cv2.imwrite("debug_preprocessed.png", np.clip(X, 0, 255).astype(np.uint8))
    print("wrote debug_preprocessed.png")
