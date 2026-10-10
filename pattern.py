"""Literal pattern generation, Pramila et al. 2018 (Section 2 / Fig. 2),
falling back to Pramila et al. 2012 only where 2018 explicitly defers to
it for embedding detail.
"""
import cv2
import numpy as np

RNG_SEED = 1234
TILE_W = 7
TILE_H = 28

RNG_KIND = "PCG64"


#: Готовая плитка вместо псевдослучайной; ставится через set_tile().
TILE_OVERRIDE = None


def set_tile(tile):
    """Встраивать заданной плиткой. None возвращает псевдослучайную."""
    global TILE_OVERRIDE
    TILE_OVERRIDE = None if tile is None else np.asarray(tile, dtype=np.float64)


def load_tile(path):
    """Плитка из .npy, либо из картинки: светлое даёт +1, тёмное -1."""
    if path.lower().endswith('.npy'):
        tile = np.load(path).astype(np.float64)
    else:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise SystemExit(f"не прочитать плитку: {path}")
        tile = np.where(img.astype(np.float64) > 127.5, 1.0, -1.0)
    if tile.ndim != 2 or min(tile.shape) < 2:
        raise SystemExit(f"плитка должна быть двумерной и не меньше 2x2, а тут {tile.shape}")
    return tile


def _make_bit_generator(seed, rng_kind):
    cls = getattr(np.random, rng_kind)
    return np.random.Generator(cls(seed))


def make_tile(seed=RNG_SEED, rng_kind=RNG_KIND, dc=0.0, balanced=True):
    """The RNG is always drawn in the canonical (short, long) shape and only
    then transposed, so that swapping TILE_W/TILE_H rotates the pattern by 90deg
    without also changing which pseudorandom realization it is. Drawing directly
    into the configured shape would reshape the same 196 values into a different
    layout instead of transposing them (measured: MSSIM 0.425 between the two,
    mean |diff| 14.6 gray levels) -- i.e. a different watermark, not a rotated
    one, which would confound any orientation A/B.
    """
    if TILE_OVERRIDE is not None:
        return TILE_OVERRIDE + dc
    rng = _make_bit_generator(seed, rng_kind) if rng_kind != "PCG64" else np.random.default_rng(seed)
    short, long_ = min(TILE_H, TILE_W), max(TILE_H, TILE_W)
    t = rng.integers(0, 2, size=(short, long_)).astype(np.float64) * 2 - 1
    if balanced:
        t = _balance(t, rng)
    t = t + dc
    return t.T if TILE_H > TILE_W else t


def _balance(t, rng, iters=4000):
    """Drive every row and column sum toward zero by greedy sign flips. ON by
    default since 2026-09-08.
    """
    cost = lambda a: (a.sum(axis=1) ** 2).sum() + (a.sum(axis=0) ** 2).sum()
    c = cost(t)
    for _ in range(iters):
        i, j = rng.integers(t.shape[0]), rng.integers(t.shape[1])
        t[i, j] *= -1
        c2 = cost(t)
        if c2 <= c:
            c = c2
        else:
            t[i, j] *= -1
    return t


def make_pattern(block_size, theta_deg, seed=RNG_SEED, interpolation=cv2.INTER_LINEAR, rng_kind=RNG_KIND,
                 dc=0.0, balanced=True, noise=0.0):
    """Tile the raw 28x7 pseudorandom tile to fill a canvas, rotate by
    theta_deg with bilinear interpolation (2012 Eq. 14), crop the center
    block_size x block_size region.
    """
    tile = make_tile(seed, rng_kind=rng_kind, dc=dc, balanced=balanced)
    diag = int(np.ceil(block_size * np.sqrt(2))) + 4
    reps_x = int(np.ceil(diag / tile.shape[1]))
    reps_y = int(np.ceil(diag / tile.shape[0]))
    canvas = np.tile(tile, (reps_y, reps_x))
    ch, cw = canvas.shape

    center = (cw / 2.0, ch / 2.0)
    M = cv2.getRotationMatrix2D(center, -theta_deg, 1.0)
    rotated = cv2.warpAffine(canvas, M, (cw, ch), flags=interpolation,
                              borderMode=cv2.BORDER_WRAP)

    y0 = int(round(ch / 2.0 - block_size / 2.0))
    x0 = int(round(cw / 2.0 - block_size / 2.0))
    out = rotated[y0:y0 + block_size, x0:x0 + block_size]
    if noise > 0:
        out = _mix_noise(out, noise, seed, theta_deg)
    return out


def _mix_noise(p, alpha, seed, theta_deg):
    """Blend an aperiodic component in, renormalized to the tiled pattern's own std."""
    rng = np.random.default_rng(seed + 100000 + int(round(theta_deg * 100)))
    m = (1 - alpha) * p + alpha * rng.normal(0, 1, p.shape)
    return m / m.std() * p.std()
