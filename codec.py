"""Message codec: 32-bit payload <-> 15 rotation angles."""
import unireedsolomon as rs

N, K = 15, 8            # RS(15,8): 8 data symbols -> 15 code symbols
C_EXP = 4                # GF(16): symbols are 4-bit nibbles
BITS_PER_BLOCK = 4
LEVELS = 1 << BITS_PER_BLOCK        # 16 quantization levels
ALPHA = 180.0 / LEVELS                # 11.25 deg/level (Pramila 2012 Eq. 11)
PAYLOAD_BYTES = K * BITS_PER_BLOCK // 8   # 4 bytes = 32 bits

# GF(16) with primitive polynomial x^4+x+1 (0b10011 = 19), the standard
# minimal-degree primitive poly for GF(16).
_coder = rs.RSCoder(N, K, generator=2, prim=19, fcr=1, c_exp=C_EXP)


def _binary_to_gray(n):
    return n ^ (n >> 1)


def _gray_to_binary(g):
    n = g
    shift = 1
    while g >> shift:
        n ^= g >> shift
        shift += 1
    return n


def payload_to_symbols(payload: bytes):
    if len(payload) != PAYLOAD_BYTES:
        raise ValueError(f"payload must be exactly {PAYLOAD_BYTES} bytes ({PAYLOAD_BYTES * 8} bits)")
    val = int.from_bytes(payload, "big")
    return [(val >> (4 * i)) & 0xF for i in reversed(range(K))]


def symbols_to_payload(symbols):
    val = 0
    for s in symbols:
        val = (val << 4) | (int(s) & 0xF)
    return val.to_bytes(PAYLOAD_BYTES, "big")


def level_to_angle(level):
    """Angle level (0..15) -> rotation angle in degrees (bin center)."""
    return level * ALPHA + ALPHA / 2.0


def angle_to_level(theta):
    """Measured rotation angle (degrees, any real value) -> nearest of the
    16 quantization levels, using circular (mod 180) nearest-bin distance.
    """
    if theta is None:
        return None
    theta = theta % 180.0
    level = int(theta // ALPHA) % LEVELS
    return level


def encode_message(payload: bytes):
    """4-byte payload -> list of 15 rotation angles (degrees), one per
    non-sync block, in embedding order.
    """
    data_symbols = payload_to_symbols(payload)
    code = _coder.encode(data_symbols, return_string=False)
    thetas = []
    for sym in code:
        d = int(sym)
        level = _gray_to_binary(d)
        thetas.append(level_to_angle(level))
    return thetas


def decode_message(measured_thetas):
    """15 measured angles (degrees; entries may be None for an unreadable
    block) -> 4-byte payload, or None if RS(15,8) could not correct the
    errors (more than 3 bad/unreadable symbols).
    """
    if len(measured_thetas) != N:
        raise ValueError(f"expected {N} angles, got {len(measured_thetas)}")

    code_symbols = []
    for theta in measured_thetas:
        level = angle_to_level(theta)
        code_symbols.append(0 if level is None else _binary_to_gray(level))

    try:
        decoded, _ecc = _coder.decode(code_symbols, nostrip=True, return_string=False)
    except Exception:
        return None

    return symbols_to_payload(decoded[:K])


def decode_verified(measured_thetas, limit=None):
    """`decode_message`, but checked by re-encoding what it returned."""
    if limit is None:
        limit = (N - K) // 2
    payload = decode_message(measured_thetas)
    if payload is None:
        return None
    expected = encode_message(payload)
    if len(expected) != len(measured_thetas):
        return None
    disagreements = sum(1 for measured, wanted in zip(measured_thetas, expected)
                        if angle_to_level(measured) != angle_to_level(wanted))
    return (payload, disagreements) if disagreements <= limit else None


if __name__ == "__main__":
    import os

    payload = os.urandom(PAYLOAD_BYTES)
    thetas = encode_message(payload)
    print(f"payload:  {payload.hex()}")
    print(f"thetas:   {[round(t, 2) for t in thetas]}")

    # sanity: clean round trip
    out = decode_message(thetas)
    assert out == payload, f"clean round trip failed: {out.hex()} != {payload.hex()}"
    print("clean round trip: OK")

    noisy = list(thetas)
    import random
    rng = random.Random(0)
    corrupt_idx = rng.sample(range(N), 3)
    for i in corrupt_idx:
        noisy[i] = (noisy[i] + 90.0) % 180.0  # force a wrong symbol
    out = decode_message(noisy)
    assert out == payload, f"3-symbol-error round trip failed: {out}"
    print(f"3-symbol-error round trip (corrupted idx={corrupt_idx}): OK")

    # 4 corrupted symbols should exceed RS(15,8)'s correction capability
    noisy2 = list(thetas)
    corrupt_idx2 = rng.sample(range(N), 4)
    for i in corrupt_idx2:
        noisy2[i] = (noisy2[i] + 90.0) % 180.0
    out2 = decode_message(noisy2)
    print(f"4-symbol-error round trip (corrupted idx={corrupt_idx2}): "
          f"{'unexpectedly succeeded (silently wrong or got lucky)' if out2 == payload else 'correctly failed/garbled'}")
