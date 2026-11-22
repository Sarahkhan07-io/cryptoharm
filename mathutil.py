"""Small, dependency-free number theory + toy elliptic-curve helpers (generation/verification only)."""
import math

_SMALL = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37)


def is_prime(n: int) -> bool:
    if n < 2:
        return False
    for p in _SMALL:
        if n % p == 0:
            return n == p
    d, s = n - 1, 0
    while d % 2 == 0:
        d //= 2
        s += 1
    for a in _SMALL:
        x = pow(a, d, n)
        if x in (1, n - 1):
            continue
        for _ in range(s - 1):
            x = x * x % n
            if x == n - 1:
                break
        else:
            return False
    return True


def next_prime(n: int, mod4=None) -> int:
    n += 1
    while True:
        if is_prime(n) and (mod4 is None or n % 4 == mod4):
            return n
        n += 1


def gen_prime(bits: int, rng, mod4=None) -> int:
    while True:
        c = rng.getrandbits(bits) | (1 << (bits - 1)) | 1
        if is_prime(c) and (mod4 is None or c % 4 == mod4):
            return c


# ---- elliptic curves y^2 = x^3 + a x + b over F_p (affine points, None = infinity) ----
def ec_add(P, Q, a, p):
    if P is None:
        return Q
    if Q is None:
        return P
    x1, y1 = P
    x2, y2 = Q
    if x1 == x2 and (y1 + y2) % p == 0:
        return None
    if P == Q:
        lam = (3 * x1 * x1 + a) * pow(2 * y1, -1, p) % p
    else:
        lam = (y2 - y1) * pow(x2 - x1, -1, p) % p
    x3 = (lam * lam - x1 - x2) % p
    return (x3, (lam * (x1 - x3) - y1) % p)


def ec_mul(k, P, a, p):
    R = None
    while k > 0:
        if k & 1:
            R = ec_add(R, P, a, p)
        P = ec_add(P, P, a, p)
        k >>= 1
    return R


def random_point(a, b, p, rng):
    """Requires p % 4 == 3 so that sqrt is a single modular exponentiation."""
    assert p % 4 == 3
    while True:
        x = rng.randrange(p)
        rhs = (x ** 3 + a * x + b) % p
        if rhs == 0 or pow(rhs, (p - 1) // 2, p) == 1:
            y = pow(rhs, (p + 1) // 4, p)
            if y * y % p == rhs:
                return (x, y)


def on_curve(P, a, b, p):
    return P is None or (P[1] ** 2 - (P[0] ** 3 + a * P[0] + b)) % p == 0
