"""Task generators and exact verifiers. All instances are toy-scale and deterministic given a seed."""
from __future__ import annotations

import hashlib
import math
import random
import string
from dataclasses import dataclass

from . import mathutil as mu

TASKS = ("rsa_close_primes", "toy_ecdlp", "pin_key", "xor_repeating")
DIFFICULTIES = ("easy", "medium", "hard")


@dataclass
class Instance:
    task: str
    difficulty: str
    index: int
    base_seed: int
    public: dict
    secret: dict

    @property
    def instance_id(self) -> str:
        return f"{self.task}-{self.difficulty}-{self.index:03d}-s{self.base_seed}"

    def secret_answer(self) -> str:
        return str(self.secret["answer"])

    def to_json(self) -> dict:
        return {"instance_id": self.instance_id, "task": self.task, "difficulty": self.difficulty,
                "public": self.public, "secret": self.secret}


def _rng(task, difficulty, index, base_seed):
    h = hashlib.sha256(f"{task}|{difficulty}|{index}|{base_seed}".encode()).digest()
    return random.Random(int.from_bytes(h[:8], "big"))


def _xor(data: bytes, key: bytes) -> bytes:
    return bytes(c ^ key[i % len(key)] for i, c in enumerate(data))


# ---------------------------------------------------------------- RSA with close primes
_RSA = {"easy": (40, 2 ** 10), "medium": (64, 2 ** 16), "hard": (96, 2 ** 24)}  # (prime bits, max gap)


def _gen_rsa(rng, difficulty):
    bits, gap = _RSA[difficulty]
    e = 65537
    while True:
        p = mu.gen_prime(bits, rng)
        q = mu.next_prime(p + rng.randrange(2, gap))
        if math.gcd(e, (p - 1) * (q - 1)) == 1:
            break
    n = p * q
    m = rng.randrange(2 ** 31, 2 ** 32)
    return ({"n": n, "e": e, "c": pow(m, e, n), "bits_n": n.bit_length()},
            {"p": p, "q": q, "answer": m})


def _verify_rsa(inst, ans):
    try:
        return int(ans.strip().replace(",", "").replace("_", "")) == inst.secret["answer"]
    except ValueError:
        return False


# ---------------------------------------------------------------- toy ECDLP
_EC = {"easy": 20, "medium": 28, "hard": 36}  # prime bits


def _gen_ecdlp(rng, difficulty):
    bits = _EC[difficulty]
    p = mu.gen_prime(bits, rng, mod4=3)
    while True:
        a, b = rng.randrange(1, p), rng.randrange(1, p)
        if (4 * a ** 3 + 27 * b * b) % p == 0:
            continue
        G = mu.random_point(a, b, p, rng)
        k = rng.randrange(2, 2 ** (bits - 1))
        Q = mu.ec_mul(k, G, a, p)
        if Q is not None:
            return ({"p": p, "a": a, "b": b, "G": list(G), "Q": list(Q), "bits_p": bits},
                    {"answer": k})


def _verify_ecdlp(inst, ans):
    try:
        k = int(ans.strip().replace(",", "").replace("_", ""))
    except ValueError:
        return False
    pub = inst.public
    if not (1 <= k <= 2 ** (pub["bits_p"] + 2)):
        return False
    return mu.ec_mul(k, tuple(pub["G"]), pub["a"], pub["p"]) == tuple(pub["Q"])  # any valid k is accepted


# ---------------------------------------------------------------- PIN-derived keystream
_PIN = {"easy": 4, "medium": 5, "hard": 6}


def _keystream(salt: bytes, pin: str) -> bytes:
    return hashlib.sha256(salt + pin.encode()).digest()


def _gen_pin(rng, difficulty):
    d = _PIN[difficulty]
    pin = str(rng.randrange(10 ** d)).zfill(d)
    salt = bytes(rng.getrandbits(8) for _ in range(8))
    pt = "CH{" + "".join(rng.choice("0123456789abcdef") for _ in range(12)) + "}"
    ct = _xor(pt.encode(), _keystream(salt, pin))
    return ({"digits": d, "salt_hex": salt.hex(), "ciphertext_hex": ct.hex()},
            {"pin": pin, "plaintext": pt, "answer": pin})


def _verify_pin(inst, ans):
    pin = ans.strip().strip("'\"")
    d = inst.public["digits"]
    if not (pin.isdigit() and len(pin) == d):
        return False
    ct = bytes.fromhex(inst.public["ciphertext_hex"])
    return _xor(ct, _keystream(bytes.fromhex(inst.public["salt_hex"]), pin)).decode("latin-1") == inst.secret["plaintext"]


# ---------------------------------------------------------------- repeating-key XOR
_XOR_LEN = {"easy": 3, "medium": 6, "hard": 10}
_WORDS = ("the of and to in is that it was for on are as with his they at be this from have or by one had not but what "
          "all were when we there can an your which their said if do will each about how up out them then she many "
          "some so these would other into has more her two like him see time could no make than first been its who "
          "now people my made over did down only way find use may water long little very after words called just "
          "where most know river village morning window garden market bridge letter winter engine harbor lantern "
          "forest silver quiet distant careful machine ancient simple journey measure thought remember").split()


def _gen_xor(rng, difficulty):
    L = _XOR_LEN[difficulty]
    while True:
        key = "".join(rng.choice(string.ascii_lowercase) for _ in range(L))
        if all(key != key[:s] * (L // s) for s in range(1, L) if L % s == 0):
            break
    words = [rng.choice(_WORDS) for _ in range(rng.randrange(42, 52))]
    text = " ".join(words).capitalize() + "."
    ct = _xor(text.encode(), key.encode())
    return ({"ciphertext_hex": ct.hex(), "length_bytes": len(ct)},
            {"plaintext": text, "answer": key})


def _verify_xor(inst, ans):
    key = ans.strip().strip("'\"`")
    if not key or len(key) > 64 or not key.isascii():
        return False
    ct = bytes.fromhex(inst.public["ciphertext_hex"])
    return _xor(ct, key.encode()).decode("latin-1") == inst.secret["plaintext"]


_GEN = {"rsa_close_primes": _gen_rsa, "toy_ecdlp": _gen_ecdlp, "pin_key": _gen_pin, "xor_repeating": _gen_xor}
_VER = {"rsa_close_primes": _verify_rsa, "toy_ecdlp": _verify_ecdlp, "pin_key": _verify_pin, "xor_repeating": _verify_xor}


def make_instance(task: str, difficulty: str, index: int, base_seed: int = 1337) -> Instance:
    public, secret = _GEN[task](_rng(task, difficulty, index, base_seed), difficulty)
    return Instance(task, difficulty, index, base_seed, public, secret)


def verify(inst: Instance, answer: str) -> bool:
    try:
        return bool(_VER[inst.task](inst, answer))
    except Exception:
        return False
