"""Curated translation-validation subjects for the Proposal 1 prototype.

The subjects are small Python functions paired with target candidates that
exhibit common translation faults. They are not presented as a cross-language
benchmark; they are a reproducible stress benchmark for the validation layer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


Case = dict[str, Any]


@dataclass(frozen=True)
class TranslationSubject:
    id: str
    source: str
    target: str
    entrypoint: str
    seed_tests: list[Case]
    random_cases: list[Case]
    llm_cases: list[Case]
    obligation_cases: dict[str, list[Case]]
    bug_class: str


def _case(args: list[Any]) -> Case:
    return {"args": args}


SUBJECTS: list[TranslationSubject] = [
    TranslationSubject(
        "abs_sign",
        "def solve(x):\n    return abs(x)\n",
        "def solve(x):\n    return x\n",
        "solve",
        [_case([3])],
        [_case([0]), _case([1]), _case([-1])],
        [_case([-5]), _case([7])],
        {"boundary/type": [_case([-3]), _case([0])]},
        "boundary",
    ),
    TranslationSubject(
        "factorial_loop",
        "def solve(n):\n    out = 1\n    for i in range(1, n + 1):\n        out *= i\n    return out\n",
        "def solve(n):\n    out = 1\n    for i in range(1, n):\n        out *= i\n    return out\n",
        "solve",
        [_case([1])],
        [_case([0]), _case([2]), _case([4])],
        [_case([5])],
        {"control flow": [_case([3]), _case([5])], "boundary/type": [_case([0])]},
        "off-by-one",
    ),
    TranslationSubject(
        "safe_div_zero",
        "def solve(a, b):\n    return 0 if b == 0 else a // b\n",
        "def solve(a, b):\n    return a // b\n",
        "solve",
        [_case([4, 2])],
        [_case([1, 1]), _case([4, 0]), _case([-3, 2])],
        [_case([9, 0])],
        {"exceptions": [_case([4, 0])], "boundary/type": [_case([0, 0])]},
        "exception",
    ),
    TranslationSubject(
        "clamp_lower",
        "def solve(x):\n    return 0 if x < 0 else (10 if x > 10 else x)\n",
        "def solve(x):\n    return 10 if x > 10 else x\n",
        "solve",
        [_case([5]), _case([11])],
        [_case([-1]), _case([0]), _case([12])],
        [_case([-5])],
        {"boundary/type": [_case([-2]), _case([10]), _case([11])]},
        "boundary",
    ),
    TranslationSubject(
        "first_char_empty",
        "def solve(s):\n    return '' if s == '' else s[0]\n",
        "def solve(s):\n    return s[0]\n",
        "solve",
        [_case(["abc"])],
        [_case([""]), _case(["a"]), _case(["xyz"])],
        [_case([""])],
        {"exceptions": [_case([""])], "boundary/type": [_case([""])]},
        "exception",
    ),
    TranslationSubject(
        "middle_empty",
        "def solve(xs):\n    return 0 if not xs else xs[len(xs)//2]\n",
        "def solve(xs):\n    return xs[len(xs)//2]\n",
        "solve",
        [_case([[1, 2, 3]])],
        [_case([[]]), _case([[0]]), _case([[1, 2]])],
        [_case([[]])],
        {"exceptions": [_case([[]])], "boundary/type": [_case([[]])]},
        "exception",
    ),
    TranslationSubject(
        "count_a_case",
        "def solve(s):\n    return s.lower().count('a')\n",
        "def solve(s):\n    return s.count('a')\n",
        "solve",
        [_case(["banana"])],
        [_case(["Aardvark"]), _case(["abc"]), _case(["XYZ"])],
        [_case(["Aardvark"])],
        {"api contracts": [_case(["Aardvark"])], "boundary/type": [_case(["A"])]},
        "api",
    ),
    TranslationSubject(
        "gcd_negative",
        "def solve(a, b):\n    import math\n    return abs(math.gcd(a, b))\n",
        "def solve(a, b):\n    import math\n    return math.gcd(a, b) if a >= 0 else -math.gcd(a, b)\n",
        "solve",
        [_case([6, 3])],
        [_case([-6, 3]), _case([0, 0]), _case([4, 2])],
        [_case([-6, 3])],
        {"boundary/type": [_case([-6, 3]), _case([0, 0])]},
        "numeric",
    ),
    TranslationSubject(
        "rotate_empty",
        "def solve(xs, k):\n    if not xs:\n        return []\n    k %= len(xs)\n    return xs[k:] + xs[:k]\n",
        "def solve(xs, k):\n    k %= len(xs)\n    return xs[k:] + xs[:k]\n",
        "solve",
        [_case([[1, 2, 3], 1])],
        [_case([[], 3]), _case([[1], 0]), _case([[1, 2], 3])],
        [_case([[], 1])],
        {"exceptions": [_case([[], 3])], "boundary/type": [_case([[], 0])]},
        "exception",
    ),
    TranslationSubject(
        "round_half",
        "def solve(a, b):\n    return round(a / b)\n",
        "def solve(a, b):\n    return a // b\n",
        "solve",
        [_case([4, 2])],
        [_case([5, 2]), _case([3, 2]), _case([10, 3])],
        [_case([5, 2])],
        {"boundary/type": [_case([5, 3]), _case([1, 2])]},
        "numeric",
    ),
    TranslationSubject(
        "sum_abs",
        "def solve(xs):\n    return sum(abs(x) for x in xs)\n",
        "def solve(xs):\n    return sum(xs)\n",
        "solve",
        [_case([[1, 2, 3]])],
        [_case([[-1, 2]]), _case([[]]), _case([[3]])],
        [_case([[-2, 2]])],
        {"data flow/state": [_case([[-1, 2, -3]])], "boundary/type": [_case([[]])]},
        "data-flow",
    ),
    TranslationSubject(
        "is_prime_one",
        "def solve(n):\n    if n < 2:\n        return False\n    return all(n % d for d in range(2, int(n ** 0.5) + 1))\n",
        "def solve(n):\n    if n < 1:\n        return False\n    return all(n % d for d in range(2, int(n ** 0.5) + 1))\n",
        "solve",
        [_case([2]), _case([3])],
        [_case([1]), _case([4]), _case([9])],
        [_case([1])],
        {"control flow": [_case([1]), _case([0])], "boundary/type": [_case([1])]},
        "boundary",
    ),
    TranslationSubject(
        "fib_zero",
        "def solve(n):\n    if n <= 0:\n        return 0\n    a, b = 0, 1\n    for _ in range(n):\n        a, b = b, a + b\n    return a\n",
        "def solve(n):\n    a, b = 0, 1\n    for _ in range(1, n):\n        a, b = b, a + b\n    return b\n",
        "solve",
        [_case([1]), _case([2])],
        [_case([0]), _case([3]), _case([5])],
        [_case([0])],
        {"control flow": [_case([0]), _case([5])], "boundary/type": [_case([0])]},
        "off-by-one",
    ),
    TranslationSubject(
        "parse_empty",
        "def solve(s):\n    return 0 if s.strip() == '' else int(s)\n",
        "def solve(s):\n    return int(s)\n",
        "solve",
        [_case(["5"])],
        [_case([""]), _case(["  "]), _case(["-2"])],
        [_case([""])],
        {"exceptions": [_case([""])], "api contracts": [_case(["  "])]},
        "exception",
    ),
    TranslationSubject(
        "normalize_spaces",
        "def solve(s):\n    return ' '.join(s.split())\n",
        "def solve(s):\n    return s.strip()\n",
        "solve",
        [_case(["abc"])],
        [_case(["a  b"]), _case(["  a "]), _case([""])],
        [_case(["a   b"])],
        {"api contracts": [_case(["a   b"])], "boundary/type": [_case([""])]},
        "api",
    ),
    TranslationSubject(
        "starts_alpha",
        "def solve(s):\n    return bool(s) and s[0].isalpha()\n",
        "def solve(s):\n    return s[0].isalpha()\n",
        "solve",
        [_case(["abc"])],
        [_case([""]), _case(["1a"]), _case(["A"])],
        [_case([""])],
        {"exceptions": [_case([""])], "api contracts": [_case(["1a"])]},
        "exception",
    ),
]
