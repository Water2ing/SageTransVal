"""Small normalized benchmark catalog for the Python pilot experiment."""

from __future__ import annotations

from typing import Any


TaskSpec = tuple[str, str, str, list[dict[str, Any]]]


HUMANEVAL_PILOT_TASKS: list[TaskSpec] = [
    ("has_close_elements", "def solve(numbers: list[float], threshold: float) -> bool:\n    for i in range(len(numbers)):\n        for j in range(i + 1, len(numbers)):\n            if abs(numbers[i] - numbers[j]) < threshold:\n                return True\n    return False\n", "solve", [{"args": [[1.0, 2.0, 3.0], 0.5], "expected": False}, {"args": [[1.0, 1.2, 3.0], 0.3], "expected": True}]),
    ("separate_paren_groups", "def solve(paren_string: str) -> list[str]:\n    groups = []\n    depth = 0\n    current = []\n    for ch in paren_string.replace(' ', ''):\n        current.append(ch)\n        depth += 1 if ch == '(' else -1\n        if depth == 0 and current:\n            groups.append(''.join(current))\n            current = []\n    return groups\n", "solve", [{"args": ["(()) ()"], "expected": ["(())", "()"]}]),
    ("truncate_number", "def solve(number: float) -> float:\n    return number % 1.0\n", "solve", [{"args": [3.5], "expected": 0.5}, {"args": [1.0], "expected": 0.0}]),
    ("below_zero", "def solve(operations: list[int]) -> bool:\n    balance = 0\n    for op in operations:\n        balance += op\n        if balance < 0:\n            return True\n    return False\n", "solve", [{"args": [[1, -2, 1]], "expected": True}, {"args": [[1, 2, -2]], "expected": False}]),
    ("mean_absolute_deviation", "def solve(numbers: list[float]) -> float:\n    mean = sum(numbers) / len(numbers)\n    return sum(abs(x - mean) for x in numbers) / len(numbers)\n", "solve", [{"args": [[1.0, 2.0, 3.0]], "expected": 2.0 / 3.0}]),
    ("intersperse", "def solve(numbers: list[int], delimiter: int) -> list[int]:\n    out = []\n    for idx, number in enumerate(numbers):\n        if idx:\n            out.append(delimiter)\n        out.append(number)\n    return out\n", "solve", [{"args": [[1, 2, 3], 0], "expected": [1, 0, 2, 0, 3]}, {"args": [[], 9], "expected": []}]),
    ("parse_nested_parens", "def solve(paren_string: str) -> list[int]:\n    out = []\n    depth = 0\n    best = 0\n    for ch in paren_string:\n        if ch == '(':\n            depth += 1\n            best = max(best, depth)\n        elif ch == ')':\n            depth -= 1\n            if depth == 0:\n                out.append(best)\n                best = 0\n    return out\n", "solve", [{"args": ["(()()) ((()))"], "expected": [2, 3]}]),
    ("filter_by_substring", "def solve(strings: list[str], substring: str) -> list[str]:\n    return [s for s in strings if substring in s]\n", "solve", [{"args": [["abc", "def", "cab"], "ab"], "expected": ["abc"]}]),
    ("sum_product", "def solve(numbers: list[int]) -> list[int]:\n    total = 0\n    product = 1\n    for number in numbers:\n        total += number\n        product *= number\n    return [total, product]\n", "solve", [{"args": [[1, 2, 3]], "expected": [6, 6]}, {"args": [[]], "expected": [0, 1]}]),
    ("rolling_max", "def solve(numbers: list[int]) -> list[int]:\n    best = None\n    out = []\n    for number in numbers:\n        best = number if best is None else max(best, number)\n        out.append(best)\n    return out\n", "solve", [{"args": [[1, 3, 2, 5]], "expected": [1, 3, 3, 5]}]),
    ("is_palindrome", "def solve(text: str) -> bool:\n    return text == text[::-1]\n", "solve", [{"args": ["aba"], "expected": True}, {"args": ["abc"], "expected": False}]),
    ("string_xor", "def solve(a: str, b: str) -> str:\n    return ''.join('1' if x != y else '0' for x, y in zip(a, b))\n", "solve", [{"args": ["010", "110"], "expected": "100"}]),
    ("longest", "def solve(strings: list[str]) -> str | None:\n    if not strings:\n        return None\n    return max(strings, key=len)\n", "solve", [{"args": [["a", "abcd", "xy"]], "expected": "abcd"}, {"args": [[]], "expected": None}]),
    ("greatest_common_divisor", "def solve(a: int, b: int) -> int:\n    while b:\n        a, b = b, a % b\n    return abs(a)\n", "solve", [{"args": [12, 18], "expected": 6}]),
    ("all_prefixes", "def solve(text: str) -> list[str]:\n    return [text[:i] for i in range(len(text) + 1)]\n", "solve", [{"args": ["abc"], "expected": ["", "a", "ab", "abc"]}]),
    ("string_sequence", "def solve(n: int) -> str:\n    return ' '.join(str(i) for i in range(n + 1))\n", "solve", [{"args": [3], "expected": "0 1 2 3"}]),
    ("count_distinct_characters", "def solve(text: str) -> int:\n    return len(set(text.lower()))\n", "solve", [{"args": ["AaBb"], "expected": 2}]),
    ("parse_music", "def solve(music_string: str) -> list[int]:\n    mapping = {'o': 4, 'o|': 2, '.|': 1}\n    return [mapping[token] for token in music_string.split()]\n", "solve", [{"args": ["o o| .|"], "expected": [4, 2, 1]}]),
    ("how_many_times", "def solve(string: str, substring: str) -> int:\n    return sum(1 for i in range(len(string)) if string.startswith(substring, i))\n", "solve", [{"args": ["aaaa", "aa"], "expected": 3}]),
    ("sort_numbers", "def solve(numbers: str) -> str:\n    order = {'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5, 'six': 6, 'seven': 7, 'eight': 8, 'nine': 9}\n    return ' '.join(sorted(numbers.split(), key=order.get))\n", "solve", [{"args": ["three one two"], "expected": "one two three"}]),
]


MBPP_PILOT_TASKS: list[TaskSpec] = [
    ("mbpp_remove_duplicates", "def solve(items: list[int]) -> list[int]:\n    out = []\n    for item in items:\n        if item not in out:\n            out.append(item)\n    return out\n", "solve", [{"args": [[1, 2, 1, 3]], "expected": [1, 2, 3]}]),
    ("mbpp_count_vowels", "def solve(text: str) -> int:\n    return sum(1 for ch in text.lower() if ch in 'aeiou')\n", "solve", [{"args": ["banana"], "expected": 3}]),
    ("mbpp_is_prime", "def solve(n: int) -> bool:\n    if n < 2:\n        return False\n    for i in range(2, int(n ** 0.5) + 1):\n        if n % i == 0:\n            return False\n    return True\n", "solve", [{"args": [2], "expected": True}, {"args": [9], "expected": False}]),
    ("mbpp_fibonacci", "def solve(n: int) -> int:\n    a, b = 0, 1\n    for _ in range(n):\n        a, b = b, a + b\n    return a\n", "solve", [{"args": [7], "expected": 13}]),
    ("mbpp_reverse_words", "def solve(text: str) -> str:\n    return ' '.join(reversed(text.split()))\n", "solve", [{"args": ["hello world"], "expected": "world hello"}]),
    ("mbpp_merge_dicts", "def solve(a: dict[str, int], b: dict[str, int]) -> dict[str, int]:\n    out = dict(a)\n    for key, value in b.items():\n        out[key] = out.get(key, 0) + value\n    return out\n", "solve", [{"args": [{"a": 1}, {"a": 2, "b": 3}], "expected": {"a": 3, "b": 3}}]),
    ("mbpp_flatten", "def solve(items: list[list[int]]) -> list[int]:\n    return [x for group in items for x in group]\n", "solve", [{"args": [[[1, 2], [], [3]]], "expected": [1, 2, 3]}]),
    ("mbpp_second_largest", "def solve(items: list[int]) -> int | None:\n    unique = sorted(set(items))\n    return unique[-2] if len(unique) >= 2 else None\n", "solve", [{"args": [[1, 3, 2, 3]], "expected": 2}]),
    ("mbpp_anagram", "def solve(a: str, b: str) -> bool:\n    return sorted(a.replace(' ', '').lower()) == sorted(b.replace(' ', '').lower())\n", "solve", [{"args": ["listen", "silent"], "expected": True}]),
    ("mbpp_matrix_trace", "def solve(matrix: list[list[int]]) -> int:\n    return sum(matrix[i][i] for i in range(min(len(matrix), len(matrix[0]) if matrix else 0)))\n", "solve", [{"args": [[[1, 2], [3, 4]]], "expected": 5}]),
    ("mbpp_chunk", "def solve(items: list[int], size: int) -> list[list[int]]:\n    return [items[i:i + size] for i in range(0, len(items), size)]\n", "solve", [{"args": [[1, 2, 3, 4, 5], 2], "expected": [[1, 2], [3, 4], [5]]}]),
    ("mbpp_remove_none", "def solve(items: list[object]) -> list[object]:\n    return [item for item in items if item is not None]\n", "solve", [{"args": [[1, None, 2]], "expected": [1, 2]}]),
    ("mbpp_sum_even", "def solve(items: list[int]) -> int:\n    return sum(x for x in items if x % 2 == 0)\n", "solve", [{"args": [[1, 2, 4, 5]], "expected": 6}]),
    ("mbpp_decimal_to_binary", "def solve(n: int) -> str:\n    return bin(n)[2:]\n", "solve", [{"args": [10], "expected": "1010"}]),
    ("mbpp_capitalize_words", "def solve(text: str) -> str:\n    return ' '.join(word.capitalize() for word in text.split())\n", "solve", [{"args": ["hello world"], "expected": "Hello World"}]),
    ("mbpp_intersection", "def solve(a: list[int], b: list[int]) -> list[int]:\n    return [x for x in a if x in b]\n", "solve", [{"args": [[1, 2, 3], [2, 4, 1]], "expected": [1, 2]}]),
    ("mbpp_rotate_left", "def solve(items: list[int], k: int) -> list[int]:\n    if not items:\n        return []\n    k %= len(items)\n    return items[k:] + items[:k]\n", "solve", [{"args": [[1, 2, 3], 1], "expected": [2, 3, 1]}]),
    ("mbpp_word_lengths", "def solve(text: str) -> dict[str, int]:\n    return {word: len(word) for word in text.split()}\n", "solve", [{"args": ["a bee"], "expected": {"a": 1, "bee": 3}}]),
    ("mbpp_closest_pair", "def solve(items: list[int], target: int) -> list[int] | None:\n    best = None\n    best_dist = None\n    for i in range(len(items)):\n        for j in range(i + 1, len(items)):\n            dist = abs(items[i] + items[j] - target)\n            if best_dist is None or dist < best_dist:\n                best = [items[i], items[j]]\n                best_dist = dist\n    return best\n", "solve", [{"args": [[1, 4, 7], 8], "expected": [1, 7]}]),
    ("mbpp_balanced_brackets", "def solve(text: str) -> bool:\n    depth = 0\n    for ch in text:\n        if ch == '(':\n            depth += 1\n        elif ch == ')':\n            depth -= 1\n            if depth < 0:\n                return False\n    return depth == 0\n", "solve", [{"args": ["(())"], "expected": True}, {"args": ["(()"], "expected": False}]),
]


BUGSINPY_PILOT_TASKS: list[TaskSpec] = [
    ("bugsinpy_off_by_one_factorial", "def solve(n: int) -> int:\n    out = 1\n    for i in range(1, n):\n        out *= i\n    return out\n", "solve", [{"args": [4], "expected": 24}]),
    ("bugsinpy_empty_first_char", "def solve(text: str) -> str:\n    return text[0]\n", "solve", [{"args": [""], "expected": ""}]),
    ("bugsinpy_dedupe_order", "def solve(items: list[int]) -> list[int]:\n    return list(set(items))\n", "solve", [{"args": [[2, 1, 2]], "expected": [2, 1]}]),
    ("bugsinpy_middle_empty", "def solve(items: list[int]) -> int:\n    return items[len(items) // 2]\n", "solve", [{"args": [[]], "expected": 0}]),
    ("bugsinpy_negative_clamp", "def solve(x: int) -> int:\n    return 10 if x > 10 else x\n", "solve", [{"args": [-3], "expected": 0}]),
    ("bugsinpy_rounding_division", "def solve(a: int, b: int) -> int:\n    return a / b\n", "solve", [{"args": [5, 2], "expected": 2}]),
    ("bugsinpy_case_sensitive_count", "def solve(text: str) -> int:\n    return text.count('a')\n", "solve", [{"args": ["Aardvark"], "expected": 3}]),
    ("bugsinpy_sorted_desc", "def solve(items: list[int]) -> list[int]:\n    return sorted(items, reverse=True)\n", "solve", [{"args": [[3, 1, 2]], "expected": [1, 2, 3]}]),
    ("bugsinpy_palindrome_spaces", "def solve(text: str) -> bool:\n    return text == text[::-1]\n", "solve", [{"args": ["nurses run"], "expected": True}]),
    ("bugsinpy_zero_is_positive", "def solve(x: int) -> str:\n    return 'positive' if x >= 0 else 'negative'\n", "solve", [{"args": [0], "expected": "zero"}]),
]


STRESS_PILOT_TASKS: list[TaskSpec] = [
    ("boundary_zero_division", "def solve(x: int) -> int:\n    return 10 // x\n", "solve", [{"args": [2], "expected": 5}]),
    ("boundary_empty_string", "def solve(text: str) -> str:\n    return text[0]\n", "solve", [{"args": ["abc"], "expected": "a"}]),
    ("boundary_empty_list", "def solve(items: list[int]) -> int:\n    return items[0]\n", "solve", [{"args": [[1]], "expected": 1}]),
    ("fuzz_large_square", "def solve(x: int) -> int:\n    return x * x if abs(x) < 20 else -1\n", "solve", [{"args": [3], "expected": 9}]),
    ("fuzz_repeated_items", "def solve(items: list[int]) -> int:\n    return len(set(items))\n", "solve", [{"args": [[1, 2, 3]], "expected": 3}]),
    ("fuzz_long_string", "def solve(text: str) -> bool:\n    return text.islower()\n", "solve", [{"args": ["abc"], "expected": True}]),
    ("assertion_rounding", "def solve(a: int, b: int) -> int:\n    return a / b\n", "solve", [{"args": [4, 2], "expected": 2}]),
    ("assertion_casefold", "def solve(text: str) -> int:\n    return text.count('a')\n", "solve", [{"args": ["banana"], "expected": 3}]),
    ("counterexample_factorial", "def solve(n: int) -> int:\n    out = 1\n    for i in range(1, n):\n        out *= i\n    return out\n", "solve", [{"args": [2], "expected": 2}]),
    ("counterexample_clamp", "def solve(x: int) -> int:\n    return x if x <= 10 else 10\n", "solve", [{"args": [-5], "expected": 0}]),
    ("clean_sum", "def solve(items: list[int]) -> int:\n    return sum(items)\n", "solve", [{"args": [[1, 2]], "expected": 3}]),
    ("clean_palindrome", "def solve(text: str) -> bool:\n    return text == text[::-1]\n", "solve", [{"args": ["aba"], "expected": True}, {"args": ["abc"], "expected": False}]),
]


ACTION_BALANCED_PILOT_TASKS: list[TaskSpec] = [
    ("boundary_mod_zero", "def solve(x: int) -> int:\n    return 100 % x\n", "solve", [{"args": [4], "expected": 0}]),
    ("boundary_first_word", "def solve(text: str) -> str:\n    return text.split()[0]\n", "solve", [{"args": ["hello world"], "expected": "hello"}]),
    ("fuzz_overflow_len", "def solve(items: list[int]) -> int:\n    return len(items) if len(items) < 3 else -1\n", "solve", [{"args": [[1, 2]], "expected": 2}]),
    ("fuzz_threshold_total", "def solve(items: list[int]) -> bool:\n    return sum(items) < 10\n", "solve", [{"args": [[1, 2]], "expected": True}]),
    ("llm_assert_sorted_abs", "def solve(items: list[int]) -> list[int]:\n    return sorted(items)\n", "solve", [{"args": [[1, 2]], "expected": [1, 2]}]),
    ("llm_assert_strip_case", "def solve(text: str) -> str:\n    return text.strip().lower()\n", "solve", [{"args": [" abc "], "expected": "abc"}]),
    ("counterexample_gcd_zero", "def solve(a: int, b: int) -> int:\n    while b:\n        a, b = b, a % b\n    return a\n", "solve", [{"args": [6, 3], "expected": 3}]),
    ("counterexample_rotate_empty", "def solve(items: list[int], k: int) -> list[int]:\n    k %= len(items)\n    return items[k:] + items[:k]\n", "solve", [{"args": [[1, 2, 3], 1], "expected": [2, 3, 1]}]),
    ("clean_clamp", "def solve(x: int) -> int:\n    return 0 if x < 0 else (10 if x > 10 else x)\n", "solve", [{"args": [-2], "expected": 0}, {"args": [12], "expected": 10}]),
    ("clean_join", "def solve(items: list[str]) -> str:\n    return ','.join(items)\n", "solve", [{"args": [["a", "b"]], "expected": "a,b"}]),
    ("stop_noop_identity", "def solve(x: int) -> int:\n    return x\n", "solve", [{"args": [3], "expected": 3}]),
    ("stop_noop_reverse", "def solve(text: str) -> str:\n    return text[::-1]\n", "solve", [{"args": ["abc"], "expected": "cba"}]),
]
