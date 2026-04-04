import ast
from typing import Callable

def _syntax_ok(code: str) -> bool:
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False

def _run_tests(code: str, test_cases: list) -> tuple[int, int]:
    namespace = {}
    try:
        exec(compile(code, "<submitted>", "exec"), namespace)
    except Exception:
        return 0, len(test_cases)
    fn = None
    for v in namespace.values():
        if callable(v) and not isinstance(v, type) and not isinstance(v, type(ast)):
            fn = v
            break
    if fn is None:
        return 0, len(test_cases)

    passed = 0
    for args, expected in test_cases:
        try:
            result = fn(*args) if isinstance(args, tuple) else fn(args)
            if result == expected:
                passed += 1
        except Exception:
            pass
    return passed, len(test_cases)


def _loop_nesting_depth(code: str) -> int:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return 0
    class DepthVisitor(ast.NodeVisitor):
        def __init__(self):
            self.max_depth = 0
            self._depth = 0

        def _enter_loop(self, node):
            self._depth += 1
            self.max_depth = max(self.max_depth, self._depth)
            self.generic_visit(node)
            self._depth -= 1

        visit_For = _enter_loop
        visit_While = _enter_loop

    v = DepthVisitor()
    v.visit(tree)
    return v.max_depth

TASK_EASY = {
    "id": "easy_sum_range",
    "difficulty": "easy",
    "context": (
        "Write a function `sum_range(n)` that returns the sum of all integers "
        "from 1 to n inclusive. "
        "For example: sum_range(5) → 15, sum_range(10) → 55, sum_range(0) → 0."
    ),
    "buggy_code": (
        "def sum_range(n):\n"
        "    total = 0\n"
        "    for i in range(n):   # BUG: should include n\n"
        "        total += i\n"
        "    return total"
    ),
    "canonical_solution": (
        "def sum_range(n):\n"
        "    total = 0\n"
        "    for i in range(1, n + 1):\n"
        "        total += i\n"
        "    return total"
    ),
    "test_cases": [
        ((1,),   1),
        ((5,),   15),
        ((10,),  55),
        ((0,),   0),
        ((100,), 5050),
    ],
}

def grade_easy(submitted_code: str) -> tuple[float, dict]:
    detail: dict = {}
    syntax = _syntax_ok(submitted_code)
    detail["syntax_ok"] = syntax
    if not syntax:
        detail["final_score"] = 0.0
        return 0.0, detail

    passed, total = _run_tests(submitted_code, TASK_EASY["test_cases"])
    detail["tests_passed"] = passed
    detail["tests_total"] = total
    correctness = passed / total

    hardcoded = any(str(v) in submitted_code for v in ["5050", "15", "55"])
    detail["hardcoded_answers"] = hardcoded
    quality = 0.0 if hardcoded else 1.0

    score = round(0.20 * 1.0 + 0.70 * correctness + 0.10 * quality, 4)
    detail["final_score"] = score
    return score, detail


TASK_MEDIUM = {
    "id": "medium_flatten_list",
    "difficulty": "medium",
    "context": (
        "Write a function `flatten(lst)` that takes a nested list (up to 2 levels deep) "
        "and returns a single flat list of all values. "
        "Examples: flatten([1, [2, 3]]) → [1, 2, 3], "
        "flatten([1, [2, [3]]]) → [1, 2, 3], "
        "flatten([[], [1, 2]]) → [1, 2]."
    ),
    "buggy_code": (
        "def flatten(lst):\n"
        "    result = []\n"
        "    for item in lst:\n"
        "        if type(item) == list:   # BUG 1: should use isinstance()\n"
        "            for sub in item:\n"
        "                result.append(sub)   # BUG 2: sub may itself be a list\n"
        "        else:\n"
        "            result.append(item)\n"
        "    return result"
    ),
    "canonical_solution": (
        "def flatten(lst):\n"
        "    result = []\n"
        "    for item in lst:\n"
        "        if isinstance(item, list):\n"
        "            for sub in item:\n"
        "                if isinstance(sub, list):\n"
        "                    result.extend(sub)\n"
        "                else:\n"
        "                    result.append(sub)\n"
        "        else:\n"
        "            result.append(item)\n"
        "    return result"
    ),
    "test_cases": [
        (([1, 2, 3],),          [1, 2, 3]),
        (([1, [2, 3]],),        [1, 2, 3]),
        (([1, [2, [3]]],),      [1, 2, 3]),
        (([],),                 []),
        (([[], [1, 2]],),       [1, 2]),
        (([1, [2, 3], [4, [5, 6]]],), [1, 2, 3, 4, 5, 6]),
    ],
}

def grade_medium(submitted_code: str) -> tuple[float, dict]:
    detail: dict = {}

    syntax = _syntax_ok(submitted_code)
    detail["syntax_ok"] = syntax
    if not syntax:
        detail["final_score"] = 0.0
        return 0.0, detail

    passed, total = _run_tests(submitted_code, TASK_MEDIUM["test_cases"])
    detail["tests_passed"] = passed
    detail["tests_total"] = total
    correctness = passed / total

    uses_isinstance = "isinstance" in submitted_code
    detail["uses_isinstance"] = uses_isinstance

    score = round(0.15 * 1.0 + 0.75 * correctness + 0.10 * (1.0 if uses_isinstance else 0.0), 4)
    detail["final_score"] = score
    return score, detail

TASK_HARD = {
    "id": "hard_longest_unique_substr",
    "difficulty": "hard",
    "context": (
        "Write a function `longest_unique(s)` that returns the length of the longest "
        "substring without any repeating characters. "
        "Examples: longest_unique('abcabcbb') → 3, longest_unique('bbbbb') → 1, "
        "longest_unique('') → 0. "
        "IMPORTANT: the solution MUST use an O(n) sliding window approach — "
        "not a brute-force O(n²) nested loop."
    ),
    "buggy_code": (
        "def longest_unique(s):\n"
        "    max_len = 0\n"
        "    for i in range(len(s)):          # O(n²) — outer loop\n"
        "        seen = set()\n"
        "        for j in range(i, len(s)):   # O(n²) — inner loop (must eliminate)\n"
        "            if s[j] in seen:\n"
        "                break\n"
        "            seen.add(s[j])\n"
        "        max_len = max(max_len, len(seen))\n"
        "    return max_len"
    ),
    "canonical_solution": (
        "def longest_unique(s):\n"
        "    char_index = {}\n"
        "    left = 0\n"
        "    max_len = 0\n"
        "    for right, ch in enumerate(s):\n"
        "        if ch in char_index and char_index[ch] >= left:\n"
        "            left = char_index[ch] + 1\n"
        "        char_index[ch] = right\n"
        "        max_len = max(max_len, right - left + 1)\n"
        "    return max_len"
    ),
    "test_cases": [
        (("abcabcbb",), 3),
        (("bbbbb",),    1),
        (("pwwkew",),   3),
        (("",),         0),
        (("a",),        1),
        (("abcdefg",),  7),
        (("dvdf",),     3),
        (("anviaj",),   5),
    ],
}

def grade_hard(submitted_code: str) -> tuple[float, dict]:
    detail: dict = {}

    syntax = _syntax_ok(submitted_code)
    detail["syntax_ok"] = syntax
    if not syntax:
        detail["final_score"] = 0.0
        return 0.0, detail

    passed, total = _run_tests(submitted_code, TASK_HARD["test_cases"])
    detail["tests_passed"] = passed
    detail["tests_total"] = total
    correctness = passed / total

    depth = _loop_nesting_depth(submitted_code)
    detail["loop_nesting_depth"] = depth
    is_linear = depth <= 1
    detail["likely_linear"] = is_linear

    score = round(0.10 * 1.0 + 0.70 * correctness + 0.20 * (1.0 if is_linear else 0.0), 4)
    detail["final_score"] = score
    return score, detail

TASKS: dict[str, tuple[dict, Callable]] = {
    "easy":   (TASK_EASY,   grade_easy),
    "medium": (TASK_MEDIUM, grade_medium),
    "hard":   (TASK_HARD,   grade_hard),
}
