r"""
data/test_latex_real.py — Real LaTeX Stripping Verification
STATUS: VERIFICATION SCRIPT — run and read the output

Tests strip_latex() with actual unescaped LaTeX as it appears in real
corpus text files. The previous test used escaped Python string literals
(r'\$') which contained literal backslash-dollar, NOT real dollar signs.
This file uses actual dollar signs to test real corpus behavior.

Run: python data/test_latex_real.py
"""

import sys
import textwrap

sys.path.insert(0, ".")
from data.prepare_corpus import strip_latex

# ---------------------------------------------------------------------------
# Real test cases — text is as it would appear in an actual .txt file
# ---------------------------------------------------------------------------

CASES = [
    # ── Inline math ────────────────────────────────────────────────────────
    {
        "name": "Inline: E=mc^2",
        "input": "The equation $E = mc^2$ was derived by Einstein in 1905.",
    },
    {
        "name": "Inline: fraction",
        "input": "We know that $\\frac{a}{b} + \\frac{c}{d} = \\frac{ad+bc}{bd}$ for non-zero denominators.",
    },
    {
        "name": "Inline: multiple in one sentence",
        "input": "If $x > 0$ and $y < 0$, then $xy < 0$ by the sign rule.",
    },
    {
        "name": "Inline: subscript/superscript",
        "input": "The sum $\\sum_{i=1}^{n} i$ equals $\\frac{n(n+1)}{2}$.",
    },
    # ── Display math ($$...$$) ─────────────────────────────────────────────
    {
        "name": "Display: summation formula",
        "input": (
            "The closed form is:\n"
            "$$\\sum_{i=1}^{n} i = \\frac{n(n+1)}{2}$$\n"
            "This is Gauss's formula."
        ),
    },
    {
        "name": "Display: integral",
        "input": (
            "The area under the curve is:\n"
            "$$\\int_0^1 x^2\\, dx = \\frac{1}{3}$$"
        ),
    },
    # ── LaTeX environments ─────────────────────────────────────────────────
    {
        "name": "Environment: equation",
        "input": (
            "Newton's second law:\n"
            "\\begin{equation}\n"
            "F = ma\n"
            "\\end{equation}\n"
            "where m is mass and a is acceleration."
        ),
    },
    {
        "name": "Environment: align",
        "input": (
            "Solving the system:\n"
            "\\begin{align}\n"
            "  x + y &= 5 \\\\\n"
            "  x - y &= 1\n"
            "\\end{align}\n"
            "Adding both equations gives 2x = 6, so x = 3."
        ),
    },
    # ── Mixed prose + math (reasoning corpus pattern) ──────────────────────
    {
        "name": "Reasoning: word problem with inline math",
        "input": (
            "Step 1: Let the number of apples be $n$. "
            "If Alice starts with $n = 12$ apples and gives away $4$, "
            "she has $12 - 4 = 8$ apples remaining.\n\n"
            "Step 2: Verify: $8 + 4 = 12$. Check."
        ),
    },
    {
        "name": "Reasoning: fraction arithmetic",
        "input": (
            "To add $\\frac{1}{2} + \\frac{1}{3}$, we find a common denominator.\n"
            "LCD = 6, so $\\frac{1}{2} = \\frac{3}{6}$ and $\\frac{1}{3} = \\frac{2}{6}$.\n"
            "Therefore $\\frac{1}{2} + \\frac{1}{3} = \\frac{5}{6}$."
        ),
    },
    # ── LaTeX display math with \[...\] ────────────────────────────────────
    {
        "name": "Display: \\[...\\] block",
        "input": (
            "The quadratic formula:\n"
            "\\[\n"
            "  x = \\frac{-b \\pm \\sqrt{b^2 - 4ac}}{2a}\n"
            "\\]\n"
            "has two solutions when the discriminant $b^2 - 4ac > 0$."
        ),
    },
    # ── LaTeX commands without math delimiters ─────────────────────────────
    {
        "name": "Standalone LaTeX commands (no delimiters)",
        "input": "The \\textbf{important} result is \\emph{universally} accepted.",
    },
    {
        "name": "Curly braces",
        "input": "Let {x, y, z} \\subset \\mathbb{R} be a set of real numbers.",
    },
    # ── Edge cases ─────────────────────────────────────────────────────────
    {
        "name": "Dollar sign in non-math context (price)",
        "input": "The book costs $12 and the pen costs $3, for a total of $15.",
    },
    {
        "name": "Empty/no LaTeX",
        "input": "This sentence has no mathematical notation at all.",
    },
    {
        "name": "Multi-paragraph reasoning doc",
        "input": textwrap.dedent("""\
            Problem: A train travels at 60 mph for $t$ hours and covers distance $d = 60t$ miles.

            Solution:
            Step 1: Write the formula $d = r \\times t$ where $r = 60$ mph.
            Step 2: Substitute: $d = 60 \\times 2 = 120$ miles.

            \\begin{equation}
            d = 60t
            \\end{equation}

            Therefore the train covers 120 miles in 2 hours.
        """),
    },
]


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_tests() -> None:
    print("=" * 70)
    print("LATEX STRIPPING VERIFICATION — REAL UNESCAPED INPUT")
    print("=" * 70)
    print("Testing strip_latex() with actual dollar signs and LaTeX commands")
    print("as they appear in real corpus text files (not Python escaped strings)")
    print()

    issues = []

    for i, case in enumerate(CASES, 1):
        name = case["name"]
        raw = case["input"]
        cleaned = strip_latex(raw)

        # Detect potential issues
        flags = []
        if "$" in cleaned:
            # Dollar signs remaining — check if they're lone price-context $
            # (acceptable) vs. unstripped math delimiters ($ followed by letters)
            remaining_dollar_count = cleaned.count("$")
            if remaining_dollar_count > 0:
                import re
                math_like = re.findall(r'\$[a-zA-Z]', cleaned)
                if math_like:
                    flags.append(f"POSSIBLE UNSTRIPPED MATH: {math_like}")

        if "\\begin" in cleaned or "\\end" in cleaned:
            flags.append("UNSTRIPPED \\begin/\\end ENVIRONMENT")

        if "\\\\" in cleaned and "align" not in name.lower():
            flags.append("POSSIBLE UNSTRIPPED LaTeX DOUBLE-BACKSLASH")

        # Print result
        print(f"[{i:02d}] {name}")
        print(f"  INPUT    : {raw[:120]!r}")
        print(f"  STRIPPED : {cleaned[:120]!r}")
        if flags:
            for flag in flags:
                print(f"  [FLAG]   : {flag}")
            issues.append((name, flags))
        else:
            print(f"  STATUS   : OK")
        print()

    # Summary
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Cases tested: {len(CASES)}")
    print(f"  Issues found: {len(issues)}")
    if issues:
        print()
        print("  ISSUES REQUIRING REGEX FIX:")
        for name, flags in issues:
            print(f"    [{name}]")
            for f in flags:
                print(f"      {f}")
    else:
        print("  [OK] No unstripped LaTeX patterns detected")
    print()

    # Chinchilla note: verify [MATH] tokens appear where expected
    print("  [MATH] TOKEN COVERAGE CHECK:")
    math_cases = [c for c in CASES if "$" in c["input"] or "\\begin" in c["input"] or "\\[" in c["input"]]
    for case in math_cases:
        cleaned = strip_latex(case["input"])
        has_math_token = "[MATH]" in cleaned
        has_residual_dollar = "$" in cleaned
        status = "[OK]  " if (has_math_token or not has_residual_dollar or case["name"].startswith("Dollar sign")) else "[FLAG]"
        print(f"    {status} {case['name']}: [MATH]={'YES' if has_math_token else 'NO'}, residual_$={'YES' if has_residual_dollar else 'NO'}")

    print()
    print("RAW OUTPUT COMPLETE — read the above before claiming LaTeX stripping works.")


if __name__ == "__main__":
    run_tests()
