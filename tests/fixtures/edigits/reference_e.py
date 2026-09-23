#!/usr/bin/env python3
"""Independent arbitrary-precision reference: e to N decimals, truncated.

Deliberately a different arithmetic path than the Rust fixture (decimal
fixed-point series with Decimal floor division vs base-1e9 limb series with
small-division truncation), so agreement is evidence, not a shared-bug artifact.
Stdlib `decimal` only; floats never touch the digits.

Usage: python3 reference_e.py N
"""
import sys
from decimal import Decimal, localcontext


def e_digits(n: int) -> str:
    # Guard so both the series tail and the final truncation are exact for the
    # printed digits.
    guard = 30
    with localcontext() as ctx:
        ctx.prec = (n + guard) * 2 + 10
        term = Decimal(10) ** (n + guard)
        # k=0 and k=1 both contribute the full scale (1/0! = 1/1! = 1).
        total = term * 2
        k = 2
        while True:
            term //= k
            if term == 0:
                break
            total += term
            k += 1
        integer = int(total) // 10**guard
    text = str(integer)
    return text[0] + "." + text[1:]


def main() -> int:
    if len(sys.argv) != 2 or not sys.argv[1].isdigit():
        print("usage: reference_e.py N", file=sys.stderr)
        return 2
    print(e_digits(int(sys.argv[1])))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
