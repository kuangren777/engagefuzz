"""One number formatter for every EngageFuzz generator.

F(x, spec) renders x under a fixed-point (".Nf") or significant-digit (".Ng") spec. It rounds
once, half up, through fmtnum.py. Count-based values are passed as
fractions.Fraction built from integers, so an exact .5 boundary such as 33/80 = 0.4125 rounds up.
Continuous statistics such as rho or p are floats and go through their shortest repr.
exact(x, den) recovers the Fraction k/den behind a count ratio stored as a float in a stage-1
JSON. It takes the best rational with denominator at most den and requires that rational to
convert back to the identical float, so it uses no float rounding and no tolerance.
"""
import math
import numbers
import os
import sys
from fractions import Fraction

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fmtnum import fmt, round_half_up, to_decimal  # noqa: E402


def exact(x, den):
    """Fraction k/den (den an int bound) whose float is exactly x."""
    if isinstance(x, (int, Fraction)):
        return Fraction(x)
    f = Fraction(x).limit_denominator(int(den))
    if float(f) != x:
        raise ValueError(f"{x!r} is not a ratio with denominator <= {den}")
    return f


def a12(t):
    """Exact Vargha-Delaney A12 of a stage-1 test record: ties count one half, so the denominator is 2 n_pairs."""
    return exact(t["A12"], 2 * t["n_pairs"])


def mean(xs):
    """Exact mean of integers or Fractions."""
    xs = list(xs)
    return Fraction(sum(Fraction(v) for v in xs), len(xs))


def F(x, spec):
    if isinstance(x, numbers.Integral) and not isinstance(x, int):
        x = int(x)                      # numpy integer
    elif isinstance(x, float) or (not isinstance(x, (int, Fraction)) and isinstance(x, numbers.Real)):
        x = float(x)                    # numpy float: repr must be the plain shortest repr
    kind, n = spec[-1], int(spec[spec.index(".") + 1:-1])   # "#" keeps trailing zeros under g
    if isinstance(x, float) and not math.isfinite(x):
        return format(x, spec)
    if kind == "f" and not spec.startswith("#"):
        return fmt(x, n)
    if kind != "g":
        raise ValueError(spec)
    if x == 0 or (isinstance(x, float) and not math.isfinite(x)):
        return format(float(x), spec)
    e = to_decimal(abs(x) if not isinstance(x, Fraction) else abs(x)).adjusted()
    r = round_half_up(x, n - 1 - e)
    if r != 0 and abs(r).adjusted() > e:   # carry, e.g. 9.96 -> 10.0 at two digits
        r = round_half_up(x, n - 2 - e)
    return format(float(r), spec)
