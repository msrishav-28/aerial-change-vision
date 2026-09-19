from __future__ import annotations

from typing import Iterable

from aerial_change.schema import BLOCKED_CLASS_SUBSTR, CIVILIAN_CLASSES


class LeakageError(ValueError):
    pass


class PolicyError(ValueError):
    pass


def assert_disjoint(keys_a: Iterable[str], keys_b: Iterable[str], name: str = "split") -> None:
    overlap = set(keys_a) & set(keys_b)
    if overlap:
        raise LeakageError(f"{name} leakage: {sorted(overlap)[:8]}")


def assert_no_overlap_buffer(
    train_bounds: list[tuple[float, float, float, float]],
    test_bounds: list[tuple[float, float, float, float]],
    buffer: float,
) -> None:
    """bounds are (xmin, ymin, xmax, ymax) in the same CRS."""

    def expand(b, buf):
        return (b[0] - buf, b[1] - buf, b[2] + buf, b[3] + buf)

    def hits(a, b):
        return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])

    for tb in test_bounds:
        te = expand(tb, buffer)
        for tr in train_bounds:
            if hits(tr, te):
                raise LeakageError("train/test chip footprints overlap inside buffer")


def assert_civilian_label(label: str) -> None:
    low = str(label).lower()
    for bad in BLOCKED_CLASS_SUBSTR:
        if bad in low:
            raise PolicyError(f"blocked class token '{bad}' in label '{label}'")
    if low not in CIVILIAN_CLASSES and low not in {"0", "1"}:
        # allow unknown extras only if they are not blocked tokens
        return
