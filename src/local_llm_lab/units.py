from __future__ import annotations

import re
from math import isfinite


GIB = 1024**3


def parse_params(value: str | int | float) -> float:
    """Return parameter count in billions."""
    result: float
    if isinstance(value, (int, float)):
        if isinstance(value, bool):
            raise ValueError(f"Invalid parameter count: {value!r}")
        raw = float(value)
        result = raw / 1_000_000_000 if raw > 10_000 else raw
    else:
        text = value.strip().lower().replace("_", "").replace(" ", "")
        match = re.fullmatch(r"([0-9]+(?:\.[0-9]+)?)([bmk]?)", text)
        if not match:
            raise ValueError(f"Invalid parameter count: {value!r}")
        number = float(match.group(1))
        suffix = match.group(2)
        if suffix == "b" or suffix == "":
            result = number
        elif suffix == "m":
            result = number / 1000
        elif suffix == "k":
            result = number / 1_000_000
        else:
            raise ValueError(f"Invalid parameter suffix: {value!r}")

    if not isfinite(result) or result <= 0:
        raise ValueError(f"Parameter count must be greater than 0: {value!r}")
    return result


def parse_gib(value: str | int | float) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    text = value.strip().lower().replace(" ", "")
    match = re.fullmatch(r"([0-9]+(?:\.[0-9]+)?)(gib|gb|mib|mb|tb|t)?", text)
    if not match:
        raise ValueError(f"Invalid size: {value!r}")
    number = float(match.group(1))
    suffix = match.group(2) or "gb"
    if suffix in {"gb", "gib"}:
        return number
    if suffix in {"mb", "mib"}:
        return number / 1024
    if suffix in {"tb", "t"}:
        return number * 1024
    raise ValueError(f"Invalid size suffix: {value!r}")


def format_gib(value: float) -> str:
    if abs(value) >= 100:
        return f"{value:.0f} GiB"
    if abs(value) >= 10:
        return f"{value:.1f} GiB"
    return f"{value:.2f} GiB"


def format_params(value_b: float) -> str:
    if value_b >= 100:
        return f"{value_b:.0f}B"
    if value_b >= 10:
        return f"{value_b:.1f}B"
    return f"{value_b:.2f}B"
