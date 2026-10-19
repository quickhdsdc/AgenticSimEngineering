"""Shared CLI value types and wire-input validation."""

import argparse
import math


def positive_id(raw):
    value = int(raw)
    if value < 1:
        raise argparse.ArgumentTypeError("ID must be positive")
    return value


def quantity(raw):
    value = float(raw)
    if not math.isfinite(value) or not 0.01 <= value <= 99999999:
        raise argparse.ArgumentTypeError("Quantity must be between 0.01 and 99999999")
    return value
