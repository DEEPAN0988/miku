"""
Natural Cursor Motion using Minimum-Jerk Trajectory Interpolation.
Core 3: Execution Engine (Hands)
PRD §6.2: Uses minimum-jerk trajectory polynomial so user can visually follow and interrupt.
"""
import time
from typing import List, Tuple

def minimum_jerk_trajectory(
    start: Tuple[int, int],
    end: Tuple[int, int],
    steps: int = 20
) -> List[Tuple[int, int]]:
    """
    Computes a minimum-jerk path between start (x0, y0) and end (x1, y1).
    Polynomial formula: s(tau) = 10*tau^3 - 15*tau^4 + 6*tau^5 for tau in [0, 1].
    Provides continuous position, velocity, and acceleration without jerks.
    """
    x0, y0 = start
    x1, y1 = end
    dx = x1 - x0
    dy = y1 - y0

    points = []
    for i in range(steps + 1):
        tau = i / float(steps)
        s = 10.0 * (tau ** 3) - 15.0 * (tau ** 4) + 6.0 * (tau ** 5)
        x = int(round(x0 + dx * s))
        y = int(round(y0 + dy * s))
        points.append((x, y))

    return points
