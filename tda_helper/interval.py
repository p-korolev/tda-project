from __future__ import annotations
import numpy as np

from numbers import Real
from typing import Union, List, Tuple

class ClosedInterval:
    def __init__(self, lower: Real, upper: Real):
        if lower>upper:
            raise ValueError("Lower bound cannot exceed upper bound.")
        if lower==upper:
            raise ValueError("Lower and upper bounds cannot be equivalent.")
        self.lower = np.float64(lower)
        self.upper = np.float64(upper)
    
    def tolist(self) -> List[Real, Real]:
        return [self.lower, self.upper]
    
    def totuple(self) -> Tuple[Real, Real]:
        return tuple(self.lower, self.upper)
    
    def __repr__(self):
        return f"Interval[{self.lower}, {self.upper}]"
    
    def __str__(self) -> str:
        return f"[{self.lower}, {self.upper}]"

    def contains(self, value: Real) -> bool:
        return (value >= self.lower and value<= self.upper)

    def get_min(self) -> Real:
        return self.lower
    
    def get_max(self) -> Real:
        return self.upper
    
    def build_cover(self, N_covers: int, overlap: float = 0.35) -> List[ClosedInterval]:
        width = (self.upper - self.lower)/(N_covers - (N_covers - 1)*overlap)
        step = width*(1 - overlap)

        intervals = []
        for i in range(N_covers):
            l = self.lower + i*step
            u = l + width
            intervals.append(ClosedInterval(lower=l, upper=u))
        return intervals 