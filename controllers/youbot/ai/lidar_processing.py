import math
from dataclasses import dataclass
from typing import List, Tuple


def _clip(v: float, lo: float, hi: float) -> float:
    return hi if v > hi else lo if v < lo else v


@dataclass(frozen=True)
class LidarSectors:
    left_min: float
    front_min: float
    right_min: float


@dataclass(frozen=True)
class LidarSectors5:
    left_min: float
    front_left_min: float
    front_min: float
    front_right_min: float
    right_min: float


def sanitize_ranges(ranges: List[float], clip_min: float, clip_max: float) -> List[float]:
    out: List[float] = []
    for r in ranges:
        if r is None or math.isnan(r) or math.isinf(r):
            out.append(clip_max)
        else:
            out.append(_clip(r, clip_min, clip_max))
    return out


def sector_mins(ranges: List[float], front_frac: float = 0.20, side_frac: float = 0.25) -> LidarSectors:
    """
    Divide o scan (vetor circular) em 3 setores (esq/frente/dir) e retorna o mínimo em cada.
    Assumimos que o “frente” está no meio do vetor (padrão comum no Webots para Lidar 2D).
    Se isso não bater, ajustamos depois (é 1 linha para rotacionar o vetor).
    """
    n = len(ranges)
    if n == 0:
        return LidarSectors(left_min=999.0, front_min=999.0, right_min=999.0)

    mid = n // 2
    front_w = max(1, int(n * front_frac))
    side_w = max(1, int(n * side_frac))

    front = ranges[mid - front_w // 2 : mid + (front_w - front_w // 2)]
    left = ranges[mid + front_w // 2 : min(n, mid + front_w // 2 + side_w)]
    right = ranges[max(0, mid - front_w // 2 - side_w) : mid - front_w // 2]

    def mn(xs: List[float]) -> float:
        return min(xs) if xs else 999.0

    return LidarSectors(left_min=mn(left), front_min=mn(front), right_min=mn(right))


def sector_mins_5(ranges: List[float], front_frac: float = 0.14, flank_frac: float = 0.10, side_frac: float = 0.18) -> LidarSectors5:
    """
    5 setores: left, front-left, front, front-right, right.
    Útil para detectar quinas (front-left e front-right caem juntos) e manter-se afastado de paredes.
    """
    n = len(ranges)
    if n == 0:
        return LidarSectors5(left_min=999.0, front_left_min=999.0, front_min=999.0, front_right_min=999.0, right_min=999.0)

    mid = n // 2
    front_w = max(1, int(n * front_frac))
    flank_w = max(1, int(n * flank_frac))
    side_w = max(1, int(n * side_frac))

    front = ranges[mid - front_w // 2 : mid + (front_w - front_w // 2)]
    front_left = ranges[mid + front_w // 2 : min(n, mid + front_w // 2 + flank_w)]
    front_right = ranges[max(0, mid - front_w // 2 - flank_w) : mid - front_w // 2]
    left = ranges[mid + front_w // 2 + flank_w : min(n, mid + front_w // 2 + flank_w + side_w)]
    right = ranges[max(0, mid - front_w // 2 - flank_w - side_w) : max(0, mid - front_w // 2 - flank_w)]

    def mn(xs: List[float]) -> float:
        return min(xs) if xs else 999.0

    return LidarSectors5(
        left_min=mn(left),
        front_left_min=mn(front_left),
        front_min=mn(front),
        front_right_min=mn(front_right),
        right_min=mn(right),
    )


