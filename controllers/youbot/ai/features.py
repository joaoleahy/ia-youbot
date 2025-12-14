from dataclasses import dataclass
from typing import List, Dict


def downsample_lidar(ranges: List[float], out_dim: int = 60) -> List[float]:
    """
    Reduz um scan (lista) para tamanho fixo por média em janelas.
    Não depende de numpy.
    """
    n = len(ranges)
    if n == 0:
        return [0.0 for _ in range(out_dim)]
    out: List[float] = []
    for i in range(out_dim):
        a = int(i * n / out_dim)
        b = int((i + 1) * n / out_dim)
        if b <= a:
            b = min(n, a + 1)
        chunk = ranges[a:b]
        out.append(sum(chunk) / max(1, len(chunk)))
    return out


@dataclass(frozen=True)
class ColorHist:
    bins: int
    r: List[float]
    g: List[float]
    b: List[float]

    def as_dict(self) -> Dict:
        return {"bins": self.bins, "r": self.r, "g": self.g, "b": self.b}


def rgb_histogram(camera, step: int = 4, bins: int = 8) -> ColorHist:
    """
    Histograma RGB bem barato (para place recognition / assinatura do lugar).
    - step: subamostragem espacial
    - bins: quantização por canal
    """
    w = camera.getWidth()
    h = camera.getHeight()
    img = camera.getImage()
    if not img:
        return ColorHist(bins=bins, r=[0.0] * bins, g=[0.0] * bins, b=[0.0] * bins)

    hr = [0] * bins
    hg = [0] * bins
    hb = [0] * bins
    total = 0

    for y in range(0, h, step):
        for x in range(0, w, step):
            r = camera.imageGetRed(img, w, x, y)
            g = camera.imageGetGreen(img, w, x, y)
            b = camera.imageGetBlue(img, w, x, y)
            br = min(bins - 1, int(r * bins / 256))
            bg = min(bins - 1, int(g * bins / 256))
            bb = min(bins - 1, int(b * bins / 256))
            hr[br] += 1
            hg[bg] += 1
            hb[bb] += 1
            total += 1

    if total <= 0:
        return ColorHist(bins=bins, r=[0.0] * bins, g=[0.0] * bins, b=[0.0] * bins)

    inv = 1.0 / total
    return ColorHist(
        bins=bins,
        r=[v * inv for v in hr],
        g=[v * inv for v in hg],
        b=[v * inv for v in hb],
    )


