from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class CubeDetection:
    found: bool
    color: Optional[str]  # "RED" | "GREEN" | "BLUE"
    cx: float  # 0..1 (normalizado)
    cy: float  # 0..1
    area: int  # pixels contados (subamostrado)

@dataclass(frozen=True)
class ColorBlob:
    color: str
    cx: float
    cy: float
    area: int


def _is_red(r: int, g: int, b: int, r_min: int, other_max: int) -> bool:
    return r >= r_min and g <= other_max and b <= other_max


def _is_green(r: int, g: int, b: int, g_min: int, other_max: int) -> bool:
    return g >= g_min and r <= other_max and b <= other_max


def _is_blue(r: int, g: int, b: int, b_min: int, other_max: int) -> bool:
    return b >= b_min and r <= other_max and g <= other_max


def detect_cube_rgb(
    camera,
    step: int = 2,
    min_blob_pixels: int = 35,
    max_cube_pixels: int = 450,
    r_min: int = 170,
    g_min: int = 170,
    b_min: int = 170,
    other_max: int = 90,
) -> CubeDetection:
    """
    Detector simples por threshold em RGB (bom para cubos com baseColor “puro”).
    Retorna o maior blob (por contagem) dentre R/G/B.
    """
    w = camera.getWidth()
    h = camera.getHeight()
    img = camera.getImage()

    # Acumula centroides por cor
    acc = {
        "RED": [0, 0, 0],    # sum_x, sum_y, count
        "GREEN": [0, 0, 0],
        "BLUE": [0, 0, 0],
    }

    for y in range(0, h, step):
        for x in range(0, w, step):
            r = camera.imageGetRed(img, w, x, y)
            g = camera.imageGetGreen(img, w, x, y)
            b = camera.imageGetBlue(img, w, x, y)

            if _is_red(r, g, b, r_min, other_max):
                a = acc["RED"]
                a[0] += x
                a[1] += y
                a[2] += 1
            elif _is_green(r, g, b, g_min, other_max):
                a = acc["GREEN"]
                a[0] += x
                a[1] += y
                a[2] += 1
            elif _is_blue(r, g, b, b_min, other_max):
                a = acc["BLUE"]
                a[0] += x
                a[1] += y
                a[2] += 1

    # Seleciona maior blob dentro da faixa de “cubo”
    best_color = None
    best_count = 0
    for c in ("RED", "GREEN", "BLUE"):
        cnt = acc[c][2]
        if min_blob_pixels <= cnt <= max_cube_pixels and cnt > best_count:
            best_count = cnt
            best_color = c

    if best_color is None:
        # Retorna “não encontrado”, mas mantém o maior count como diagnóstico
        max_cnt = max(acc["RED"][2], acc["GREEN"][2], acc["BLUE"][2])
        return CubeDetection(found=False, color=None, cx=0.5, cy=0.5, area=max_cnt)

    sx, sy, cnt = acc[best_color]
    cx = (sx / cnt) / max(1, (w - 1))
    cy = (sy / cnt) / max(1, (h - 1))
    return CubeDetection(found=True, color=best_color, cx=cx, cy=cy, area=cnt)


def detect_bin_rgb(
    camera,
    step: int = 2,
    min_bin_pixels: int = 900,
    r_min: int = 170,
    g_min: int = 170,
    b_min: int = 170,
    other_max: int = 90,
) -> Optional[ColorBlob]:
    """
    Detecta “caixa” (bin) por cor: blobs bem grandes (área alta).
    Útil para confirmar que o robô está vendo a área de entrega.
    """
    w = camera.getWidth()
    h = camera.getHeight()
    img = camera.getImage()

    acc = {"RED": [0, 0, 0], "GREEN": [0, 0, 0], "BLUE": [0, 0, 0]}
    for y in range(0, h, step):
        for x in range(0, w, step):
            r = camera.imageGetRed(img, w, x, y)
            g = camera.imageGetGreen(img, w, x, y)
            b = camera.imageGetBlue(img, w, x, y)
            if _is_red(r, g, b, r_min, other_max):
                a = acc["RED"]; a[0] += x; a[1] += y; a[2] += 1
            elif _is_green(r, g, b, g_min, other_max):
                a = acc["GREEN"]; a[0] += x; a[1] += y; a[2] += 1
            elif _is_blue(r, g, b, b_min, other_max):
                a = acc["BLUE"]; a[0] += x; a[1] += y; a[2] += 1

    best_color = None
    best_count = 0
    for c in ("RED", "GREEN", "BLUE"):
        cnt = acc[c][2]
        if cnt >= min_bin_pixels and cnt > best_count:
            best_color = c
            best_count = cnt

    if best_color is None:
        return None
    sx, sy, cnt = acc[best_color]
    cx = (sx / cnt) / max(1, (w - 1))
    cy = (sy / cnt) / max(1, (h - 1))
    return ColorBlob(color=best_color, cx=cx, cy=cy, area=cnt)


def bearing_error_from_detection(det: CubeDetection) -> float:
    """
    Erro horizontal normalizado (-1..+1):
    - negativo: alvo está à esquerda
    - positivo: alvo está à direita
    """
    if not det.found:
        return 0.0
    return (det.cx - 0.5) * 2.0


