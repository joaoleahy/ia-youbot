from dataclasses import dataclass

import math


def tri(x: float, a: float, b: float, c: float) -> float:
    """Função triangular (a <= b <= c)."""
    if x <= a or x >= c:
        return 0.0
    if x == b:
        return 1.0
    if x < b:
        return (x - a) / (b - a) if b != a else 0.0
    return (c - x) / (c - b) if c != b else 0.0


def trap(x: float, a: float, b: float, c: float, d: float) -> float:
    """Função trapezoidal (a <= b <= c <= d)."""
    if x <= a or x >= d:
        return 0.0
    if b <= x <= c:
        return 1.0
    if x < b:
        return (x - a) / (b - a) if b != a else 0.0
    return (d - x) / (d - c) if d != c else 0.0


def l_shoulder(x: float, a: float, b: float) -> float:
    """
    Ombro à esquerda:
    - 1.0 para x <= a
    - 0.0 para x >= b
    - linear entre a e b
    """
    if x <= a:
        return 1.0
    if x >= b:
        return 0.0
    return (b - x) / (b - a) if b != a else 0.0


def r_shoulder(x: float, a: float, b: float) -> float:
    """
    Ombro à direita:
    - 0.0 para x <= a
    - 1.0 para x >= b
    - linear entre a e b
    """
    if x <= a:
        return 0.0
    if x >= b:
        return 1.0
    return (x - a) / (b - a) if b != a else 0.0


@dataclass(frozen=True)
class FuzzyOutput:
    vx: float
    vy: float
    omega: float


class FuzzyAvoider:
    """
    Controlador fuzzy bem simples e “explicável”:
    Inputs: distâncias mínimas (m) em setores esq/frente/dir.
    Outputs: vx (m/s) e omega (rad/s). vy é mantido 0 por enquanto.
    """

    def __init__(self, cruise_vx: float, max_vx: float, max_omega: float):
        self.cruise_vx = cruise_vx
        self.max_vx = max_vx
        self.max_omega = max_omega

        # parâmetros de pertinência (em metros)
        # near: ombro à esquerda (muito perto -> 1)
        self.near_hi = 0.12
        self.near_lo = 0.28

        self.med_a = 0.18
        self.med_b = 0.35
        self.med_c = 0.60

        # far: ombro à direita (bem longe -> 1)
        # Ajustado para considerar "longe" mais cedo (0.35-0.65 vs 0.45-0.85)
        self.far_lo = 0.35
        self.far_hi = 0.65

    def __call__(self, left: float, front: float, right: float) -> FuzzyOutput:
        # pertinências
        f_near = l_shoulder(front, self.near_hi, self.near_lo)
        f_med = tri(front, self.med_a, self.med_b, self.med_c)
        f_far = r_shoulder(front, self.far_lo, self.far_hi)

        l_near = l_shoulder(left, self.near_hi, self.near_lo)
        r_near = l_shoulder(right, self.near_hi, self.near_lo)

        # Omega: desvia para o lado mais livre usando:
        # - termos "fortes" quando um lado está muito perto
        # - um termo contínuo baseado no diferencial (left-right) para evitar ficar neutro
        w_left_strong = f_near * r_near          # obstáculo à direita => virar esquerda
        w_right_strong = f_near * l_near         # obstáculo à esquerda => virar direita

        # diferencial normalizado (-1..+1): positivo => mais espaço à esquerda => virar esquerda
        # tanh dá saturação suave.
        diff = left - right
        diff_term = math.tanh(0.9 * diff)        # ~[-1, +1]
        w_diff = f_near * (1.0 - max(l_near, r_near))

        omega = 0.0
        omega += (+1.0) * w_left_strong
        omega += (-1.0) * w_right_strong
        omega += (diff_term) * w_diff

        den = w_left_strong + w_right_strong + w_diff
        omega = (omega / den) if den > 1e-6 else 0.0
        omega *= self.max_omega

        # Vx: reduz quando frente não é “far”
        # - far => cruise
        # - med => mais lento
        # - near => quase parar
        vx = (
            self.cruise_vx * f_far
            + (0.55 * self.cruise_vx) * f_med
            + (0.05 * self.cruise_vx) * f_near
        )
        # normaliza pelo total de ativação (se tudo 0, mantém cruise)
        total = f_far + f_med + f_near
        if total > 1e-6:
            vx = vx / total

        # saturação
        if vx > self.max_vx:
            vx = self.max_vx
        if vx < 0.0:
            vx = 0.0

        return FuzzyOutput(vx=vx, vy=0.0, omega=omega)


