"""Contato solo-sapata sem resistência à tração.

Hipóteses (Aula 7 – Sapata isolada, Graziano):
    1) o solo responde de forma elástica linear, mas sem resistência à tração;
    2) a sapata é rígida: sua base permanece plana após a deformação do solo.

Convenções (mantidas do código original):
    - B é a dimensão em x e L a dimensão em y; origem no centro da base.
    - N > 0 é compressão.
    - ex = My / N  e  ey = -Mx / N  (posição da resultante na base).
    - pressão linear com compressão positiva:  p = p0 + px*x + py*y.

Zonas da resultante (figura "Caso Geral" da aula), com u = |ex|/B e v = |ey|/L:
    Zona 1 - contato total (núcleo central):           u + v <= 1/6
    Zona 2 - pentágono (um canto da base levantado)
    Zona 3 - trapézio (a linha neutra corta duas arestas opostas)
    Zona 4 - triângulo (só um canto comprimido):        u >= 1/4 e v >= 1/4
    Fora da base (|ex| >= B/2 ou |ey| >= L/2) não há equilíbrio.
A zona 4 (fora das elipses, < 50 % de contato) não atende: o solver não calcula tensões.
As fronteiras entre as zonas 2 e 3 são as curvas em que a linha neutra passa por
um canto da sapata (funções limite_x e limite_y).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import brentq

ZONAS = {
    1: "Zona 1 — contato total",
    2: "Zona 2 — pentágono",
    3: "Zona 3 — trapézio (> 50 % em contato)",
    4: "Zona 4 — fora das elipses (< 50 % em contato)",
}

# =============================================================================
# CLASSIFICAÇÃO DAS ZONAS DE EXCENTRICIDADE
# =============================================================================


def limite_y(x: np.ndarray | float, B: float, L: float) -> np.ndarray | float:
    """Curva y(x) em que a linha neutra passa pelo canto (+x, -y) da base.

    Separa a zona 2 (abaixo da curva) da zona 3 (acima). Válida para 0 <= x <= B/4;
    vai de (0, L/6) até (B/4, L/4).
    """
    u = np.abs(np.asarray(x, dtype=float)) / B
    raiz = np.sqrt(np.maximum(0.0, 1.0 - 12.0 * u**2))
    resultado = L / 2 - L / 6 * (2.0 - raiz) * (1.0 + 2.0 * u + raiz) / (1.0 + 4.0 * u)
    return float(resultado) if np.ndim(resultado) == 0 else resultado


def limite_x(y: np.ndarray | float, B: float, L: float) -> np.ndarray | float:
    """Curva x(y) em que a linha neutra passa pelo canto (-x, +y) da base.

    Separa a zona 2 (à esquerda da curva) da zona 3 (à direita). Válida para
    0 <= y <= L/4; vai de (B/6, 0) até (B/4, L/4).
    """
    v = np.abs(np.asarray(y, dtype=float)) / L
    raiz = np.sqrt(np.maximum(0.0, 1.0 - 12.0 * v**2))
    resultado = B / 2 - B / 6 * (2.0 - raiz) * (1.0 + 2.0 * v + raiz) / (1.0 + 4.0 * v)
    return float(resultado) if np.ndim(resultado) == 0 else resultado


def _forma_contato(ex: float, ey: float, B: float, L: float) -> Optional[int]:
    """Forma geométrica da área comprimida: 1 total, 2 pentágono, 3 trapézio, 4 triângulo.

    Usada internamente pelo solver; None se a resultante estiver fora da base.
    """
    tol = 1e-12
    x, y = abs(ex), abs(ey)
    u, v = x / B, y / L

    if u >= 0.5 or v >= 0.5:
        return None
    if u + v <= 1.0 / 6.0 + tol:
        return 1
    if u >= 0.25 - tol and v >= 0.25 - tol:
        return 4
    if v <= 0.25 and x >= limite_x(y, B, L) - tol * B:
        return 3
    if u <= 0.25 and y >= limite_y(x, B, L) - tol * L:
        return 3
    return 2


def dentro_elipses(u, v):
    """True onde a resultante está nas duas elipses da planilha (contato > ~50 %).

    u = |ex|/B e v = |ey|/L. Fora delas, menos da metade da base fica comprimida.
    """
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)
    return (7.0 * u**2 + 9.0 * v**2 <= 1.0) & (9.0 * u**2 + 7.0 * v**2 <= 1.0)


def classificar_zona(ex: float, ey: float, B: float, L: float) -> Optional[int]:
    """Zona (1 a 4) da resultante (ex, ey); None se estiver fora da base.

    1: contato total (núcleo central)
    2: pentágono (dentro das elipses)
    3: trapézio (dentro das elipses)
    4: fora das elipses (menos de ~50 % da base comprimida: triângulo ou
       trapézio estreito)
    """
    forma = _forma_contato(ex, ey, B, L)
    if forma in (None, 1):
        return forma
    if not bool(dentro_elipses(abs(ex) / B, abs(ey) / L)):
        return 4
    return forma  # 2 ou 3 (um triângulo nunca fica dentro das elipses)


# =============================================================================
# TIPOS E RESULTADOS
# =============================================================================


class ContactType(Enum):
    FULL = "Contato total"
    PARTIAL = "Contato parcial"
    NO_SOLUTION = "Sem solução"
    ZONE_4 = "Zona 4 — não atende"


@dataclass
class FootingResult:
    """Resultados do cálculo de contato solo-sapata.

    Unidades coerentes: B/L em m, N em kN, momentos em kN.m, pressões em kPa.
    """

    contact_type: ContactType
    B: float
    L: float
    N: float
    Mx: float
    My: float
    ex: float
    ey: float
    sigma_max: float
    sigma_min: float
    contact_area: float
    contact_polygon: list[tuple[float, float]]
    corner_pressures: dict[str, float] | None = None
    theta: float | None = None
    d: float | None = None
    x_pressure: float | None = None
    y_pressure: float | None = None
    sigma_mean: float | None = None
    force_error: float | None = None
    mx_error: float | None = None
    my_error: float | None = None
    zone: int | None = None
    # pressão em um ponto (x, y) da região comprimida: p = a + bx*x + by*y
    pressure_plane: tuple[float, float, float] | None = None

    @property
    def contact_ratio(self) -> float:
        """Fração da área da base em contato com o solo (0 a 1)."""
        return self.contact_area / (self.B * self.L)


# =============================================================================
# GEOMETRIA POLIGONAL
# =============================================================================


def footing_rectangle(B: float, L: float) -> list[tuple[float, float]]:
    """Vértices do retângulo no sentido anti-horário, centrado na origem."""
    return [(-B / 2, -L / 2), (B / 2, -L / 2), (B / 2, L / 2), (-B / 2, L / 2)]


def clip_polygon_by_line(
    polygon: list[tuple[float, float]],
    nx: float,
    ny: float,
    d: float,
    tolerance: float = 1e-12,
) -> list[tuple[float, float]]:
    """Recorta o polígono (convexo) pelo semiplano nx*x + ny*y - d >= 0."""
    if not polygon:
        return []

    output: list[tuple[float, float]] = []
    px, py = polygon[-1]
    f_prev = nx * px + ny * py - d
    prev_in = f_prev >= -tolerance

    for cx, cy in polygon:
        f_cur = nx * cx + ny * cy - d
        cur_in = f_cur >= -tolerance

        if cur_in != prev_in:
            den = f_prev - f_cur
            if abs(den) > 0.0:
                t = f_prev / den
                output.append((px + t * (cx - px), py + t * (cy - py)))
        if cur_in:
            output.append((cx, cy))

        px, py, f_prev, prev_in = cx, cy, f_cur, cur_in

    # remove pontos repetidos (inclusive entre o último e o primeiro)
    cleaned: list[tuple[float, float]] = []
    for p in output:
        if not cleaned or math.hypot(p[0] - cleaned[-1][0], p[1] - cleaned[-1][1]) > tolerance:
            cleaned.append(p)
    if len(cleaned) > 1 and math.hypot(
        cleaned[0][0] - cleaned[-1][0], cleaned[0][1] - cleaned[-1][1]
    ) <= tolerance:
        cleaned.pop()
    return cleaned


def integrate_pressure_shape(
    polygon: list[tuple[float, float]], nx: float, ny: float, d: float
) -> tuple[float, float, float, float]:
    """Integra s, x*s e y*s sobre o polígono, com s = nx*x + ny*y - d.

    Retorna (área, ∫s, ∫x·s, ∫y·s). Em cada triângulo, com valores f_i de s nos
    vértices e S = Σf_i, valem as fórmulas exatas:
        ∫ s dA   = A·S/3
        ∫ x·s dA = A/12 · Σ x_i·(S + f_i)      (idem para y)
    """
    if len(polygon) < 3:
        return 0.0, 0.0, 0.0, 0.0

    x0, y0 = polygon[0]
    s0 = nx * x0 + ny * y0 - d
    area = i_s = i_xs = i_ys = 0.0

    for i in range(1, len(polygon) - 1):
        x1, y1 = polygon[i]
        x2, y2 = polygon[i + 1]
        a = 0.5 * abs((x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0))
        if a < 1e-30:
            continue
        s1 = nx * x1 + ny * y1 - d
        s2 = nx * x2 + ny * y2 - d
        soma = s0 + s1 + s2
        area += a
        i_s += a * soma / 3.0
        i_xs += a / 12.0 * (x0 * (soma + s0) + x1 * (soma + s1) + x2 * (soma + s2))
        i_ys += a / 12.0 * (y0 * (soma + s0) + y1 * (soma + s1) + y2 * (soma + s2))

    return area, i_s, i_xs, i_ys


# =============================================================================
# SOLVER DE CONTATO
# =============================================================================


class FootingContact:
    """Pressão de contato linear sem resistência do solo à tração."""

    CORNERS = {
        "Inferior esquerdo": (-1, -1),
        "Inferior direito": (1, -1),
        "Superior direito": (1, 1),
        "Superior esquerdo": (-1, 1),
    }

    def __init__(self, B: float, L: float, N: float, Mx: float, My: float):
        if B <= 0 or L <= 0:
            raise ValueError("B e L devem ser positivos.")
        if N <= 0:
            raise ValueError("Este modelo considera N > 0 (compressão).")

        self.B, self.L = float(B), float(L)
        self.N, self.Mx, self.My = float(N), float(Mx), float(My)
        self.rectangle = footing_rectangle(self.B, self.L)

    # ------------------------------------------------------------------ dados
    @property
    def ex(self) -> float:
        return self.My / self.N

    @property
    def ey(self) -> float:
        return -self.Mx / self.N

    # ---------------------------------------------------------- contato total
    def full_contact_pressure_coefficients(self) -> tuple[float, float, float]:
        """Coeficientes (p0, px, py) da distribuição linear, compressão positiva."""
        area = self.B * self.L
        Ix = self.B * self.L**3 / 12.0
        Iy = self.L * self.B**3 / 12.0
        return self.N / area, self.My / Iy, -self.Mx / Ix

    def pressure_full_contact(self, x: float, y: float) -> float:
        p0, px, py = self.full_contact_pressure_coefficients()
        return p0 + px * x + py * y

    def solve_full_contact(self) -> tuple[float, float]:
        pressures = [self.pressure_full_contact(x, y) for x, y in self.rectangle]
        return min(pressures), max(pressures)

    def calculate_corner_pressures(
        self,
        *,
        k: float | None = None,
        nx: float | None = None,
        ny: float | None = None,
        d: float | None = None,
    ) -> dict[str, float]:
        """Pressões nos cantos; valores de tração são limitados a zero."""
        pressures: dict[str, float] = {}
        for name, (sx, sy) in self.CORNERS.items():
            x, y = sx * self.B / 2.0, sy * self.L / 2.0
            if k is None:
                pressure = self.pressure_full_contact(x, y)
            else:
                if nx is None or ny is None or d is None:
                    raise ValueError("nx, ny e d são necessários para contato parcial.")
                pressure = k * (nx * x + ny * y - d)
            pressures[name] = max(0.0, float(pressure))
        return pressures

    # -------------------------------------------------------- contato parcial
    def _centroid(self, nx: float, ny: float, d: float) -> tuple[float, float]:
        """Centro de pressões da distribuição s = nx*x + ny*y - d (s >= 0)."""
        poly = clip_polygon_by_line(self.rectangle, nx, ny, d)
        _, i_s, i_xs, i_ys = integrate_pressure_shape(poly, nx, ny, d)
        return i_xs / i_s, i_ys / i_s

    def _offset_for_theta(self, theta: float, u: float, v: float) -> float:
        """Para a direção `theta` da normal, acha d tal que n·centro = n·(u, v)."""
        c, s = math.cos(theta), math.sin(theta)
        R = self.B / 2 * c + self.L / 2 * s
        alvo = u * c + v * s
        r2 = (self.B**2 * c**2 + self.L**2 * s**2) / 12.0

        # d <= -R: contato total, n·centro = r2/(-d)
        if r2 / R >= alvo:
            return -r2 / alvo

        def g(d: float) -> float:
            cx, cy = self._centroid(c, s, d)
            return cx * c + cy * s - alvo

        d_hi = R * (1.0 - 1e-10)
        if g(d_hi) <= 0.0:
            return d_hi
        return brentq(g, -R, d_hi, xtol=1e-14, rtol=1e-14, maxiter=200)

    def _solve_first_quadrant(self, u: float, v: float, forma: int) -> tuple[float, float]:
        """Acha (theta, d) com theta em [0, pi/2] para a resultante (u, v) >= 0."""
        B, L = self.B, self.L
        eps = 1e-12

        if v <= eps * L:  # flexão em torno de um eixo: linha neutra vertical
            return 0.0, 3.0 * u - B
        if u <= eps * B:  # linha neutra horizontal
            return math.pi / 2, 3.0 * v - L
        if forma == 4:  # triângulo: fórmula fechada
            lx, ly = 2.0 * B - 4.0 * u, 2.0 * L - 4.0 * v
            c0 = B / (2.0 * lx) + L / (2.0 * ly) - 1.0
            norma = math.hypot(1.0 / lx, 1.0 / ly)
            return math.atan2(1.0 / ly, 1.0 / lx), c0 / norma

        def h(theta: float) -> float:
            d = self._offset_for_theta(theta, u, v)
            cx, cy = self._centroid(math.cos(theta), math.sin(theta), d)
            sn, cs = math.sin(theta), math.cos(theta)
            return (-sn * cx + cs * cy) - (-sn * u + cs * v)

        theta = brentq(h, 0.0, math.pi / 2, xtol=1e-15, rtol=1e-14, maxiter=200)
        return theta, self._offset_for_theta(theta, u, v)

    def solve_partial_contact(self) -> FootingResult:
        ex, ey = self.ex, self.ey
        forma = _forma_contato(ex, ey, self.B, self.L)
        if forma is None:
            return self._no_solution()
        zona = classificar_zona(ex, ey, self.B, self.L)

        sx = 1.0 if ex >= 0 else -1.0
        sy = 1.0 if ey >= 0 else -1.0
        theta_q, d = self._solve_first_quadrant(abs(ex), abs(ey), forma)

        # volta para o quadrante real da resultante
        nx, ny = sx * math.cos(theta_q), sy * math.sin(theta_q)

        poly = clip_polygon_by_line(self.rectangle, nx, ny, d)
        area, i_s, i_xs, i_ys = integrate_pressure_shape(poly, nx, ny, d)
        if area <= 0 or i_s <= 0:
            raise RuntimeError("Área comprimida inválida.")

        k = self.N / i_s
        corner_pressures = self.calculate_corner_pressures(k=k, nx=nx, ny=ny, d=d)
        mx_error = -k * i_ys - self.Mx
        my_error = k * i_xs - self.My

        if abs(my_error) > 1e-6 * self.N * self.B or abs(mx_error) > 1e-6 * self.N * self.L:
            raise RuntimeError(
                f"Equilíbrio não atingido (ΔMx={mx_error:.3e}, ΔMy={my_error:.3e})."
            )

        return FootingResult(
            contact_type=ContactType.PARTIAL,
            B=self.B, L=self.L, N=self.N, Mx=self.Mx, My=self.My,
            ex=ex, ey=ey,
            sigma_max=max(corner_pressures.values()), sigma_min=0.0,
            contact_area=area, contact_polygon=poly,
            corner_pressures=corner_pressures,
            theta=math.atan2(ny, nx), d=d,
            x_pressure=i_xs / i_s, y_pressure=i_ys / i_s,
            sigma_mean=self.N / area,
            force_error=k * i_s - self.N, mx_error=mx_error, my_error=my_error,
            zone=zona, pressure_plane=(-k * d, k * nx, k * ny),
        )

    # ------------------------------------------------------------------ solve
    def _no_solution(self) -> FootingResult:
        return FootingResult(
            contact_type=ContactType.NO_SOLUTION,
            B=self.B, L=self.L, N=self.N, Mx=self.Mx, My=self.My,
            ex=self.ex, ey=self.ey,
            sigma_max=np.nan, sigma_min=np.nan,
            contact_area=0.0, contact_polygon=[], corner_pressures=None,
            zone=None,
        )

    def _zona_4(self) -> FootingResult:
        """Resultante na zona 4 (< 50 % da base comprimida): reprovada, sem cálculo de tensões."""
        return FootingResult(
            contact_type=ContactType.ZONE_4,
            B=self.B, L=self.L, N=self.N, Mx=self.Mx, My=self.My,
            ex=self.ex, ey=self.ey,
            sigma_max=np.nan, sigma_min=np.nan,
            contact_area=0.0, contact_polygon=[], corner_pressures=None,
            zone=4,
        )

    def solve(self) -> FootingResult:
        forma = _forma_contato(self.ex, self.ey, self.B, self.L)

        # Resultante fora da base: não há equilíbrio só com pressões de compressão.
        if forma is None:
            return self._no_solution()

        # Zona 4: não atende; as tensões nem são calculadas.
        if classificar_zona(self.ex, self.ey, self.B, self.L) == 4:
            return self._zona_4()

        if forma == 1:
            sigma_min, sigma_max = self.solve_full_contact()
            return FootingResult(
                contact_type=ContactType.FULL,
                B=self.B, L=self.L, N=self.N, Mx=self.Mx, My=self.My,
                ex=self.ex, ey=self.ey,
                sigma_max=sigma_max, sigma_min=max(0.0, sigma_min),
                contact_area=self.B * self.L,
                contact_polygon=list(self.rectangle),
                corner_pressures=self.calculate_corner_pressures(),
                x_pressure=self.ex, y_pressure=self.ey,
                sigma_mean=self.N / (self.B * self.L),
                force_error=0.0, mx_error=0.0, my_error=0.0,
                zone=1, pressure_plane=self.full_contact_pressure_coefficients(),
            )

        return self.solve_partial_contact()


# =============================================================================
# RELATÓRIO
# =============================================================================


def print_result(result: FootingResult) -> None:
    """Imprime um resumo dos resultados do cálculo."""
    print("\n" + "=" * 60)
    print("RESULTADO DO CÁLCULO DA SAPATA")
    print("=" * 60)
    print(f"Tipo de contato: {result.contact_type.value}")
    if result.zone is not None:
        print(f"Posição da resultante: {ZONAS[result.zone]}")
    print(f"B = {result.B:.3f} | L = {result.L:.3f}")
    print(f"N = {result.N:.3f} | Mx = {result.Mx:.3f} | My = {result.My:.3f}")
    print(f"ex = {result.ex:.6f} | ey = {result.ey:.6f}")
    print(f"Tensão máxima = {result.sigma_max:.3f}")
    print(f"Tensão mínima = {result.sigma_min:.3f}")
    if result.sigma_mean is not None:
        print(f"Tensão média na área comprimida = {result.sigma_mean:.3f}")
    print(f"Área comprimida = {result.contact_area:.6f} ({100 * result.contact_ratio:.2f}%)")

    if result.corner_pressures:
        print("\nTensões nos vértices:")
        for name, pressure in result.corner_pressures.items():
            print(f"  {name}: {pressure:.3f}")

    if result.theta is not None:
        print(f"\nLinha neutra: theta = {np.degrees(result.theta):.3f}°; d = {result.d:.6f}")

    if result.force_error is not None:
        print("\nErros de equilíbrio:")
        print(f"  ΔN  = {result.force_error:.3e}")
        print(f"  ΔMx = {result.mx_error:.3e}")
        print(f"  ΔMy = {result.my_error:.3e}")


# =============================================================================
# GRÁFICOS
# =============================================================================

def _quatro_quadrantes(pontos):
    """Reflete uma polilinha do 1º quadrante nos quatro quadrantes."""
    p = np.asarray(pontos, dtype=float)
    return [
        np.column_stack([sx * p[:, 0], sy * p[:, 1]])
        for sx in (1.0, -1.0)
        for sy in (1.0, -1.0)
    ]


def _fronteiras_das_zonas(B: float, L: float, n: int = 200):
    """Polilinhas (1º quadrante) que separam as zonas, em metros."""
    y = np.linspace(0.0, L / 4, n)
    x = np.linspace(0.0, B / 4, n)
    nucleo = np.array([(B / 6, 0.0), (0.0, L / 6)])                       # zona 1
    curva_x = np.column_stack([limite_x(y, B, L), y])                      # zona 2 | 3
    curva_y = np.column_stack([x, limite_y(x, B, L)])                      # zona 2 | 3

    # fronteira das elipses (zona 3 | zona 4): ramo 9u²+7v²=1 até (1/4, 1/4),
    # depois ramo 7u²+9v²=1 até o eixo y
    ph1 = np.linspace(0.0, math.acos(0.75), n)
    ram1 = np.column_stack([B * np.cos(ph1) / 3, L * np.sin(ph1) / math.sqrt(7)])
    ph2 = np.linspace(math.asin(0.75), math.pi / 2, n)
    ram2 = np.column_stack([B * np.cos(ph2) / math.sqrt(7), L * np.sin(ph2) / 3])
    elipses = np.vstack([ram1, ram2])
    return [nucleo, curva_x, curva_y, elipses]


def _draw_zone_curves_on_footing(ax, result: FootingResult) -> None:
    """Desenha as fronteiras das zonas 1 a 4 nos quatro quadrantes (só linhas)."""
    B, L = result.B, result.L
    cores = ["#26733b", "#8a8a8a", "#8a8a8a", "#6d4c9f"]
    larguras = [1.3, 1.0, 1.0, 1.6]
    for pol, cor, lw in zip(_fronteiras_das_zonas(B, L), cores, larguras):
        for q in _quatro_quadrantes(pol):
            ax.plot(q[:, 0], q[:, 1], color=cor, lw=lw, zorder=3)

    rotulos = [
        ("Z1", 0.0, 0.0), ("Z2", 0.115 * B, 0.115 * L),
        ("Z3", 0.27 * B, 0.04 * L), ("Z3", 0.04 * B, 0.27 * L),
        ("Z4", 0.42 * B, 0.40 * L),
    ]
    for texto, x, y in rotulos:
        ax.text(x, y, texto, ha="center", va="center", fontsize=8,
                color="#333333", zorder=7,
                bbox=dict(fc="white", ec="none", alpha=0.6, pad=0.8))


def _draw_pressure_gradient(fig, ax, result: FootingResult) -> None:
    """Degradê da tensão na área comprimida: verde (baixa) -> vermelho (alta)."""
    polygon = result.contact_polygon
    if result.pressure_plane is None or len(polygon) < 3:
        return
    from matplotlib.patches import Polygon as MplPolygon

    B, L = result.B, result.L
    a, bx, by = result.pressure_plane
    n = 400
    xs = np.linspace(-B / 2, B / 2, n)
    ys = np.linspace(-L / 2, L / 2, n)
    X, Y = np.meshgrid(xs, ys)
    P = np.maximum(a + bx * X + by * Y, 0.0)

    vmax = result.sigma_max if np.isfinite(result.sigma_max) and result.sigma_max > 0 else 1.0
    im = ax.imshow(
        P, extent=(-B / 2, B / 2, -L / 2, L / 2), origin="lower",
        cmap="RdYlGn_r", vmin=0.0, vmax=vmax, interpolation="bilinear",
        aspect="equal", zorder=2,
    )
    clip = MplPolygon(np.asarray(polygon, dtype=float), closed=True, transform=ax.transData)
    im.set_clip_path(clip)
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label("Tensão no solo (kPa)")


COR_CRITICA = "#d7191c"
COR_OUTRAS = "#37474f"


def _draw_resultante_critica(ax, result: FootingResult, zona, rotulo=None) -> None:
    """Resultante da combinação dimensionante, em vermelho."""
    ax.scatter(result.ex, result.ey, color=COR_CRITICA, edgecolor="white", linewidth=0.7,
               s=75, zorder=11, label="Resultante — combinação crítica")
    prefixo = f"Comb. {rotulo}: " if rotulo is not None else ""
    nome_zona = ZONAS[zona].split(" — ")[0] if zona else "Fora da base"

    # O rótulo vai para fora, na direção oposta ao centro, para não cobrir as demais
    # resultantes (que costumam ficar agrupadas perto do centro da base).
    dx, dy = result.ex / result.B, result.ey / result.L
    n = math.hypot(dx, dy)
    ux, uy = (dx / n, dy / n) if n > 1e-9 else (0.7071, 0.7071)
    ax.annotate(
        f"{prefixo}R ({result.ex:.3f}, {result.ey:.3f})\n{nome_zona}",
        (result.ex, result.ey), xytext=(34 * ux, 34 * uy), textcoords="offset points",
        ha="left" if ux >= 0 else "right", va="bottom" if uy >= 0 else "top", fontsize=8,
        bbox=dict(facecolor="white", edgecolor=COR_CRITICA, alpha=0.95, boxstyle="round,pad=0.2"),
        arrowprops=dict(arrowstyle="-", color=COR_CRITICA, lw=0.8, shrinkA=0, shrinkB=3),
        zorder=12,
    )


def _draw_outras_resultantes(ax, outras) -> None:
    """Resultantes das demais combinações, em outra cor.

    `outras`: iterável de (rotulo, ex, ey).
    """
    outras = list(outras or [])
    if not outras:
        return
    xs = [o[1] for o in outras]
    ys = [o[2] for o in outras]
    ax.scatter(xs, ys, marker="D", color=COR_OUTRAS, edgecolor="white", linewidth=0.6,
               s=38, zorder=10, label="Resultantes — demais combinações")
    for rot, x, y in outras:
        ax.annotate(str(rot), (x, y), xytext=(5, -9), textcoords="offset points",
                    fontsize=7.5, color=COR_OUTRAS, zorder=10,
                    bbox=dict(facecolor="white", edgecolor="none", alpha=0.7, pad=0.6))


def _draw_contact_on_same_plan(fig, ax, result: FootingResult, rotulo=None) -> None:
    """Sobrepõe tensões, linha neutra, resultante e pressões nos cantos."""
    B, L = result.B, result.L
    vertices = footing_rectangle(B, L)
    rectangle = np.asarray(vertices + [vertices[0]], dtype=float)

    _draw_pressure_gradient(fig, ax, result)

    ax.plot(rectangle[:, 0], rectangle[:, 1], color="#26384b", lw=2.0,
            label="Contorno da sapata", zorder=6)

    polygon = result.contact_polygon
    if len(polygon) >= 3:
        contact = np.asarray(list(polygon) + [polygon[0]], dtype=float)
        ax.plot(contact[:, 0], contact[:, 1], color="#1565c0", lw=1.3, zorder=5)

    # Linha neutra do contato parcial
    if result.contact_type == ContactType.PARTIAL and result.theta is not None and result.d is not None:
        nx, ny, d = np.cos(result.theta), np.sin(result.theta), result.d
        intersections: list[np.ndarray] = []
        for i, p1_tuple in enumerate(vertices):
            p1 = np.asarray(p1_tuple, dtype=float)
            p2 = np.asarray(vertices[(i + 1) % len(vertices)], dtype=float)
            f1 = nx * p1[0] + ny * p1[1] - d
            f2 = nx * p2[0] + ny * p2[1] - d
            if abs(f1) < 1e-10:
                intersections.append(p1)
            if f1 * f2 < 0.0:
                intersections.append(p1 + f1 / (f1 - f2) * (p2 - p1))

        unique: list[np.ndarray] = []
        for point in intersections:
            if not any(np.linalg.norm(point - other) < 1e-8 for other in unique):
                unique.append(point)
        if len(unique) >= 2:
            p1, p2 = unique[:2]
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color="#c62828", ls="-.",
                    lw=1.7, label="Linha neutra de contato", zorder=7)

    # Resultante da combinação crítica (vermelha)
    zona = result.zone if result.zone is not None else classificar_zona(result.ex, result.ey, B, L)
    _draw_resultante_critica(ax, result, zona, rotulo)

    # Pressões nos vértices
    corner_locations = {
        "Inferior esquerdo": (-B / 2, -L / 2),
        "Inferior direito": (B / 2, -L / 2),
        "Superior direito": (B / 2, L / 2),
        "Superior esquerdo": (-B / 2, L / 2),
    }
    pressures = result.corner_pressures or {}
    for name, (x, y) in corner_locations.items():
        pressure = pressures.get(name, np.nan)
        ax.scatter(x, y, color="#0d47a1", s=24, zorder=9)
        label = f"{pressure:.1f} kPa" if np.isfinite(pressure) else "N/D"
        ax.annotate(
            label, (x, y), xytext=(5 if x >= 0 else -5, 5 if y >= 0 else -5),
            textcoords="offset points",
            ha="left" if x >= 0 else "right", va="bottom" if y >= 0 else "top",
            fontsize=7.5,
            bbox=dict(facecolor="white", edgecolor="#90a4ae", alpha=0.92, boxstyle="round,pad=0.2"),
            zorder=10,
        )


def plot_result(result: FootingResult, outras=None, rotulo=None):
    """Plota zonas (linhas), tensões em degradê, linha neutra e resultantes.

    result : combinação crítica (resultante em vermelho, com tensões).
    outras : (rotulo, ex, ey) das demais combinações (resultantes em outra cor).
    rotulo : identificação da combinação crítica.
    """
    fig, ax = plt.subplots(figsize=(9, 7.5))
    B, L = result.B, result.L
    outras = list(outras or [])

    _draw_zone_curves_on_footing(ax, result)

    if result.contact_type not in (ContactType.NO_SOLUTION, ContactType.ZONE_4):
        _draw_contact_on_same_plan(fig, ax, result, rotulo)
    else:
        v = footing_rectangle(B, L)
        r = np.asarray(v + [v[0]])
        ax.plot(r[:, 0], r[:, 1], color="#26384b", linewidth=2, label="Contorno da sapata", zorder=6)
        zona = 4 if result.contact_type == ContactType.ZONE_4 else None
        _draw_resultante_critica(ax, result, zona, rotulo)
        if result.contact_type == ContactType.ZONE_4:
            msg = "Resultante na zona 4 (menos de 50 % da base comprimida):\nnão atende — tensões não calculadas"
        else:
            msg = "Resultante fora da base: sem equilíbrio com compressão exclusiva"
        ax.text(0.5, 0.97, msg, ha="center", va="top", transform=ax.transAxes,
                color="#b71c1c", fontsize=9,
                bbox=dict(facecolor="white", edgecolor="#b71c1c", alpha=0.9))

    _draw_outras_resultantes(ax, outras)

    max_ex = max([abs(result.ex)] + [abs(o[1]) for o in outras])
    max_ey = max([abs(result.ey)] + [abs(o[2]) for o in outras])
    lim_x = max(B / 2 * 1.2, max_ex * 1.1)
    lim_y = max(L / 2 * 1.2, max_ey * 1.1)
    ax.set_xlim(-lim_x, lim_x)
    ax.set_ylim(-lim_y, lim_y)
    ax.set_aspect("equal", adjustable="box")  # mesma escala em x e y

    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title("Sapata — tensões no solo e zonas de excentricidade")
    ax.axhline(0, color="#666666", linewidth=0.6, zorder=1)
    ax.axvline(0, color="#666666", linewidth=0.6, zorder=1)
    ax.grid(alpha=0.2)

    # legenda das zonas (linhas) + elementos do desenho
    from matplotlib.lines import Line2D
    extras = [
        Line2D([0], [0], color="#26733b", lw=1.3, label="Z1: contato total (núcleo central)"),
        Line2D([0], [0], color="#8a8a8a", lw=1.0, label="Z2 | Z3: linha neutra passa por um canto"),
        Line2D([0], [0], color="#6d4c9f", lw=1.6, label="Z3 | Z4: elipses (50 % em contato)"),
    ]
    h, l = ax.get_legend_handles_labels()
    ax.legend(handles=h + extras, fontsize=7.5, loc="upper center",
              bbox_to_anchor=(0.5, -0.12), ncol=2, frameon=False)

    fig.tight_layout()
    return fig
