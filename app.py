"""Protótipo de interface para verificação de sapatas.

Executar: streamlit run app.py
"""
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon, Patch
from matplotlib.lines import Line2D
from matplotlib.transforms import offset_copy
import numpy as np
import pandas as pd
import streamlit as st
from tensao import plot_result
from backend import Geometria, Solo, Concreto, calcular_verificacoes, calcula_peso_concreto, calcula_peso_solo

st.set_page_config(page_title="Verificação de Sapatas", page_icon="🏗️", layout="wide")

# ============================================================
# ESTILO
# ============================================================
st.markdown("""
<style>
    :root {
        --azul: #1f3a5f;
        --azul-claro: #eef2f7;
        --borda: #d9e0e8;
        --ok: #1e8e4e;
        --ok-bg: #e6f4ec;
        --erro: #c62828;
        --erro-bg: #fdeaea;
        --texto-suave: #5d6b7a;
    }

    /* Área útil: respiro no topo para o título não ser cortado */
    .block-container {
        padding: 3.2rem 2rem 1.5rem 2rem !important;
        max-width: 1500px !important;
    }

    /* Tipografia */
    h1 { font-size: 1.6rem !important; font-weight: 700 !important; color: var(--azul);
         margin: 0 0 .1rem 0 !important; padding: 0 !important; line-height: 1.3 !important; }
    h2, h3 { color: var(--azul); }
    h3 { font-size: 1rem !important; font-weight: 650 !important;
         margin: 0 0 .5rem 0 !important; padding: 0 !important; }

    /* Cartões (containers com borda) */
    [data-testid="stVerticalBlockBorderWrapper"] {
        border-radius: 10px !important;
        border: 1px solid var(--borda) !important;
        background: #fff;
        padding: .9rem 1rem !important;
        box-shadow: 0 1px 3px rgba(31, 58, 95, .06);
    }
    [data-testid="stVerticalBlock"] { gap: .6rem; }
    [data-testid="stHorizontalBlock"] { gap: 1rem; }

    /* Campos de entrada */
    [data-testid="stWidgetLabel"] p { font-size: .78rem !important; color: var(--texto-suave); }
    /* ===== Caixas numéricas compactas ===== */
    [data-testid="stNumberInput"],
    [data-testid="stNumberInputContainer"],
    [data-testid="stNumberInput"] div[data-baseweb="input"],
    [data-testid="stNumberInput"] div[data-baseweb="base-input"] {
        min-height: 0 !important;
        height: 1.6rem !important;
        max-height: 1.6rem !important;
    }
    [data-testid="stNumberInput"] input {
        height: 1.6rem !important;
        min-height: 0 !important;
        padding: 0 .4rem !important;
        font-size: .78rem !important;
        line-height: 1.6rem !important;
    }
    [data-testid="stNumberInput"] button {
        height: 1.6rem !important;
        min-height: 0 !important;
        width: 1.2rem !important;
        min-width: 1.2rem !important;
        padding: 0 !important;
    }
    [data-testid="stNumberInput"] button svg { width: .6rem !important; height: .6rem !important; }

    /* O rótulo fica fora da altura fixa */
    [data-testid="stNumberInput"] { height: auto !important; max-height: none !important; }
    [data-testid="stNumberInput"] [data-testid="stWidgetLabel"] {
        min-height: 0 !important;
        margin-bottom: .1rem !important;
    }
    [data-testid="stNumberInput"] [data-testid="stWidgetLabel"] p { font-size: .72rem !important; }

    /* Abas */
    button[data-baseweb="tab"] { font-size: .85rem !important; padding: .35rem .8rem !important; }

    /* Botões */
    .stButton button, .stDownloadButton button {
        font-size: .82rem !important; min-height: 2rem !important; padding: .2rem .6rem !important;
        border-radius: 8px !important;
    }

    /* Gráficos nunca ficam gigantes */
    [data-testid="stImage"] img { max-height: 42vh; object-fit: contain; }

    /* Cartões de indicadores (KPI) */
    .kpi { background: var(--azul-claro); border-radius: 8px; padding: .55rem .75rem; height: 100%; }
    .kpi .rotulo { font-size: .7rem; color: var(--texto-suave); text-transform: uppercase; letter-spacing: .03em; }
    .kpi .valor { font-size: 1.3rem; font-weight: 700; color: var(--azul); line-height: 1.25; }
    .kpi .sub { font-size: .72rem; color: var(--texto-suave); }

    /* Selo de status */
    .selo { display: inline-block; font-size: .78rem; font-weight: 600; padding: .22rem .65rem;
            border-radius: 999px; margin-top: .15rem; }
    .selo.ok { background: var(--ok-bg); color: var(--ok); }
    .selo.erro { background: var(--erro-bg); color: var(--erro); }

    /* Texto auxiliar */
    .aux { font-size: .78rem; color: var(--texto-suave); margin: .15rem 0; }
    [data-testid="stCaptionContainer"] { margin: 0 !important; }
</style>
""", unsafe_allow_html=True)

DEFAULT_COMB = pd.DataFrame([
    #[1, 1200., 50, 40, 0., 0.],
    #[2, 1200., 50, 40, 50., 0.],
    #[3, 1200., 50, 40, -50., 0.],
    #[4, 1200., 50, 40, 0., 70.],
    #[5, 1200., 50, 40, 0., -70.],
    #[6, 900., 50, 40, 120., 80.],
    #[7, 900., 50, 40, -120., 80.],
    #[8, 900., 50, 40, 100., -100.],
], columns=["Comb.", "N", "Fx", "Fy", "Mx", "My"])


def normaliza_tabela(df):
    """Todas as colunas de esforço como float (colunas int64 não aceitam decimais no editor)."""
    df = df.reset_index(drop=True).copy()
    for col in df.columns:
        if col != "Comb.":
            df[col] = pd.to_numeric(df[col], errors="coerce").astype(float)
    return df


DEFAULT_COMB = normaliza_tabela(DEFAULT_COMB)


def define_tabela(df, com_base=False):
    """Substitui a tabela e recria o editor.

    `comb` guarda a tabela atual (só entradas); `comb_base` é o que o editor recebe e
    só muda aqui. A chave do editor muda junto (ed_ver), para ele partir do zero.
    """
    df = normaliza_tabela(df)
    st.session_state.comb = df
    st.session_state.comb_base = df
    st.session_state.mostra_base = False
    st.session_state.ed_ver = st.session_state.get("ed_ver", 0) + 1


if "comb" not in st.session_state:
    define_tabela(DEFAULT_COMB)
if "resultado" not in st.session_state:
    st.session_state.resultado = None


# ============================================================
# COMPONENTES AUXILIARES
# ============================================================
def kpi(rotulo, valor, sub=""):
    sub_html = f'<div class="sub">{sub}</div>' if sub else ""
    st.markdown(
        f'<div class="kpi"><div class="rotulo">{rotulo}</div>'
        f'<div class="valor">{valor}</div>{sub_html}</div>',
        unsafe_allow_html=True,
    )


def selo(aprovado, texto_ok="Atendida", texto_erro="Não atendida"):
    classe, icone, texto = ("ok", "✔", texto_ok) if aprovado else ("erro", "✖", texto_erro)
    st.markdown(f'<span class="selo {classe}">{icone} {texto}</span>', unsafe_allow_html=True)


def fmt_fs(valor):
    return f"{valor:.2f}" if np.isfinite(valor) else "∞"


def linha_esforcos(esf):
    """Converte o dicionário de esforços em uma linha compacta."""
    partes = []
    for k, v in (esf or {}).items():
        try:
            partes.append(f"{k} = {float(v):.1f}")
        except (TypeError, ValueError):
            partes.append(f"{k} = {v}")
    return "  ·  ".join(partes)

COLS_BASE = ["N*", "Mx*", "My*"]


def assinatura_calculo(g, s, c, comb, posicao, natureza):
    """Identifica o estado dos dados usados no cálculo (para saber se o resultado está atual)."""
    num = comb.apply(pd.to_numeric, errors="coerce").astype(float).round(9)
    return (g, s, c, num.to_csv(index=False), posicao, natureza)


def adiciona_esforcos_base(comb, resultado):
    """Acrescenta N*, Mx* e My* (esforços na base da sapata) à tabela de combinações."""
    df = comb.copy()
    esf = [r["esforcos"] for r in resultado["resultados_por_combinacao"]]
    df["N*"] = [e["N"] for e in esf]
    df["Mx*"] = [e["Mx"] for e in esf]
    df["My*"] = [e["My"] for e in esf]
    return df

# ============================================================
# DESENHOS
# ============================================================
COR_CONCRETO = "#c9ced4"
COR_SOLO = "#e6d5ae"
COR_LINHA = "#222222"
COR_COTA = "#333333"


def _cota_h(ax, x1, x2, y, texto, ref=None, side=1, ext=0.0, fs=7.5):
    """Cota horizontal em y; ref = y de referência das linhas de extensão."""
    ax.annotate("", (x1, y), (x2, y),
                arrowprops={"arrowstyle": "<->", "lw": .8, "color": COR_COTA,
                            "shrinkA": 0, "shrinkB": 0})
    if ref is not None:
        for x in (x1, x2):
            ax.plot([x, x], [ref, y + side * ext], lw=.5, color="#777", ls=":", zorder=1)
    ax.text((x1 + x2) / 2, y, texto, ha="center", va="center", fontsize=fs,
            color=COR_COTA, bbox={"fc": "white", "ec": "none", "pad": .8}, zorder=6)


def _cota_v(ax, x, y1, y2, texto, ref1=None, ref2=None, side=1, ext=0.0, fs=7.5):
    """Cota vertical em x; ref1/ref2 = x de referência das linhas de extensão."""
    ax.annotate("", (x, y1), (x, y2),
                arrowprops={"arrowstyle": "<->", "lw": .8, "color": COR_COTA,
                            "shrinkA": 0, "shrinkB": 0})
    for y, r in ((y1, ref1), (y2, ref2)):
        if r is not None:
            ax.plot([r, x + side * ext], [y, y], lw=.5, color="#777", ls=":", zorder=1)
    ax.text(x, (y1 + y2) / 2, texto, ha="center", va="center", rotation=90,
            fontsize=fs, color=COR_COTA,
            bbox={"fc": "white", "ec": "none", "pad": .8}, zorder=6)


COR_FORCA = "#1565c0"
COR_MOMENTO = "#c2410c"
COR_EIXO = "#555555"


def _seta(fig, ax, origem, eixo, comp, cor, pontas=1, ms=7.0, lw=1.8, rotulo=None,
          lado=1, fs=9.5):
    """Seta paralela a x ou y, no sentido positivo do eixo.

    pontas=1: seta simples (força); pontas=2: seta de duas pontas (vetor momento, regra
    da mão direita). O tamanho das pontas é em pontos (independe da escala do desenho).
    lado: lado do rótulo em relação à seta (+1 / -1).
    """
    x0, y0 = origem
    if eixo == "x":
        x1, y1, marca, u = x0 + comp, y0, ">", (1.0, 0.0)
    else:
        x1, y1, marca, u = x0, y0 + comp, "^", (0.0, 1.0)

    passo = 0.8 * ms
    recuo_ultima = ms / 2 + (pontas - 1) * passo
    # haste (termina no centro da última ponta)
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0), zorder=8,
                arrowprops=dict(arrowstyle="-", color=cor, lw=lw, shrinkA=0,
                                shrinkB=recuo_ultima, capstyle="butt"))
    # pontas: triângulos deslocados em pontos
    for k in range(pontas):
        recuo = ms / 2 + k * passo
        tr = offset_copy(ax.transData, fig=fig, x=-u[0] * recuo, y=-u[1] * recuo, units="points")
        ax.plot([x1], [y1], ls="none", marker=marca, ms=ms, color=cor, mec=cor,
                transform=tr, zorder=9, clip_on=False)

    if rotulo:
        if eixo == "x":
            off, ha, va = (0, 7 * lado), "center", "bottom" if lado > 0 else "top"
        else:
            off, ha, va = (7 * lado, 0), "left" if lado > 0 else "right", "center"
        ax.annotate(rotulo, (x1, y1), xytext=off, textcoords="offset points",
                    ha=ha, va=va, fontsize=fs, color=cor, zorder=10)


def desenha_eixos_e_esforcos(fig, ax, g: Geometria, d: float):
    """Eixos x, y e sentidos positivos de Fx, Fy (setas simples) e Mx, My (setas duplas).

    Convenção do backend: x, y no plano da base, z para cima; momentos pela regra da mão
    direita (o vetor Mx aponta para +x e o vetor My para +y).
    """
    off = 0.13 * min(g.a, g.b)  # afastamento dos vetores de momento em relação aos eixos
    # comprimentos esquemáticos, proporcionais à dimensão da sapata em cada eixo
    fx, fy = 0.38 * g.a, 0.38 * g.b      # forças
    mx, my = 0.25 * g.a, 0.25 * g.b      # momentos

    # --- eixos: metade negativa tracejada, metade positiva com seta e rótulo
    xe, ye = g.a / 2 + 0.5 * d, g.b / 2 + 0.5 * d
    ax.plot([-g.a / 2 - 0.15 * d, 0], [0, 0], ls="-.", lw=.6, color=COR_EIXO, zorder=1)
    ax.plot([0, 0], [-g.b / 2 - 0.15 * d, 0], ls="-.", lw=.6, color=COR_EIXO, zorder=1)
    _seta(fig, ax, (0, 0), "x", xe, COR_EIXO, ms=5.5, lw=.9, rotulo="x", lado=-1, fs=10)
    _seta(fig, ax, (0, 0), "y", ye, COR_EIXO, ms=5.5, lw=.9, rotulo="y", lado=1, fs=10)

    # --- forças: ao longo dos eixos
    _seta(fig, ax, (0, 0), "x", fx, COR_FORCA, rotulo="$F_x$", lado=1)
    _seta(fig, ax, (0, 0), "y", fy, COR_FORCA, rotulo="$F_y$", lado=1)
    # --- momentos: setas de duas pontas, afastadas dos eixos (quadrantes livres)
    _seta(fig, ax, (0, 0), "x", mx, COR_MOMENTO, pontas=2, rotulo="$M_x$", lado=-1)
    _seta(fig, ax, (0, 0), "y", my, COR_MOMENTO, pontas=2, rotulo="$M_y$", lado=-1)


def desenho_geometria(g: Geometria):
    """Planta e corte lado a lado, com cotas, pedestal e solo sobre a sapata."""
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(9.5, 4.4), gridspec_kw={"width_ratios": [1, 1.2]}
    )
    fig.patch.set_facecolor("white")

    # =========================== PLANTA ===========================
    d = 0.12 * max(g.a, g.b)
    e = 0.3 * d  # prolongamento das linhas de extensão

    # sapata (base)
    ax1.add_patch(Rectangle((-g.a/2, -g.b/2), g.a, g.b,
                            fc=COR_CONCRETO, ec=COR_LINHA, lw=1.4, zorder=2))
    # topo da sapata (ap × bp), oculto pelo solo/pedestal -> tracejado
    ax1.add_patch(Rectangle((-g.ap/2, -g.bp/2), g.ap, g.bp,
                            fc="none", ec=COR_LINHA, lw=.8, ls="--", zorder=3))
    # pedestal
    ax1.add_patch(Rectangle((-g.a_ped/2, -g.b_ped/2), g.a_ped, g.b_ped,
                            fc=COR_CONCRETO, ec=COR_LINHA, lw=1.2, zorder=4))
    desenha_eixos_e_esforcos(fig, ax1, g, d)

    # cotas da planta
    _cota_h(ax1, -g.a/2, g.a/2, g.b/2 + d, f"a = {g.a:.2f}", ref=g.b/2, ext=e)
    _cota_v(ax1, -g.a/2 - d, -g.b/2, g.b/2, f"b = {g.b:.2f}",
            ref1=-g.a/2, ref2=-g.a/2, side=-1, ext=e)
    _cota_h(ax1, -g.a_ped/2, g.a_ped/2, -g.b/2 - d, f"a_ped = {g.a_ped:.2f}",
            ref=-g.b_ped/2, side=-1, ext=e)
    _cota_v(ax1, g.a/2 + d, -g.b_ped/2, g.b_ped/2, f"b_ped = {g.b_ped:.2f}",
            ref1=g.a_ped/2, ref2=g.a_ped/2, side=1, ext=e)

    ax1.set_title("Planta", loc="left", fontweight="bold", fontsize=10)
    ax1.set_aspect("equal")
    ax1.set_xlim(-g.a/2 - 2.2*d, g.a/2 + 2.2*d)
    ax1.set_ylim(-g.b/2 - 2.0*d, g.b/2 + 1.8*d)
    ax1.axis("off")

    # ===================== CORTE (direção X) ======================
    z_solo = g.h0 + g.h_solo            # cota do terreno (a partir da base da sapata)
    z_ped = g.h + g.h_ped               # topo do pedestal
    z_top = max(z_ped, z_solo)
    d = 0.08 * max(g.a, z_top)
    e = 0.3 * d
    xe = g.a/2 + d                      # largura lateral de solo desenhada

    # solo (atrás do concreto)
    ax2.add_patch(Rectangle((-xe, 0), 2*xe, z_solo,
                            fc=COR_SOLO, ec="none", zorder=1))
    ax2.plot([-xe, xe], [z_solo, z_solo], color="#6b5a3a", lw=1.3, zorder=3)

    # concreto: sapata (rodapé + tronco de pirâmide) + pedestal, num único contorno
    contorno = [
        (-g.a/2, 0), (g.a/2, 0), (g.a/2, g.h0), (g.ap/2, g.h),
        (g.a_ped/2, g.h), (g.a_ped/2, z_ped),
        (-g.a_ped/2, z_ped), (-g.a_ped/2, g.h),
        (-g.ap/2, g.h), (-g.a/2, g.h0),
    ]
    ax2.add_patch(Polygon(contorno, closed=True, fc=COR_CONCRETO,
                          ec=COR_LINHA, lw=1.4, zorder=4))
    # divisa sapata/pedestal e topo do rodapé
    ax2.plot([-g.a_ped/2, g.a_ped/2], [g.h, g.h], lw=.6, color=COR_LINHA, zorder=5)
    ax2.plot([-g.a/2, g.a/2], [g.h0, g.h0], lw=.6, color=COR_LINHA, ls="--", zorder=5)

    # cotas horizontais
    _cota_h(ax2, -g.a/2, g.a/2, -d, f"a = {g.a:.2f}", ref=0, side=-1, ext=e)
    _cota_h(ax2, -g.a_ped/2, g.a_ped/2, z_top + d, f"a_ped = {g.a_ped:.2f}",
            ref=z_ped, ext=e)
    _cota_h(ax2, -g.ap/2, g.ap/2, z_top + 2.4*d, f"ap = {g.ap:.2f}",
            ref=g.h, ext=e)

    # cotas verticais — direita
    xr1, xr2 = xe + d, xe + 2.6*d
    _cota_v(ax2, xr1, 0, g.h0, f"h0 = {g.h0:.2f}", ref1=g.a/2, ref2=g.a/2, ext=e)
    _cota_v(ax2, xr2, 0, g.h, f"h = {g.h:.2f}", ref1=g.a/2, ref2=g.ap/2, ext=e)
    _cota_v(ax2, xr1, g.h, z_ped, f"h_ped = {g.h_ped:.2f}",
            ref1=g.a_ped/2, ref2=g.a_ped/2, ext=e)
    # cota vertical — esquerda (solo sobre a sapata, medido a partir do topo do rodapé)
    xl = -(xe + d)
    _cota_v(ax2, xl, g.h0, z_solo, f"h_solo = {g.h_solo:.2f}",
            ref1=-g.a/2, ref2=-xe, side=-1, ext=e)

    ax2.set_title("Corte (direção X)", loc="left", fontweight="bold", fontsize=10)
    ax2.set_aspect("equal")
    ax2.set_xlim(-(xe + 2.2*d), xe + 3.8*d)
    ax2.set_ylim(-2.2*d, z_top + 3.4*d)
    ax2.axis("off")

    # # ========================== LEGENDA ===========================
    # fig.legend(
    #     handles=[
    #         Patch(fc=COR_CONCRETO, ec=COR_LINHA, label="Concreto (sapata e pedestal)"),
    #         Patch(fc=COR_SOLO, ec="#6b5a3a", label="Solo sobre a sapata"),
    #         Line2D([0], [0], color=COR_LINHA, lw=.8, ls="--",
    #                label=f"Topo da sapata ({g.ap:.2f} × {g.bp:.2f})"),
    #     ],
    #     loc="lower center", ncol=3, frameon=False, fontsize=8,
    # )
    # fig.tight_layout(pad=.8, rect=(0, 0.07, 1, 1))
    return fig


def desenho_tensoes(g: Geometria, campo=None, comprimida=None):
    """campo: matriz de tensões em kPa; comprimida: máscara booleana opcional."""
    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    fig.patch.set_facecolor("white")
    if campo is None:
        ax.add_patch(Rectangle((0, 0), g.a, g.b, facecolor="#e5eaf0", edgecolor="#26384b", lw=2))
        ax.text(g.a/2, g.b/2, "Aguardando cálculo\ndo backend", ha="center", va="center", color="#54657a")
    else:
        campo = np.asarray(campo, dtype=float)
        if campo.ndim != 2:
            raise ValueError("O campo de tensões deve ser uma matriz 2D.")
        if comprimida is not None:
            mask = np.asarray(comprimida, dtype=bool)
            if mask.shape != campo.shape:
                raise ValueError("Máscara e campo precisam ter as mesmas dimensões.")
            campo = np.ma.masked_where(~mask, campo)
        im = ax.imshow(campo, extent=(0, g.a, 0, g.b), origin="lower", cmap="turbo", aspect="equal", interpolation="nearest")
        fig.colorbar(im, ax=ax, label="Tensão (kPa)", fraction=.046, pad=.04)
        ax.add_patch(Rectangle((0, 0), g.a, g.b, fill=False, edgecolor="#26384b", lw=1.5))
    ax.add_patch(Rectangle(((g.a-g.ap)/2, (g.b-g.bp)/2), g.ap, g.bp, facecolor="#888", edgecolor="#222", alpha=.7))
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_aspect("equal")
    fig.tight_layout()
    return fig


# ============================================================
# CABEÇALHO
# ============================================================
# st.title("🏗️ Verificação de Sapatas")
st.title("Verificação de Sapatas")
st.caption("Eng. Guilherme Vick")
# st.caption("Deslizamento · Tombamento · Tensões no solo")

# ============================================================
# LINHA 1 — DADOS DE ENTRADA  |  CROQUI
# ============================================================
col_dados, col_croqui = st.columns([1.05, 1.0], gap="medium")

with col_dados:
    with st.container(border=True):
        st.subheader("Dados de entrada")
        aba_geo, aba_mat = st.tabs(["📐 Geometria", "🧱 Solo e concreto"])

        with aba_geo:
            st.markdown('<div class="aux"><b>Sapata</b></div>', unsafe_allow_html=True)
            c1, c2, c3, c4 = st.columns(4, gap="small")
            a = c1.number_input("a — X (m)", min_value=.01, value=2.0, step=.05, format="%.2f")
            b = c2.number_input("b — Y (m)", min_value=.01, value=2.5, step=.05, format="%.2f")
            h = c3.number_input("h — total (m)", min_value=.01, value=.6, step=.05, format="%.2f")
            h0 = c4.number_input("h0 — rodapé (m)", min_value=.01, value=.4, step=.05, format="%.2f")

            st.markdown('<div class="aux"><b>Colarinho</b></div>', unsafe_allow_html=True)
            c1, c2, _, _ = st.columns(4, gap="small")
            ap = c1.number_input("ap — X (m)", min_value=.01, value=.75, step=.05, format="%.2f")
            bp = c2.number_input("bp — Y (m)", min_value=.01, value=.95, step=.05, format="%.2f")

            st.markdown('<div class="aux"><b>Pedestal e solo</b></div>', unsafe_allow_html=True)
            c1, c2, c3, c4 = st.columns(4, gap="small")
            a_ped = c1.number_input("aped — X (m)", min_value=.01, value=.6, step=.05, format="%.2f")
            b_ped = c2.number_input("bped — Y (m)", min_value=.01, value=.8, step=.05, format="%.2f")
            h_ped = c3.number_input("h_ped (m)", min_value=.01, value=1.2, step=.05, format="%.2f")
            h_solo = c4.number_input("h_solo (m)", min_value=.01, value=1., step=.05, format="%.2f")
            st.caption("h_ped: altura do pedestal · h_solo: altura de solo sobre a sapata")

        with aba_mat:
            st.markdown('<div class="aux"><b>Solo</b></div>', unsafe_allow_html=True)
            c1, c2, c3 = st.columns(3, gap="small")
            peso_esp_solo = c1.number_input("γsolo (kN/m³)", min_value=0.01, value=18., step=1.)
            mu = c2.number_input("μ — atrito", min_value=0., value=.5, step=.05)
            sigma = c3.number_input("σadm (kPa)", min_value=0.01, value=200., step=10.)

            st.markdown('<div class="aux"><b>Concreto</b></div>', unsafe_allow_html=True)
            c1, _, _ = st.columns(3, gap="small")
            peso_esp_concreto = c1.number_input("γconc (kN/m³)", min_value=.01, value=25., step=.05, format="%.2f")

g = Geometria(a, b, ap, bp, a_ped, b_ped, h, h0, h_ped, h_solo)
c = Concreto(peso_esp_concreto)
s = Solo(sigma, mu, peso_esp_solo)
valida = ap <= a and bp <= b and h0 <= h

# ---------- Pesos próprios ----------
with st.container(border=True):
    st.subheader("Pesos")

    # dh = max(h - h0, 0.0)                      # altura da parte inclinada
    # A1, A2 = a * b, ap * bp                    # base e topo do tronco de pirâmide
    # v_base = A1 * h0                           # prisma da base (rodapé)
    # v_tronco = dh / 3 * (A1 + A2 + np.sqrt(A1 * A2))
    # v_ped = a_ped * b_ped * h_ped              # pedestal

    # v_conc = v_base + v_tronco + v_ped
    # # Solo sobre a sapata: coluna acima da face superior + volume sobre as faces
    # # inclinadas, descontando o pedestal embutido no solo.
    # v_solo = (A1 * h_solo
    #           + (A1 * dh - v_tronco)
    #           - a_ped * b_ped * min(h_ped, h_solo))
    # v_solo = max(v_solo, 0.0)

    # p_sapata = peso_esp_concreto * v_conc
    # p_solo = peso_esp_solo * v_solo
    p_sapata = calcula_peso_concreto(geometria=g, concreto=c)
    p_solo = calcula_peso_solo(geometria=g, solo=s)

    k1, k2, k3 = st.columns(3, gap="small")
    with k1:
        kpi("Peso da sapata", f"{p_sapata:.1f} kN")
    with k2:
        kpi("Peso do solo sobre a sapata", f"{p_solo:.1f} kN")
    with k3:
        kpi("Total", f"{p_sapata + p_solo:.1f} kN")


with col_croqui:
    with st.container(border=True):
        st.subheader("Croqui")
        if not valida:
            st.error("Verifique a geometria: ap ≤ a, bp ≤ b e h0 ≤ h.")
        fig = desenho_geometria(g)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
        st.caption("Sentidos positivos dos esforços informados: setas simples = Fx e Fy; "
                   "setas de duas pontas = Mx e My (regra da mão direita); z para cima.")

# ============================================================
# LINHA 2 — CARREGAMENTOS
# ============================================================
with st.container(border=True):
    st.subheader("Carregamentos (combinações)")
    col_cfg, col_tab = st.columns([0.8, 2.2], gap="medium")

    with col_cfg:
        posicao = st.radio("Esforços fornecidos no:", ["Topo do pedestal", "Base da sapata"])
        natureza = st.radio("Esforços fornecidos como:", ["Ações", "Reações"], index=1, horizontal=True)

        b1, b2 = st.columns(2, gap="small")
        with b1:
            if st.button("➕ Adicionar", use_container_width=True):
                df = st.session_state.comb
                new_id = int(pd.to_numeric(df["Comb."], errors="coerce").max()) + 1 if len(df) else 1
                define_tabela(pd.concat(
                    [df, pd.DataFrame([[new_id, 0., 0., 0., 0., 0.]], columns=df.columns)], ignore_index=True))
                st.rerun()
        with b2:
            if st.button("📋 Duplicar", use_container_width=True):
                df = st.session_state.comb
                if len(df):
                    nova = df.iloc[[-1]].copy()
                    nova["Comb."] = int(pd.to_numeric(df["Comb."], errors="coerce").max()) + 1
                    define_tabela(pd.concat([df, nova], ignore_index=True))
                    st.rerun()
        b3, b4 = st.columns(2, gap="small")
        with b3:
            if st.button("🗑️ Excluir", use_container_width=True):
                define_tabela(st.session_state.comb.iloc[:-1])
                st.rerun()
        with b4:
            st.download_button(
                "⬇️ CSV", use_container_width=True,
                data=st.session_state.comb.to_csv(index=False, sep=";").encode("utf-8-sig"),
                file_name="combinacoes.csv", mime="text/csv")

        with st.expander("Importar CSV"):
            arquivo = st.file_uploader("CSV com separador ;", type="csv", label_visibility="collapsed")
            if arquivo is not None and st.button("Aplicar importação"):
                novo = pd.read_csv(arquivo, sep=";", dtype=str)
                if set(DEFAULT_COMB.columns).issubset(novo.columns):
                    novo = novo[DEFAULT_COMB.columns].apply(
                        lambda col: pd.to_numeric(col.str.strip().str.replace(",", ".", regex=False),
                                                  errors="coerce"))
                    define_tabela(novo)
                    st.rerun()
                else:
                    st.error("O CSV deve conter: Comb., N, Fx, Fy, Mx, My")

    with col_tab:
        # O editor recebe sempre `comb_base` (que só muda em define_tabela) e a saída dele
        # nunca é devolvida como entrada: reatribuir a saída aos próprios dados fazia a
        # primeira edição ser descartada.
        editado = st.data_editor(
            st.session_state.comb_base, num_rows="dynamic", hide_index=True,
            column_config={
                "Comb.": st.column_config.NumberColumn("Comb.", step=1, format="%d", width="small"),
                "N": st.column_config.NumberColumn("N (kN)", format="%.2f"),
                "Fx": st.column_config.NumberColumn("Fx (kN)", format="%.2f"),
                "Fy": st.column_config.NumberColumn("Fy (kN)", format="%.2f"),
                "Mx": st.column_config.NumberColumn("Mx (kN·m)", format="%.2f"),
                "My": st.column_config.NumberColumn("My (kN·m)", format="%.2f"),
                "N*": st.column_config.NumberColumn(
                    "N* (kN)", format="%.2f", help="Esforço normal na base da sapata (inclui pesos próprios)"),
                "Mx*": st.column_config.NumberColumn(
                    "Mx* (kN·m)", format="%.2f", help="Momento Mx na base da sapata"),
                "My*": st.column_config.NumberColumn(
                    "My* (kN·m)", format="%.2f", help="Momento My na base da sapata"),
            },
            disabled=COLS_BASE if st.session_state.mostra_base else False,
            use_container_width=True, height=300,
            key=f"editor_comb_{st.session_state.ed_ver}",
        )

        st.session_state.comb = normaliza_tabela(editado.drop(columns=COLS_BASE, errors="ignore"))

        # Se algo mudou depois do cálculo, N*, Mx* e My* deixam de valer: remove as colunas.
        if st.session_state.mostra_base and st.session_state.get("assinatura") != assinatura_calculo(
                g, s, c, st.session_state.comb, posicao, natureza):
            define_tabela(st.session_state.comb)
            st.rerun()

# ============================================================
# LINHA 3 — RESULTADOS
# ============================================================
head_t, head_b = st.columns([3, 1], gap="medium", vertical_alignment="center")
with head_t:
    st.subheader("Resultados")
with head_b:
    executar = st.button("▶ Calcular verificações", type="primary", disabled=not valida, use_container_width=True)

if executar:
    sucesso = False
    try:
        st.session_state.resultado = calcular_verificacoes(
            g, s, c, st.session_state.comb.copy(), posicao, natureza)
        st.session_state.assinatura = assinatura_calculo(
            g, s, c, st.session_state.comb, posicao, natureza)
        sucesso = True
    except NotImplementedError as e:
        st.session_state.resultado = None
        st.warning(str(e))
    except Exception as e:
        st.session_state.resultado = None
        st.error(f"Falha no cálculo: {e}")

    if sucesso:
        tabela = st.session_state.comb
        base = adiciona_esforcos_base(tabela, st.session_state.resultado)
        st.session_state.comb_base = base
        st.session_state.mostra_base = True
        st.session_state.ed_ver += 1
        st.rerun()  # redesenha a tabela já com N*, Mx* e My*

resultado = st.session_state.resultado

r1, r2, r3 = st.columns([0.8, 0.8, 1.6], gap="medium")

# ---------------- Deslizamento ----------------
with r1:
    with st.container(border=True):
        st.subheader("Deslizamento")
        des = resultado.get("deslizamento") if resultado else None
        if des:
            d = des["deslizamento"]
            kpi("Menor fator de segurança", fmt_fs(d["fs"]))
            selo(d["aprovado"])
            st.markdown(f'<div class="aux"><b>Combinação crítica:</b> {des["descricao"]}</div>', unsafe_allow_html=True)
            st.caption(linha_esforcos(des.get("esforcos")))
        else:
            kpi("Menor fator de segurança", "—")
            st.caption("Execute o cálculo para visualizar os resultados.")

# ---------------- Tombamento ----------------
with r2:
    with st.container(border=True):
        st.subheader("Tombamento")
        tom = resultado.get("tombamento") if resultado else None
        if tom:
            d = tom["tombamento"]
            kpi("Menor fator de segurança", fmt_fs(d["fs"]), f'FSx = {fmt_fs(d["fs_x"])} · FSy = {fmt_fs(d["fs_y"])}')
            selo(d["aprovado"])
            st.markdown(f'<div class="aux"><b>Combinação crítica:</b> {tom["descricao"]}</div>', unsafe_allow_html=True)
            st.caption(linha_esforcos(tom.get("esforcos")))
        else:
            kpi("Menor fator de segurança", "—")
            st.caption("Execute o cálculo para visualizar os resultados.")

# ---------------- Tensões ----------------
with r3:
    with st.container(border=True):
        st.subheader("Tensões e área comprimida")
        ten = resultado.get("tensao") if resultado else None

        if ten:
            dados_tensao = ten["tensao"]
            esf = ten["esforcos"]
            st.markdown(
                f'<div class="aux"><b>Combinação crítica:</b> {ten["descricao"]} &nbsp;|&nbsp; '
                f'N = {esf["N"]:.1f} kN · Mx = {esf["Mx"]:.1f} kN·m · My = {esf["My"]:.1f} kN·m</div>',
                unsafe_allow_html=True)

            # Indicadores
            area_pct = dados_tensao["percentual_area_comprimida"]
            sigma_max = dados_tensao["sigma_max"]
            sigma_media = dados_tensao["sigma_media"]
            sigma_adm = ten["sigma_adm"]
            atende = ten["atende_sigma_adm"]

            k1, k2, k3, k4 = st.columns(4, gap="small")
            with k1:
                kpi("Área comprimida", f"{area_pct:.1f}%" if np.isfinite(area_pct) else "N/D",
                    (dados_tensao.get("zona_descricao") or "").split(" — ")[0])
            with k2:
                kpi("Tensão máxima", f"{sigma_max:.1f}" if np.isfinite(sigma_max) else "N/D", "kPa")
            with k3:
                kpi("Tensão média", f"{sigma_media:.1f}" if np.isfinite(sigma_media) else "N/D", "kPa")
            with k4:
                kpi("Tensão admissível", f"{sigma_adm:.1f}", "kPa")

            if np.isfinite(sigma_max):
                if atende:
                    st.success(f"Atende: σmáx = {sigma_max:.1f} kPa ≤ σadm = {sigma_adm:.1f} kPa")
                else:
                    st.error(f"Não atende: σmáx = {sigma_max:.1f} kPa > σadm = {sigma_adm:.1f} kPa")
            else:
                st.error(dados_tensao.get("mensagem") or "Não foi possível verificar a tensão admissível.")

            # Gráfico original do FootingContact
            resultado_bruto = dados_tensao.get("resultado_bruto")

            if resultado_bruto is not None:
                # resultantes das demais combinações (a crítica fica em vermelho)
                outras = []
                for r in resultado["resultados_por_combinacao"]:
                    bruto = r["tensao"].get("resultado_bruto")
                    if r is ten or bruto is None:
                        continue
                    outras.append((r["combinacao"], bruto.ex, bruto.ey))

                fig_tensao = plot_result(resultado_bruto, outras=outras, rotulo=ten["combinacao"])

                st.pyplot(
                    fig_tensao,
                    use_container_width=True,
                )

                plt.close(fig_tensao)
            # if resultado_bruto is not None:
            #     try:
            #         figuras_antes = set(plt.get_fignums())
            #         fig_retorno = plot_result(resultado_bruto)
            #         if hasattr(fig_retorno, "savefig"):
            #             fig_tensao = fig_retorno
            #         else:
            #             novas = set(plt.get_fignums()) - figuras_antes
            #             fig_tensao = plt.figure(max(novas)) if novas else None

            #         if fig_tensao is not None:
            #             st.pyplot(fig_tensao, use_container_width=True)
            #             for numero in set(plt.get_fignums()) - figuras_antes:
            #                 plt.close(numero)
            #         else:
            #             st.warning("A função plot_result() não retornou uma figura Matplotlib identificável.")
            #     except Exception as erro:
            #         st.error(f"Erro ao gerar o gráfico de tensões: {erro}")
            else:
                st.caption("Sem distribuição de tensões para esta combinação.")

        #     # Tensões nos vértices (tabela compacta)
        #     vertices = dados_tensao.get("vertices")
        #     if isinstance(vertices, np.ndarray):
        #         vertices = vertices.tolist()
        #     if isinstance(vertices, dict):
        #         vertices = {
        #             str(k): (v.item() if isinstance(v, np.generic) else v.tolist() if isinstance(v, np.ndarray) else v)
        #             for k, v in vertices.items()
        #         }
        #     if vertices is not None:
        #         with st.expander("Tensões nos vértices"):
        #             if isinstance(vertices, dict):
        #                 try:
        #                     st.dataframe(
        #                         pd.DataFrame({"Vértice": list(vertices.keys()),
        #                                       "Tensão (kPa)": list(vertices.values())}),
        #                         hide_index=True, use_container_width=True)
        #                 except Exception:
        #                     st.write(vertices)
        #             else:
        #                 st.write(vertices)
        #     else:
        #         st.caption("O backend não disponibilizou as tensões nos vértices.")
        # else:
        #     st.info("Os resultados de tensão aparecerão após a execução do cálculo.")