from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from tensao import FootingContact, ContactType, ZONAS


# ============================================================
# MODELOS DE DADOS
# ============================================================

@dataclass(frozen=True)
class Geometria:
    a: float
    b: float
    ap: float
    bp: float
    a_ped: float
    b_ped: float
    h: float
    h0: float
    h_ped: float
    h_solo: float

    @property
    def area_base_inferior(self):
        return self.a * self.b

    @property
    def area_base_superior(self):
        return self.ap * self.bp

    @property
    def area_pedestal(self):
        return self.a_ped * self.b_ped

    @property
    def volume_rodape(self):
        return self.area_base_inferior * self.h0

    @property
    def volume_tronco_piramide(self):
        return (
            (self.h - self.h0) / 3
            * (
                self.area_base_inferior
                + self.area_base_superior
                + np.sqrt(
                    self.area_base_inferior
                    * self.area_base_superior
                )
            )
        )

    @property
    def volume_pedestal(self):
        return self.area_pedestal * self.h_ped

    @property
    def volume_sapata(self):
        return (
            self.volume_rodape
            + self.volume_tronco_piramide
        )

    @property
    def volume_solo(self):
        return (
            (self.area_base_inferior - self.area_pedestal)
            * self.h_solo
            - self.volume_tronco_piramide
        )


@dataclass(frozen=True)
class Concreto:
    peso_especifico: float


@dataclass(frozen=True)
class Solo:
    sigma_adm: float
    mu: float
    peso_especifico: float


# ============================================================
# PESOS
# ============================================================

def calcula_peso_concreto(
    geometria: Geometria,
    concreto: Concreto,
) -> float:
    return (geometria.volume_sapata + geometria.volume_pedestal) * concreto.peso_especifico


def calcula_peso_solo(
    geometria: Geometria,
    solo: Solo,
) -> float:
    return geometria.volume_solo * solo.peso_especifico


# ============================================================
# TRANSFORMAÇÃO DOS ESFORÇOS
# ============================================================

def transforma_esforcos(
    N,
    Mx,
    My,
    Fx,
    Fy,
    geometria,
    posicao,
    natureza,
    solo,
    concreto,
):
    """
    Transforma os esforços para a base da sapata.

    Convenção de saída:
        N_base > 0: compressão
        Mx, My: regra da mão direita

    Unidades:
        N, Fx, Fy: kN
        Mx, My: kN.m
        dimensões: m
        pesos específicos: kN/m³
    """

    if natureza == "Ações":
        multiplicador = 1
    elif natureza == "Reações":
        multiplicador = -1
    else:
        raise ValueError(
            f"Natureza inválida: {natureza!r}"
        )

    if posicao == "Topo do pedestal":
        altura = geometria.h + geometria.h_ped
    elif posicao == "Base da sapata":
        altura = 0.0
    else:
        raise ValueError(
            f"Posição inválida: {posicao!r}"
        )

    peso = (
        calcula_peso_solo(geometria, solo)
        + calcula_peso_concreto(geometria, concreto)
    )

    # Conversão dos esforços para a convenção adotada.
    N_convertido = -multiplicador * N
    Mx_convertido = multiplicador * Mx
    My_convertido = multiplicador * My
    Fx_convertido = multiplicador * Fx
    Fy_convertido = multiplicador * Fy

    # Transporte dos momentos para a base.
    N_base = N_convertido + peso

    Mx_base = Mx_convertido - Fy_convertido * altura
    My_base = My_convertido + Fx_convertido * altura

    return N_base, Mx_base, My_base

# def transforma_esforcos(
#     N,
#     Mx,
#     My,
#     Fx,
#     Fy,
#     geometria,
#     posicao,
#     natureza,
#     solo,
#     concreto,
# ):
#     """
#     Transforma os esforços para a base da sapata.

#     Mantém a convenção de sinais fornecida pelo usuário.
#     Validar essa convenção antes de usar os resultados.

#     Unidades esperadas:
#         N, Fx, Fy: kN
#         Mx, My: kN.m
#         dimensões: m
#         pesos específicos: kN/m³
#     """

#     if natureza == "Ações":
#         multiplicador = 1
#     elif natureza == "Reações":
#         multiplicador = -1
#     else:
#         raise ValueError(
#             f"Natureza inválida: {natureza!r}"
#         )

#     if posicao == "Topo do pedestal":
#         altura = geometria.h + geometria.h_ped
#     elif posicao == "Base da sapata":
#         altura = 0.0
#     else:
#         raise ValueError(
#             f"Posição inválida: {posicao!r}"
#         )

#     peso = (
#         calcula_peso_solo(geometria, solo)
#         + calcula_peso_concreto(geometria, concreto)
#     )

#     # Convenção preservada do código original.
#     N_base = -multiplicador * N + peso

#     Mx_base = multiplicador * (
#         Mx - Fy * altura
#     )

#     My_base = multiplicador * (
#         My + Fx * altura
#     )

#     return N_base, Mx_base, My_base


# ============================================================
# DESLIZAMENTO
# ============================================================

def verifica_deslizamento(
    N,
    Fx,
    Fy,
    solo: Solo,
    limite=1.5,
):
    """
    Retorna:
        força solicitante;
        força resistente;
        fator de segurança;
        aprovação.

    Pressupõe N positivo quando há compressão.
    """

    forca_solicitante = np.hypot(Fx, Fy)

    forca_resistente = max(N, 0.0) * solo.mu

    if forca_solicitante == 0:
        FS = np.inf if forca_resistente > 0 else np.nan
    else:
        FS = forca_resistente / forca_solicitante

    aprovado = bool(
        np.isfinite(FS) and FS >= limite
    ) or bool(np.isinf(FS))

    return (
        forca_solicitante,
        forca_resistente,
        FS,
        aprovado,
    )


# ============================================================
# TOMBAMENTO
# ============================================================

def verifica_tombamento(
    N,
    Mx,
    My,
    geometria: Geometria,
    limite=1.5,
):
    """
    Verificação simplificada em torno dos eixos centrais.

    A convenção dos eixos e dos momentos deve ser validada
    para a aplicação específica.
    """

    momento_resistente_x = max(N, 0.0) * geometria.b / 2
    momento_resistente_y = max(N, 0.0) * geometria.a / 2

    solicitante_x = abs(Mx)
    solicitante_y = abs(My)

    FSx = (
        momento_resistente_x / solicitante_x
        if solicitante_x > 0
        else np.inf
    )

    FSy = (
        momento_resistente_y / solicitante_y
        if solicitante_y > 0
        else np.inf
    )

    aprovado_x = FSx >= limite
    aprovado_y = FSy >= limite

    return (
        momento_resistente_x,
        momento_resistente_y,
        FSx,
        FSy,
        aprovado_x,
        aprovado_y,
    )


# ============================================================
# TENSÕES NO SOLO (tensao.py)
# ============================================================

def _tensao_sem_resultado(status, mensagem):
    """Dicionário de tensão para combinações em que não há distribuição de contato."""
    return {
        "status": status,
        "mensagem": mensagem,
        "zona": None,
        "zona_descricao": None,
        "sigma_max": np.nan,
        "sigma_min": np.nan,
        "sigma_media": np.nan,
        "area_comprimida": 0.0,
        "percentual_area_comprimida": 0.0,
        "vertices": None,
        "poligono_comprimido": None,
        "resultado_bruto": None,
    }


def extrai_resultado_tensao(result):
    """Converte o FootingResult de tensao.py no dicionário usado pelo app."""

    if result.contact_type == ContactType.NO_SOLUTION:
        dados = _tensao_sem_resultado(
            "SEM_SOLUCAO",
            "Resultante fora da base: não há equilíbrio com compressão "
            "exclusiva do solo (tombamento).",
        )
        dados["resultado_bruto"] = result
        return dados

    if result.contact_type == ContactType.ZONE_4:
        dados = _tensao_sem_resultado(
            "ZONA_4",
            "Resultante na zona 4 (menos de 50 % da base comprimida): "
            "não atende. As tensões não foram calculadas.",
        )
        dados["zona"] = result.zone
        dados["zona_descricao"] = ZONAS.get(result.zone)
        dados["percentual_area_comprimida"] = np.nan
        dados["resultado_bruto"] = result
        return dados

    return {
        "status": result.contact_type.value,
        "mensagem": None,
        "zona": result.zone,
        "zona_descricao": ZONAS.get(result.zone),
        "sigma_max": float(result.sigma_max),
        "sigma_min": float(result.sigma_min),
        "sigma_media": float(result.sigma_mean),
        "area_comprimida": float(result.contact_area),
        "percentual_area_comprimida": 100.0 * result.contact_ratio,
        "vertices": result.corner_pressures,
        "poligono_comprimido": result.contact_polygon,
        "resultado_bruto": result,
    }


def calcula_tensao(
    N,
    Mx,
    My,
    geometria: Geometria,
):
    """
    Executa o solver de contato sem tração (tensao.FootingContact).

    Convenção: N > 0 é compressão; ex = My/N e ey = -Mx/N.
    """

    if N <= 0:
        return _tensao_sem_resultado(
            "SEM_COMPRESSAO",
            "N na base ≤ 0: não há compressão na base da sapata.",
        )

    footing = FootingContact(
        B=geometria.a,
        L=geometria.b,
        N=N,
        Mx=Mx,
        My=My,
    )

    return extrai_resultado_tensao(footing.solve())


# ============================================================
# LEITURA DAS COMBINAÇÕES
# ============================================================

def _coluna(df, *candidatos, obrigatoria=True):
    """
    Localiza coluna ignorando maiúsculas/minúsculas
    e espaços externos.
    """
    mapa = {
        str(c).strip().casefold(): c
        for c in df.columns
    }

    for nome in candidatos:
        chave = nome.strip().casefold()

        if chave in mapa:
            return mapa[chave]

    if obrigatoria:
        raise ValueError(
            f"Coluna não encontrada. Esperada uma de: {candidatos}"
        )

    return None


def _valor_linha(linha, coluna, default=0.0):
    if coluna is None:
        return default

    valor = linha[coluna]

    if pd.isna(valor):
        return default

    return float(valor)


# ============================================================
# MOTOR PRINCIPAL
# ============================================================

def calcular_verificacoes(
    geometria: Geometria,
    solo: Solo,
    concreto: Concreto,
    combinacoes: pd.DataFrame,
    posicao: str,
    natureza: str,
) -> dict[str, Any]:

    if combinacoes.empty:
        raise ValueError(
            "Informe pelo menos uma combinação de carregamento."
        )

    if geometria.a <= 0 or geometria.b <= 0:
        raise ValueError(
            "As dimensões a e b devem ser positivas."
        )

    if geometria.ap <= 0 or geometria.bp <= 0:
        raise ValueError(
            "As dimensões ap e bp devem ser positivas."
        )

    if solo.mu < 0:
        raise ValueError(
            "O coeficiente de atrito não pode ser negativo."
        )

    if solo.sigma_adm <= 0:
        raise ValueError(
            "A tensão admissível deve ser positiva."
        )

    if geometria.volume_solo < 0:
        raise ValueError(
            "O volume de solo considerado ficou negativo. "
            "Confira a geometria e h_solo."
        )

    # Identificação flexível das colunas.
    col_N = _coluna(combinacoes, "N")
    col_Mx = _coluna(combinacoes, "Mx", "M_x", "MX")
    col_My = _coluna(combinacoes, "My", "M_y", "MY")

    col_Fx = _coluna(
        combinacoes, "Fx", "F_x", "FX",
        obrigatoria=False,
    )
    col_Fy = _coluna(
        combinacoes, "Fy", "F_y", "FY",
        obrigatoria=False,
    )

    col_id = _coluna(
        combinacoes,
        "Combinação", "Combinacao", "Comb.", "ID",
        obrigatoria=False,
    )

    col_desc = _coluna(
        combinacoes,
        "Descrição", "Descricao", "Nome",
        obrigatoria=False,
    )

    peso_sapata = calcula_peso_concreto(
        geometria, concreto
    )

    peso_solo = calcula_peso_solo(
        geometria, solo
    )

    resultados = []

    for indice, linha in combinacoes.iterrows():

        identificador = (
            linha[col_id]
            if col_id is not None
            else indice + 1
        )

        if isinstance(identificador, (float, np.floating)) and float(identificador).is_integer():
            identificador = int(identificador)

        descricao = (
            str(linha[col_desc])
            if col_desc is not None
            else str(identificador)
        )

        N = _valor_linha(linha, col_N)
        Mx = _valor_linha(linha, col_Mx)
        My = _valor_linha(linha, col_My)
        Fx = _valor_linha(linha, col_Fx)
        Fy = _valor_linha(linha, col_Fy)

        # ----------------------------------------------------
        # 1. Transformar esforços para a base da sapata
        # ----------------------------------------------------

        N_base, Mx_base, My_base = transforma_esforcos(
            N=N,
            Mx=Mx,
            My=My,
            Fx=Fx,
            Fy=Fy,
            geometria=geometria,
            posicao=posicao,
            natureza=natureza,
            solo=solo,
            concreto=concreto,
        )

        # Preserva as forças horizontais transformadas.
        # A transformação de sinal é coerente com a função
        # transforma_esforcos() existente.
        multiplicador = 1 if natureza == "Ações" else -1

        Fx_base = multiplicador * Fx
        Fy_base = multiplicador * Fy

        # ----------------------------------------------------
        # 2. Deslizamento
        # ----------------------------------------------------

        (
            forca_solicitante,
            forca_resistente,
            FS_deslizamento,
            aprovado_deslizamento,
        ) = verifica_deslizamento(
            N=N_base,
            Fx=Fx_base,
            Fy=Fy_base,
            solo=solo,
        )

        # ----------------------------------------------------
        # 3. Tombamento
        # ----------------------------------------------------

        (
            Mr_x,
            Mr_y,
            FS_tombamento_x,
            FS_tombamento_y,
            aprovado_tombamento_x,
            aprovado_tombamento_y,
        ) = verifica_tombamento(
            N=N_base,
            Mx=Mx_base,
            My=My_base,
            geometria=geometria,
        )

        FS_tombamento = min(
            FS_tombamento_x,
            FS_tombamento_y,
        )

        aprovado_tombamento = (
            aprovado_tombamento_x
            and aprovado_tombamento_y
        )

        # ----------------------------------------------------
        # 4. Tensões sem tração
        # ----------------------------------------------------

        tensao = calcula_tensao(
            N=N_base,
            Mx=Mx_base,
            My=My_base,
            geometria=geometria,
        )

        sigma_max = tensao["sigma_max"]

        atende_sigma_adm = (
            np.isfinite(sigma_max)
            and sigma_max <= solo.sigma_adm
        )

        # ----------------------------------------------------
        # 5. Guardar resultados da combinação
        # ----------------------------------------------------

        resultados.append({
            "combinacao": identificador,
            "descricao": descricao,

            "esforcos": {
                "N": N_base,
                "Mx": Mx_base,
                "My": My_base,
                "Fx": Fx_base,
                "Fy": Fy_base,
            },

            "esforcos_informados": {
                "N": N,
                "Mx": Mx,
                "My": My,
                "Fx": Fx,
                "Fy": Fy,
            },

            "deslizamento": {
                "forca_solicitante": forca_solicitante,
                "forca_resistente": forca_resistente,
                "fs": FS_deslizamento,
                "aprovado": aprovado_deslizamento,
            },

            "tombamento": {
                "momento_resistente_x": Mr_x,
                "momento_resistente_y": Mr_y,
                "fs_x": FS_tombamento_x,
                "fs_y": FS_tombamento_y,
                "fs": FS_tombamento,
                "aprovado": aprovado_tombamento,
            },

            "tensao": tensao,
            "sigma_adm": solo.sigma_adm,
            "atende_sigma_adm": atende_sigma_adm,
        })

    # --------------------------------------------------------
    # 6. Identificar combinações críticas
    # --------------------------------------------------------

    validos_deslizamento = [
        r for r in resultados
        if not np.isnan(r["deslizamento"]["fs"])
    ]

    validos_tombamento = [
        r for r in resultados
        if not np.isnan(r["tombamento"]["fs"])
    ]

    critico_deslizamento = (
        min(
            validos_deslizamento,
            key=lambda r: r["deslizamento"]["fs"],
        )
        if validos_deslizamento
        else None
    )

    critico_tombamento = (
        min(
            validos_tombamento,
            key=lambda r: r["tombamento"]["fs"],
        )
        if validos_tombamento
        else None
    )

    # Combinação sem equilíbrio, sem compressão ou na zona 4 reprova e é
    # pior que qualquer tensão finita; só depois vale a maior σmáx.
    sem_contato = [
        r for r in resultados
        if not np.isfinite(r["tensao"]["sigma_max"])
    ]

    if sem_contato:
        critico_tensao = sem_contato[0]
    else:
        critico_tensao = max(
            resultados,
            key=lambda r: r["tensao"]["sigma_max"],
        )

    # --------------------------------------------------------
    # 7. Retorno principal
    # --------------------------------------------------------

    return {
        "peso_sapata": peso_sapata,
        "peso_solo": peso_solo,

        "deslizamento": critico_deslizamento,
        "tombamento": critico_tombamento,
        "tensao": critico_tensao,

        "resultados_por_combinacao": resultados,

        "criterios": {
            "fs_minimo_deslizamento": 1.5,
            "fs_minimo_tombamento": 1.5,
            "sigma_adm": solo.sigma_adm,
        },
    }
