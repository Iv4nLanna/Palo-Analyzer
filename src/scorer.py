from dataclasses import dataclass
from statistics import mean, pstdev
from typing import Dict, List, Optional

from config import (
    DEFAULT_BLOCK_SIZE_LINES,
    DEFAULT_ERROR_PENALTY,
    DEFAULT_TIME_PER_BLOCK_SECONDS,
    DEFAULT_VARIABILITY_PENALTY_FACTOR,
)


@dataclass
class ScoreConfig:
    time_per_block_seconds: float = DEFAULT_TIME_PER_BLOCK_SECONDS
    block_size_lines: int = DEFAULT_BLOCK_SIZE_LINES
    error_penalty: float = DEFAULT_ERROR_PENALTY
    variability_penalty_factor: float = DEFAULT_VARIABILITY_PENALTY_FACTOR


def _mk_class(nivel: str, faixa: str, regra_id: str) -> Dict[str, str]:
    return {"nivel": nivel, "faixa": faixa, "regra_id": regra_id}


def _classify_productivity(total: int) -> Dict[str, str]:
    if total > 862:
        return _mk_class("Superior ou Muito Alta", "total > 862", "PROD_001")
    if 607 <= total <= 754:
        return _mk_class("Medio Superior ou Alta", "607-754", "PROD_002")
    if 377 <= total <= 571:
        return _mk_class("Media", "377-571", "PROD_003")
    if 267 <= total <= 348:
        return _mk_class("Medio Inferior ou Baixa", "267-348", "PROD_004")
    if total < 230:
        return _mk_class("Inferior ou Lento", "< 230", "PROD_005")
    return _mk_class("Faixa de transicao", "230-266, 349-376, 572-606 ou 755-862", "PROD_999")


def _split_blocks(line_counts: List[int], block_size_lines: int) -> List[int]:
    if not line_counts:
        return []
    size = max(1, int(block_size_lines))
    blocks: List[int] = []
    for i in range(0, len(line_counts), size):
        blocks.append(int(sum(line_counts[i : i + size])))
    return blocks


def _compute_nor(block_totals: List[int]) -> float:
    if len(block_totals) <= 1:
        return 0.0
    diffs = [abs(block_totals[i] - block_totals[i - 1]) for i in range(1, len(block_totals))]
    return float(sum(diffs) / len(diffs))


def _classify_rhythm(nor: Optional[float]) -> Dict[str, str]:
    if nor is None:
        return _mk_class("Nao calculado", "sem dados", "RIT_000")
    if nor >= 15.6:
        return _mk_class("Muito Alto", ">= 15.6", "RIT_001")
    if 8.6 <= nor <= 12.8:
        return _mk_class("Alto", "8.6-12.8", "RIT_002")
    if 4.2 <= nor <= 8.0:
        return _mk_class("Medio", "4.2-8.0", "RIT_003")
    if 2.6 <= nor <= 3.8:
        return _mk_class("Baixo", "2.6-3.8", "RIT_004")
    if 1.2 <= nor <= 2.0:
        return _mk_class("Muito Baixo", "1.2-2.0", "RIT_005")
    return _mk_class("Intermediario", "fora das faixas centrais da apostila", "RIT_999")


def _classify_spacing_mm(spacing_mm: Optional[float]) -> Dict[str, str]:
    if spacing_mm is None:
        return _mk_class("Nao calculado", "sem dados", "DISTPALO_000")
    if spacing_mm >= 4.8:
        return _mk_class("Muito Aumentada ou Muito Ampla", ">= 4.8 mm", "DISTPALO_001")
    if 4.0 <= spacing_mm <= 4.7:
        return _mk_class("Aumentada ou Ampla", "4.0-4.7 mm", "DISTPALO_002")
    if 2.3 <= spacing_mm <= 3.9:
        return _mk_class("Normal ou Media", "2.3-3.9 mm", "DISTPALO_003")
    if 1.5 <= spacing_mm <= 2.2:
        return _mk_class("Diminuida ou Estreita", "1.5-2.2 mm", "DISTPALO_004")
    if spacing_mm < 1.4:
        return _mk_class("Muito Diminuida ou Muito Estreita", "< 1.4 mm", "DISTPALO_005")
    return _mk_class("Intermediaria", "fora das faixas centrais da apostila", "DISTPALO_999")


def _classify_stroke_size(height_mm: Optional[float]) -> Dict[str, str]:
    if height_mm is None:
        return _mk_class("Nao calculado", "sem dados", "TAM_000")
    if height_mm > 9.8:
        return _mk_class("Muito Aumentado ou Muito Grande", "> 9.8 mm", "TAM_001")
    if 8.5 <= height_mm <= 9.7:
        return _mk_class("Aumentado ou Grande", "8.5-9.7 mm", "TAM_002")
    if 5.7 <= height_mm <= 8.4:
        return _mk_class("Normal ou Medio", "5.7-8.4 mm", "TAM_003")
    if 4.3 <= height_mm <= 5.6:
        return _mk_class("Diminuido ou Pequeno", "4.3-5.6 mm", "TAM_004")
    return _mk_class("Muito Diminuido ou Muito Pequeno", "< 4.3 mm", "TAM_005")


def _classify_line_spacing_mm(line_spacing_mm: Optional[float]) -> Dict[str, str]:
    if line_spacing_mm is None:
        return _mk_class("Nao calculado", "sem dados", "DISTLIN_000")
    if line_spacing_mm >= 8.9:
        return _mk_class("Muito Aumentada ou Afastada", ">= 8.9 mm", "DISTLIN_001")
    if 6.9 <= line_spacing_mm <= 8.8:
        return _mk_class("Aumentada ou Afastada", "6.9-8.8 mm", "DISTLIN_002")
    if 3.0 <= line_spacing_mm <= 6.8:
        return _mk_class("Normal ou Media", "3.0-6.8 mm", "DISTLIN_003")
    if 1.1 <= line_spacing_mm <= 2.9:
        return _mk_class("Diminuida, Estreita ou Proxima", "1.1-2.9 mm", "DISTLIN_004")
    if 0.0 <= line_spacing_mm <= 1.0:
        return _mk_class("Muito Diminuida", "0.0-1.0 mm", "DISTLIN_005")
    return _mk_class("Linhas tocando/sobrepostas", "< 0.0 mm", "DISTLIN_006")


def _classify_line_direction(angle_deg: Optional[float]) -> Dict[str, str]:
    if angle_deg is None:
        return _mk_class("Nao calculado", "sem dados", "DIRLIN_000")
    if angle_deg >= 3.1:
        return _mk_class("Muito Ascendente", ">= +3.1 graus", "DIRLIN_001")
    if 1.5 <= angle_deg <= 3.0:
        return _mk_class("Ascendente", "+1.5 a +3.0 graus", "DIRLIN_002")
    if -2.0 <= angle_deg <= 1.4:
        return _mk_class("Horizontal ou Retilinea Normal", "-2.0 a +1.4 graus", "DIRLIN_003")
    if -3.5 <= angle_deg <= -2.0:
        return _mk_class("Descendente", "-3.5 a -2.0 graus", "DIRLIN_004")
    return _mk_class("Muito Descendente", "< -3.5 graus", "DIRLIN_005")


def _classify_stroke_inclination(angle_deg: Optional[float]) -> Dict[str, str]:
    if angle_deg is None:
        return _mk_class("Nao calculado", "sem dados", "INCPALO_000")
    if angle_deg >= 99.8:
        return _mk_class("Muito inclinado para a Direita", ">= 99.8 graus", "INCPALO_001")
    if 94.5 <= angle_deg < 99.8:
        return _mk_class("Inclinado para a Direita", "94.5-99.7 graus", "INCPALO_002")
    if 83.8 <= angle_deg <= 94.4:
        return _mk_class("Vertical ou Reta", "83.8-94.4 graus", "INCPALO_003")
    if 78.5 <= angle_deg < 83.8:
        return _mk_class("Inclinado para a Esquerda", "78.5-83.7 graus", "INCPALO_004")
    return _mk_class("Muito inclinado para a Esquerda", "< 78.5 graus", "INCPALO_005")


def _classify_margin_left(mm: Optional[float]) -> Dict[str, str]:
    if mm is None:
        return _mk_class("Nao calculado", "sem dados", "MARGEME_000")
    if mm >= 13.8:
        return _mk_class("Muito Aumentada", ">= 13.8 mm", "MARGEME_001")
    if 10.9 <= mm <= 13.7:
        return _mk_class("Aumentada ou Larga", "10.9-13.7 mm", "MARGEME_002")
    if 4.9 <= mm <= 10.8:
        return _mk_class("Normal ou Media", "4.9-10.8 mm", "MARGEME_003")
    if 1.9 <= mm <= 4.8:
        return _mk_class("Diminuida ou Estreita", "1.9-4.8 mm", "MARGEME_004")
    return _mk_class("Muito Diminuida ou Estreita", "<= 1.8 mm", "MARGEME_005")


def _classify_margin_right(mm: Optional[float]) -> Dict[str, str]:
    if mm is None:
        return _mk_class("Nao calculado", "sem dados", "MARGEMD_000")
    if mm >= 8.7:
        return _mk_class("Aumentada ou Larga", ">= 8.7 mm", "MARGEMD_001")
    if 1.8 <= mm <= 8.6:
        return _mk_class("Normal", "1.8-8.6 mm", "MARGEMD_002")
    return _mk_class("Diminuida", "<= 1.7 mm", "MARGEMD_003")


def _classify_margin_top(mm: Optional[float]) -> Dict[str, str]:
    if mm is None:
        return _mk_class("Nao calculado", "sem dados", "MARGEMS_000")
    if mm >= 8.5:
        return _mk_class("Aumentada", ">= 8.5 mm", "MARGEMS_001")
    if 2.4 <= mm <= 8.4:
        return _mk_class("Normal", "2.4-8.4 mm", "MARGEMS_002")
    return _mk_class("Diminuida", "<= 2.3 mm", "MARGEMS_003")


def _classify_pressure(pressure_level: str) -> Dict[str, str]:
    level = (pressure_level or "").strip().lower()
    if "forte" in level:
        return _mk_class("Forte", "qualitativa", "PRESS_001")
    if "media" in level or "normal" in level:
        return _mk_class("Media ou Normal", "qualitativa", "PRESS_002")
    if "leve" in level or "fraca" in level or "delicada" in level:
        return _mk_class("Fraca, Leve ou Delicada", "qualitativa", "PRESS_003")
    if "irregular" in level:
        return _mk_class("Irregular", "qualitativa", "PRESS_004")
    mapping = {
        "forte": _mk_class("Forte", "qualitativa", "PRESS_001"),
        "media": _mk_class("Media ou Normal", "qualitativa", "PRESS_002"),
        "normal": _mk_class("Media ou Normal", "qualitativa", "PRESS_002"),
        "leve": _mk_class("Fraca, Leve ou Delicada", "qualitativa", "PRESS_003"),
        "fraca": _mk_class("Fraca, Leve ou Delicada", "qualitativa", "PRESS_003"),
        "irregular": _mk_class("Irregular", "qualitativa", "PRESS_004"),
    }
    return mapping.get(level, _mk_class("Nao calculado", "sem dados", "PRESS_000"))


def _classify_stroke_quality(quality_level: str) -> Dict[str, str]:
    level = (quality_level or "").strip().lower()
    if "reto" in level or "firme" in level:
        return _mk_class("Tracos Firmes ou Retos", "qualitativa", "TRACO_001")
    if "curv" in level or "froux" in level or "brando" in level:
        return _mk_class("Tracos Frouxos, Curvos ou Brandos", "qualitativa", "TRACO_002")
    if "descontin" in level or "interromp" in level:
        return _mk_class("Interrompida ou Descontinua", "qualitativa", "TRACO_003")
    mapping = {
        "reta": _mk_class("Tracos Firmes ou Retos", "qualitativa", "TRACO_001"),
        "firme": _mk_class("Tracos Firmes ou Retos", "qualitativa", "TRACO_001"),
        "curva": _mk_class("Tracos Frouxos, Curvos ou Brandos", "qualitativa", "TRACO_002"),
        "frouxa": _mk_class("Tracos Frouxos, Curvos ou Brandos", "qualitativa", "TRACO_002"),
        "descontinua": _mk_class("Interrompida ou Descontinua", "qualitativa", "TRACO_003"),
        "interrompida": _mk_class("Interrompida ou Descontinua", "qualitativa", "TRACO_003"),
    }
    return mapping.get(level, _mk_class("Nao calculado", "sem dados", "TRACO_000"))


def _classify_organization(level_text: str) -> Dict[str, str]:
    level = (level_text or "").strip().lower()
    if "muito boa" in level:
        return _mk_class("Muito Boa", "qualitativa", "ORG_001")
    if level == "boa" or " boa" in f" {level}":
        return _mk_class("Boa", "qualitativa", "ORG_002")
    if "regular" in level:
        return _mk_class("Regular", "qualitativa", "ORG_003")
    if "muito ruim" in level:
        return _mk_class("Muito Ruim", "qualitativa", "ORG_005")
    if "ruim" in level:
        return _mk_class("Ruim", "qualitativa", "ORG_004")
    mapping = {
        "muito boa": _mk_class("Muito Boa", "qualitativa", "ORG_001"),
        "boa": _mk_class("Boa", "qualitativa", "ORG_002"),
        "regular": _mk_class("Regular", "qualitativa", "ORG_003"),
        "ruim": _mk_class("Ruim", "qualitativa", "ORG_004"),
        "muito ruim": _mk_class("Muito Ruim", "qualitativa", "ORG_005"),
    }
    return mapping.get(level, _mk_class("Nao calculado", "sem dados", "ORG_000"))


def parse_irregularities_text(text: str) -> List[str]:
    if not text:
        return []
    raw = [x.strip().lower() for x in text.replace(",", ";").split(";")]
    return [x for x in raw if x]


def _evaluate_irregularities(items: List[str]) -> List[Dict[str, str]]:
    mapping = {
        "tremor inicial": ("IRREG_001", "Tremor Inicial", "Inseguranca inicial diante de situacoes novas."),
        "tremor constante": ("IRREG_002", "Tremor Constante", "Sugere alteracao persistente no controle motor/emocional."),
        "tremor acentuado": ("IRREG_003", "Tremor Acentuado", "Oscilacao intensa com possiveis sinais neurologicos/alta tensao."),
        "gancho inferior direito": ("IRREG_004", "Gancho Inferior Direito", "Pode reagir com mal-humor em conflitos."),
        "gancho inferior esquerdo": ("IRREG_005", "Gancho Inferior Esquerdo", "Tendencia a autocritica agressiva e dificuldade de esquecer conflitos."),
        "gancho superior direito": ("IRREG_006", "Gancho Superior Direito", "Tendencia a explosoes e critica aos outros."),
        "gancho superior esquerdo": ("IRREG_007", "Gancho Superior Esquerdo", "Tendencia a autocobranca e autopunicao."),
        "lacos": ("IRREG_008", "Lacos", "Tendencia a conter energia sem externalizacao adequada."),
        "palos quebrados": ("IRREG_009", "Palos Quebrados", "Indicador de irregularidade importante do tracado."),
        "chamines": ("IRREG_010", "Chamines", "Pode associar-se a ansiedade quando confirmado por outros sinais."),
        "tracado repassado": ("IRREG_011", "Tracado Repassado", "Sugere dificuldade de decisao e rigidez mental."),
        "correcoes": ("IRREG_012", "Correcoes e Retoques", "Sinal de inseguranca e insatisfacao."),
    }
    findings: List[Dict[str, str]] = []
    for item in items:
        if item in mapping:
            rid, label, interp = mapping[item]
            findings.append({"item": label, "regra_id": rid, "interpretacao": interp})
    return findings


def _shape_classification(block_totals: List[int], nor: Optional[float]) -> str:
    if nor is None or len(block_totals) < 3:
        return "Indeterminado"

    first = block_totals[0]
    last = block_totals[-1]
    mid_idx = len(block_totals) // 2
    mid = block_totals[mid_idx]

    if nor <= 6:
        return "Regular"
    if last > first and all(block_totals[i] >= block_totals[i - 1] for i in range(1, len(block_totals))):
        return "Ascendente"
    if first > last and all(block_totals[i] <= block_totals[i - 1] for i in range(1, len(block_totals))):
        return "Descendente"

    end_mean = (first + last) / 2.0
    if mid > end_mean:
        return "Convexa"
    if mid < end_mean:
        return "Concava"
    return "Irregular"


def _quality_interpretation(productivity_level: str, nor: Optional[float], shape: str) -> str:
    if nor is None:
        return "Nao classificado"

    prod_low = productivity_level in {"Inferior ou Lento", "Medio Inferior ou Baixa"}
    prod_med = productivity_level == "Media"

    if 4 <= nor <= 6 and prod_med:
        return "Equilibrado"
    if 0 <= nor <= 3 and (prod_med or prod_low):
        return "Rigido"
    if nor > 6 and shape == "Ascendente" and prod_med:
        return "Ascendente ou Crescente"
    if nor > 6 and shape == "Descendente":
        return "Descendente ou Decrescente"
    if nor > 6 and shape == "Convexa":
        return "Convexa"
    if nor > 6 and shape == "Concava":
        return "Concava"
    if nor > 6:
        return "Irregular ou Oscilante"
    return "Nao classificado"


def _nor_productivity_notes(total: int, nor: Optional[float]) -> List[str]:
    if nor is None:
        return []

    notes: List[str] = []

    if total < 377:
        if nor < 5:
            notes.append("Produtividade abaixo da media com NOR < 5: tendencia a regularidade e estabilidade.")
        if nor > 8:
            notes.append("Produtividade abaixo da media com NOR > 8: tendencia a instabilidade emocional.")
        if nor > 15:
            notes.append("Produtividade abaixo da media com NOR > 15: tendencia a maior emotividade e descontrole.")

    if 377 <= total <= 754:
        if nor < 5:
            notes.append("Produtividade media/alta com NOR < 5: bom equilibrio ritmico.")
        if 8 <= nor <= 10:
            notes.append("Produtividade media/alta com NOR 8-10: rapidez com queda de qualidade/ritmo.")
        if nor > 15:
            notes.append("Produtividade media/alta com NOR > 15: possivel descontrole da atividade.")

    if total > 862:
        if nor < 6:
            notes.append("Produtividade muito alta com NOR < 6: rapidez com melhor controle.")
        if nor > 8:
            notes.append("Produtividade muito alta com NOR > 8: rapidez com menor controle.")
        if nor > 12:
            notes.append("Produtividade muito alta com NOR > 12: risco de precipitacao e baixa precisao.")

    return notes


def _classify_speed_group(total: int) -> str:
    if total < 377:
        return "lentidao"
    if total > 571:
        return "rapidez"
    return "normal"


def _apply_reasoning_adjustment(text: str, reasoning_level: str) -> str:
    if reasoning_level == "medio_inferior_ou_inferior":
        return text.replace(" e facilidade para resolver problemas", "")
    return text


def _prod_order_interpretation(total: int, order_pattern: str, reasoning_level: str) -> str:
    speed_group = _classify_speed_group(total)

    if speed_group == "lentidao":
        if order_pattern == "ordenados":
            return "Lentidao com palos ordenados: boa capacidade de observar, ordenar e classificar; aptidao para reproduzir mais do que criar."
        if order_pattern == "desordenados":
            return "Lentidao com palos desordenados: sugere dificuldade de compreensao e organizacao da tarefa."
        return "Lentidao: rendimento abaixo do esperado para o grupo de referencia."

    if speed_group == "normal":
        text = "Numero de palos em faixa normal: capacidade de executar tarefas com vivacidade, adaptacao as situacoes e facilidade para resolver problemas."
        return _apply_reasoning_adjustment(text, reasoning_level)

    if order_pattern == "desordenados":
        return "Rapidez com palos desordenados: tende a valorizar rapidez acima da qualidade do trabalho."

    text = "Rapidez com palos ordenados: capacidade de executar tarefas com vivacidade, adaptacao as situacoes e facilidade para resolver problemas."
    return _apply_reasoning_adjustment(text, reasoning_level)


def _rhythm_text(rhythm_level: str) -> str:
    mapping = {
        "Muito Alto": "Ritmo muito alto: grandes variacoes no desempenho.",
        "Alto": "Ritmo alto: flutuacoes e instabilidade no desempenho das tarefas.",
        "Medio": "Ritmo medio: boa adaptacao a tarefas rotineiras.",
        "Baixo": "Ritmo baixo: estabilidade no ritmo de producao com certa uniformidade.",
        "Muito Baixo": "Ritmo muito baixo: alta regularidade, baixa oscilacao e tendencia a rigidez.",
    }
    return mapping.get(rhythm_level, "Ritmo sem classificacao conclusiva.")


def _trait_entry(name: str, classification: Dict[str, str], interpretation: str) -> Dict[str, str]:
    return {
        "dimensao": name,
        "nivel": classification.get("nivel", "Nao classificado"),
        "faixa": classification.get("faixa", "sem dados"),
        "regra_id": classification.get("regra_id", "N/A"),
        "interpretacao": interpretation,
    }


def _productivity_personality_text(level: str) -> str:
    mapping = {
        "Superior ou Muito Alta": "Rendimento muito acima do esperado, com alta capacidade produtiva.",
        "Medio Superior ou Alta": "Rendimento acima do esperado para a faixa de referencia.",
        "Media": "Rendimento adequado ao esperado para a funcao.",
        "Medio Inferior ou Baixa": "Rendimento abaixo do esperado, com menor produtividade.",
        "Inferior ou Lento": "Rendimento muito abaixo do esperado, com producao deficiente.",
        "Faixa de transicao": "Resultado em faixa intermediaria de transicao entre categorias.",
    }
    return mapping.get(level, "Sem interpretacao conclusiva para produtividade.")


def _rhythm_personality_text(level: str) -> str:
    mapping = {
        "Muito Alto": "Grandes oscilacoes no desempenho e tendencia a instabilidade de ritmo.",
        "Alto": "Flutuacoes no ritmo, com menor constancia na execucao.",
        "Medio": "Adaptacao razoavel a tarefas rotineiras.",
        "Baixo": "Ritmo estavel, com maior uniformidade de producao.",
        "Muito Baixo": "Alta regularidade na execucao, com baixa oscilacao e possivel rigidez.",
        "Intermediario": "Ritmo em faixa intermediaria, requer leitura conjunta com outros indicadores.",
    }
    return mapping.get(level, "Sem interpretacao conclusiva para ritmo.")


def _spacing_personality_text(level: str) -> str:
    mapping = {
        "Normal ou Media": "Boa organizacao, metodo e foco em objetivos.",
        "Aumentada ou Ampla": "Maior extroversao e necessidade de contato/apoio externo.",
        "Muito Aumentada ou Muito Ampla": "Tendencia a dispersao e necessidade de chamar atencao.",
        "Diminuida ou Estreita": "Perfil mais reservado, cuidadoso e autoexigente.",
        "Muito Diminuida ou Muito Estreita": "Tendencia a desconfiança, ciumes e foco excessivo em detalhes.",
        "Intermediaria": "Padrao intermediario, exigindo correlacao com outros achados.",
    }
    return mapping.get(level, "Sem interpretacao conclusiva para distancia entre palos.")


def _stroke_size_personality_text(level: str) -> str:
    mapping = {
        "Normal ou Medio": "Boa adaptacao ao meio social.",
        "Aumentado ou Grande": "Perfil mais expansivo, autoconfiante e ambicioso.",
        "Muito Aumentado ou Muito Grande": "Tendencia a exibicionismo e atitudes mais extravagantes.",
        "Diminuido ou Pequeno": "Maior introversao, concentracao e foco em detalhes.",
        "Muito Diminuido ou Muito Pequeno": "Tendencia a inseguranca e sentimento de inferioridade.",
    }
    return mapping.get(level, "Sem interpretacao conclusiva para tamanho dos palos.")


def _line_spacing_personality_text(level: str) -> str:
    mapping = {
        "Normal ou Media": "Relacao interpessoal equilibrada, com limites adequados.",
        "Aumentada ou Afastada": "Relacoes mais formais e cautelosas; prefere maior distancia social.",
        "Muito Aumentada ou Afastada": "Excesso de cautela e maior afastamento interpessoal.",
        "Diminuida, Estreita ou Proxima": "Busca contato social frequente, com risco de excesso de proximidade.",
        "Muito Diminuida": "Contato intenso com menor percepcao de limites interpessoais.",
        "Linhas tocando/sobrepostas": "Dificuldade acentuada em limites interpessoais.",
    }
    return mapping.get(level, "Sem interpretacao conclusiva para distancia entre linhas.")


def _line_direction_personality_text(level: str) -> str:
    mapping = {
        "Horizontal ou Retilinea Normal": "Tendencia a comportamento mais equilibrado e convencional.",
        "Ascendente": "Maior iniciativa, dinamismo e otimismo diante de tarefas.",
        "Muito Ascendente": "Impulso elevado, com risco de exagero e menor realismo.",
        "Descendente": "Tendencia a queda de energia/esforco diante de dificuldades.",
        "Muito Descendente": "Indicador de desanimo acentuado e baixo vigor de continuidade.",
    }
    return mapping.get(level, "Sem interpretacao conclusiva para direcao das linhas.")


def _quality_personality_text(level: str) -> str:
    mapping = {
        "Equilibrado": "Execucao uniforme e estavel.",
        "Rigido": "Maior rigidez de estilo e controle.",
        "Ascendente ou Crescente": "Dinamismo com aumento progressivo de rendimento.",
        "Descendente ou Decrescente": "Inicia com mais energia e reduz ao longo da tarefa.",
        "Convexa": "Aumenta producao no meio e perde folego ao final.",
        "Concava": "Oscila com recuperacao de rendimento apos queda inicial.",
        "Irregular ou Oscilante": "Variabilidade marcante no ritmo de execucao.",
    }
    return mapping.get(level, "Sem interpretacao conclusiva para qualidade do rendimento.")


def _inclination_personality_text(level: str) -> str:
    mapping = {
        "Vertical ou Reta": "Perfil mais reservado e objetivo nas interacoes.",
        "Inclinado para a Direita": "Maior extroversao e busca de contato social.",
        "Muito inclinado para a Direita": "Subjetividade elevada e tendencia a impulsos expansivos.",
        "Inclinado para a Esquerda": "Maior introversao, reserva e cautela social.",
        "Muito inclinado para a Esquerda": "Reserva acentuada e maior tendencia ao isolamento.",
    }
    return mapping.get(level, "Sem interpretacao conclusiva para inclinacao dos palos.")


def _margin_left_personality_text(level: str) -> str:
    mapping = {
        "Normal ou Media": "Interesse por iniciativa com controle de responsabilidades.",
        "Aumentada ou Larga": "Maior extroversao e menor foco em obrigacoes.",
        "Muito Aumentada": "Tendencia a despreocupacao com obrigacoes e limite financeiro.",
        "Diminuida ou Estreita": "Perfil mais recatado e reflexivo para decidir.",
        "Muito Diminuida ou Estreita": "Reserva social acentuada e alta prudencia.",
    }
    return mapping.get(level, "Sem interpretacao conclusiva para margem esquerda.")


def _margin_right_personality_text(level: str) -> str:
    mapping = {
        "Normal": "Adaptacao social funcional diante de novas situacoes.",
        "Aumentada ou Larga": "Maior dificuldade de adaptacao ao novo e exposicao.",
        "Diminuida": "Perfil mais dinamico, com risco de precipitacao em decisoes.",
    }
    return mapping.get(level, "Sem interpretacao conclusiva para margem direita.")


def _margin_top_personality_text(level: str) -> str:
    mapping = {
        "Normal": "Relacao de respeito adequada com figuras de autoridade.",
        "Aumentada": "Postura defensiva e distanciamento em relacao a autoridade.",
        "Diminuida": "Tendencia a menor delimitacao no contato com autoridade.",
    }
    return mapping.get(level, "Sem interpretacao conclusiva para margem superior.")


def _pressure_personality_text(level: str) -> str:
    mapping = {
        "Forte": "Maior vigor, com possibilidade de menor precisao fina.",
        "Media ou Normal": "Equilibrio entre energia e planejamento.",
        "Fraca, Leve ou Delicada": "Maior delicadeza e menor disposicao para esforco fisico.",
        "Irregular": "Instabilidade de energia e persistencia na tarefa.",
    }
    return mapping.get(level, "Sem interpretacao conclusiva para pressao.")


def _stroke_quality_personality_text(level: str) -> str:
    mapping = {
        "Tracos Firmes ou Retos": "Maior determinacao e objetividade na conduta.",
        "Tracos Frouxos, Curvos ou Brandos": "Maior flexibilidade com menor firmeza de imposicao.",
        "Interrompida ou Descontinua": "Pode indicar tensao/ansiedade e oscilacao de continuidade.",
    }
    return mapping.get(level, "Sem interpretacao conclusiva para qualidade do tracado.")


def _organization_personality_text(level: str) -> str:
    mapping = {
        "Muito Boa": "Elevado cuidado com ordem, metodo e apresentacao.",
        "Boa": "Boa organizacao e metodo de execucao.",
        "Regular": "Organizacao intermediaria com limites parcialmente oscilantes.",
        "Ruim": "Baixa objetividade e metodo na execucao.",
        "Muito Ruim": "Necessidade de supervisao para tarefas de ordem e metodo.",
    }
    return mapping.get(level, "Sem interpretacao conclusiva para organizacao.")


def build_personality_traits(
    classes: Dict,
    order_pattern: str = "nao_informado",
    reasoning_level: str = "nao_informado",
    total: Optional[int] = None,
) -> List[Dict[str, str]]:
    traits: List[Dict[str, str]] = []

    prod = classes.get("produtividade", {})
    rit = classes.get("ritmo", {})
    dist = classes.get("distancia", {})
    tam = classes.get("tamanho_palos", {})
    dist_l = classes.get("distancia_entre_linhas", {})
    dir_l = classes.get("direcao_linhas", {})
    inc_p = classes.get("inclinacao_palos", {})
    marg_e = classes.get("margem_esquerda", {})
    marg_d = classes.get("margem_direita", {})
    marg_s = classes.get("margem_superior", {})
    press = classes.get("pressao", {})
    qual_t = classes.get("qualidade_tracado", {})
    org = classes.get("organizacao", {})
    qual = classes.get("qualidade_rendimento", "Nao classificado")

    traits.append(_trait_entry("Produtividade", prod, _productivity_personality_text(prod.get("nivel", ""))))
    traits.append(_trait_entry("Ritmo (NOR)", rit, _rhythm_personality_text(rit.get("nivel", ""))))
    traits.append(_trait_entry("Distancia entre palos", dist, _spacing_personality_text(dist.get("nivel", ""))))
    traits.append(_trait_entry("Tamanho dos palos", tam, _stroke_size_personality_text(tam.get("nivel", ""))))
    traits.append(
        _trait_entry(
            "Distancia entre linhas",
            dist_l,
            _line_spacing_personality_text(dist_l.get("nivel", "")),
        )
    )
    traits.append(_trait_entry("Inclinacao dos palos", inc_p, _inclination_personality_text(inc_p.get("nivel", ""))))
    traits.append(_trait_entry("Direcao das linhas", dir_l, _line_direction_personality_text(dir_l.get("nivel", ""))))
    traits.append(_trait_entry("Margem esquerda", marg_e, _margin_left_personality_text(marg_e.get("nivel", ""))))
    traits.append(_trait_entry("Margem direita", marg_d, _margin_right_personality_text(marg_d.get("nivel", ""))))
    traits.append(_trait_entry("Margem superior", marg_s, _margin_top_personality_text(marg_s.get("nivel", ""))))
    traits.append(_trait_entry("Pressao", press, _pressure_personality_text(press.get("nivel", ""))))
    traits.append(
        _trait_entry("Qualidade do tracado", qual_t, _stroke_quality_personality_text(qual_t.get("nivel", "")))
    )
    traits.append(_trait_entry("Organizacao/Ordem", org, _organization_personality_text(org.get("nivel", ""))))
    traits.append(
        {
            "dimensao": "Qualidade do rendimento",
            "nivel": qual,
            "faixa": "qualitativa",
            "interpretacao": _quality_personality_text(qual),
        }
    )

    if total is not None:
        traits.append(
            {
                "dimensao": "Ordem x Velocidade",
                "nivel": order_pattern,
                "faixa": "qualitativa",
                "interpretacao": _prod_order_interpretation(int(total), order_pattern, reasoning_level),
            }
        )

    return traits


def parse_block_totals_text(text: str) -> List[int]:
    if not text:
        return []
    parts = [p.strip() for p in text.split(";")]
    out = []
    for p in parts:
        if p:
            out.append(int(p))
    return out


def evaluate_manual_assessment(
    total_palos: int,
    nor: Optional[float] = None,
    block_totals: Optional[List[int]] = None,
    avg_spacing_mm: Optional[float] = None,
    avg_height_mm: Optional[float] = None,
    line_spacing_mm: Optional[float] = None,
    line_direction_angle_deg: Optional[float] = None,
    stroke_inclination_angle_deg: Optional[float] = None,
    margin_left_mm: Optional[float] = None,
    margin_right_mm: Optional[float] = None,
    margin_top_mm: Optional[float] = None,
    pressure_level: str = "",
    stroke_quality_level: str = "",
    organization_level: str = "",
    irregularities: Optional[List[str]] = None,
    order_pattern: str = "nao_informado",
    reasoning_level: str = "nao_informado",
    error_count: int = 0,
) -> Dict:
    block_totals = block_totals or []
    irregularities = irregularities or []

    nor_calc = float(nor) if nor is not None else (float(_compute_nor(block_totals)) if len(block_totals) > 1 else None)

    produtividade = _classify_productivity(int(total_palos))
    ritmo = _classify_rhythm(nor_calc)
    distancia = _classify_spacing_mm(avg_spacing_mm)
    tamanho_palos = _classify_stroke_size(avg_height_mm)
    dist_linhas = _classify_line_spacing_mm(line_spacing_mm)
    direcao_linhas = _classify_line_direction(line_direction_angle_deg)
    inclinacao_palos = _classify_stroke_inclination(stroke_inclination_angle_deg)
    margem_esquerda = _classify_margin_left(margin_left_mm)
    margem_direita = _classify_margin_right(margin_right_mm)
    margem_superior = _classify_margin_top(margin_top_mm)
    pressao = _classify_pressure(pressure_level)
    qualidade_tracado = _classify_stroke_quality(stroke_quality_level)
    organizacao = _classify_organization(organization_level)
    irregularidades_avaliadas = _evaluate_irregularities(irregularities)
    shape = _shape_classification(block_totals, nor_calc)
    qualidade = _quality_interpretation(produtividade["nivel"], nor_calc, shape)

    notas_nor = _nor_productivity_notes(int(total_palos), nor_calc)
    interpret_prod_order = _prod_order_interpretation(int(total_palos), order_pattern, reasoning_level)
    interpret_ritmo = _rhythm_text(ritmo["nivel"])

    observacoes = [interpret_prod_order, interpret_ritmo]
    observacoes.extend(notas_nor)
    tracos_personalidade = build_personality_traits(
        classes={
            "produtividade": produtividade,
            "ritmo": ritmo,
            "distancia": distancia,
            "tamanho_palos": tamanho_palos,
            "distancia_entre_linhas": dist_linhas,
            "direcao_linhas": direcao_linhas,
            "inclinacao_palos": inclinacao_palos,
            "margem_esquerda": margem_esquerda,
            "margem_direita": margem_direita,
            "margem_superior": margem_superior,
            "pressao": pressao,
            "qualidade_tracado": qualidade_tracado,
            "organizacao": organizacao,
            "qualidade_rendimento": qualidade,
            "forma_curva": shape,
        },
        order_pattern=order_pattern,
        reasoning_level=reasoning_level,
        total=int(total_palos),
    )

    return {
        "modo": "manual",
        "inputs": {
            "total_palos": int(total_palos),
            "nor_informado": nor,
            "blocos": block_totals,
            "avg_spacing_mm": avg_spacing_mm,
            "avg_height_mm": avg_height_mm,
            "line_spacing_mm": line_spacing_mm,
            "line_direction_angle_deg": line_direction_angle_deg,
            "stroke_inclination_angle_deg": stroke_inclination_angle_deg,
            "margin_left_mm": margin_left_mm,
            "margin_right_mm": margin_right_mm,
            "margin_top_mm": margin_top_mm,
            "pressure_level": pressure_level,
            "stroke_quality_level": stroke_quality_level,
            "organization_level": organization_level,
            "irregularities": irregularities,
            "order_pattern": order_pattern,
            "reasoning_level": reasoning_level,
            "error_count": int(error_count),
        },
        "metrics": {
            "total": int(total_palos),
            "linhas": None,
            "nor": round(nor_calc, 4) if nor_calc is not None else None,
            "blocos": block_totals,
            "espacamento_medio_mm": round(avg_spacing_mm, 4) if avg_spacing_mm is not None else None,
            "altura_media_palos_mm": round(avg_height_mm, 4) if avg_height_mm is not None else None,
            "distancia_entre_linhas_mm": round(line_spacing_mm, 4) if line_spacing_mm is not None else None,
            "angulo_direcao_linhas_graus": round(line_direction_angle_deg, 4) if line_direction_angle_deg is not None else None,
            "angulo_inclinacao_palos_graus": round(stroke_inclination_angle_deg, 4) if stroke_inclination_angle_deg is not None else None,
            "margem_esquerda_mm": round(margin_left_mm, 4) if margin_left_mm is not None else None,
            "margem_direita_mm": round(margin_right_mm, 4) if margin_right_mm is not None else None,
            "margem_superior_mm": round(margin_top_mm, 4) if margin_top_mm is not None else None,
            "erros": int(error_count),
        },
        "classificacoes": {
            "produtividade": produtividade,
            "ritmo": ritmo,
            "distancia": distancia,
            "tamanho_palos": tamanho_palos,
            "distancia_entre_linhas": dist_linhas,
            "direcao_linhas": direcao_linhas,
            "inclinacao_palos": inclinacao_palos,
            "margem_esquerda": margem_esquerda,
            "margem_direita": margem_direita,
            "margem_superior": margem_superior,
            "pressao": pressao,
            "qualidade_tracado": qualidade_tracado,
            "organizacao": organizacao,
            "qualidade_rendimento": qualidade,
            "forma_curva": shape,
        },
        "tracos_personalidade": tracos_personalidade,
        "irregularidades_avaliadas": irregularidades_avaliadas,
        "regras_aplicadas": sorted(
            {
                c.get("regra_id")
                for c in [
                    produtividade,
                    ritmo,
                    distancia,
                    tamanho_palos,
                    dist_linhas,
                    direcao_linhas,
                    inclinacao_palos,
                    margem_esquerda,
                    margem_direita,
                    margem_superior,
                    pressao,
                    qualidade_tracado,
                    organizacao,
                ]
                if isinstance(c, dict) and c.get("regra_id")
            }
            | {x.get("regra_id") for x in irregularidades_avaliadas if x.get("regra_id")}
        ),
        "observacoes": observacoes,
    }


def compute_metrics(
    line_counts: List[int],
    error_count: int = 0,
    config: Optional[ScoreConfig] = None,
    avg_spacing_mm: Optional[float] = None,
    avg_height_mm: Optional[float] = None,
    line_spacing_mm: Optional[float] = None,
    line_direction_angle_deg: Optional[float] = None,
    stroke_inclination_angle_deg: Optional[float] = None,
    margin_left_mm: Optional[float] = None,
    margin_right_mm: Optional[float] = None,
    margin_top_mm: Optional[float] = None,
    pressure_level: str = "",
    stroke_quality_level: str = "",
    organization_level: str = "",
    irregularities: Optional[List[str]] = None,
    order_pattern: str = "nao_informado",
    reasoning_level: str = "nao_informado",
) -> Dict:
    cfg = config or ScoreConfig()
    irregularities = irregularities or []

    if not line_counts:
        return {
            "total": 0,
            "linhas": 0,
            "media_por_linha": 0.0,
            "desvio_padrao": 0.0,
            "variabilidade_cv": 0.0,
            "velocidade_linha_seg": 0.0,
            "erros": int(error_count),
            "score_final": 0.0,
            "nor": 0.0,
            "blocos": [],
            "espacamento_medio_mm": avg_spacing_mm,
            "altura_media_palos_mm": avg_height_mm,
            "distancia_entre_linhas_mm": line_spacing_mm,
            "angulo_direcao_linhas_graus": line_direction_angle_deg,
            "classificacoes": {
                "produtividade": {"nivel": "Nao classificado", "faixa": "sem dados"},
                "ritmo": {"nivel": "Nao classificado", "faixa": "sem dados"},
                "distancia": {"nivel": "Nao classificado", "faixa": "sem dados"},
                "tamanho_palos": {"nivel": "Nao classificado", "faixa": "sem dados"},
                "distancia_entre_linhas": {"nivel": "Nao classificado", "faixa": "sem dados"},
                "direcao_linhas": {"nivel": "Nao classificado", "faixa": "sem dados"},
                "inclinacao_palos": {"nivel": "Nao classificado", "faixa": "sem dados"},
                "margem_esquerda": {"nivel": "Nao classificado", "faixa": "sem dados"},
                "margem_direita": {"nivel": "Nao classificado", "faixa": "sem dados"},
                "margem_superior": {"nivel": "Nao classificado", "faixa": "sem dados"},
                "pressao": {"nivel": "Nao classificado", "faixa": "sem dados"},
                "qualidade_tracado": {"nivel": "Nao classificado", "faixa": "sem dados"},
                "organizacao": {"nivel": "Nao classificado", "faixa": "sem dados"},
                "qualidade_rendimento": "Nao classificado",
                "forma_curva": "Indeterminado",
            },
            "tracos_personalidade": [],
            "irregularidades_avaliadas": [],
            "regras_aplicadas": [],
            "observacoes": [],
        }

    total = int(sum(line_counts))
    linhas = int(len(line_counts))
    avg = float(mean(line_counts))
    std = float(pstdev(line_counts)) if linhas > 1 else 0.0
    cv = float((std / avg) if avg > 0 else 0.0)

    velocidade = float(cfg.time_per_block_seconds / max(cfg.block_size_lines, 1))
    score = total - (error_count * cfg.error_penalty) - (cv * total * cfg.variability_penalty_factor)

    blocks = _split_blocks(line_counts, cfg.block_size_lines)
    nor = _compute_nor(blocks)

    produtividade = _classify_productivity(total)
    ritmo = _classify_rhythm(nor)
    distancia = _classify_spacing_mm(avg_spacing_mm)
    tamanho_palos = _classify_stroke_size(avg_height_mm)
    dist_linhas = _classify_line_spacing_mm(line_spacing_mm)
    direcao_linhas = _classify_line_direction(line_direction_angle_deg)
    inclinacao_palos = _classify_stroke_inclination(stroke_inclination_angle_deg)
    margem_esquerda = _classify_margin_left(margin_left_mm)
    margem_direita = _classify_margin_right(margin_right_mm)
    margem_superior = _classify_margin_top(margin_top_mm)
    pressao = _classify_pressure(pressure_level)
    qualidade_tracado = _classify_stroke_quality(stroke_quality_level)
    organizacao = _classify_organization(organization_level)
    irregularidades_avaliadas = _evaluate_irregularities(irregularities)
    shape = _shape_classification(blocks, nor)
    qualidade = _quality_interpretation(produtividade["nivel"], nor, shape)

    observacoes = _nor_productivity_notes(total, nor)
    classes = {
        "produtividade": produtividade,
        "ritmo": ritmo,
        "distancia": distancia,
        "tamanho_palos": tamanho_palos,
        "distancia_entre_linhas": dist_linhas,
        "direcao_linhas": direcao_linhas,
        "inclinacao_palos": inclinacao_palos,
        "margem_esquerda": margem_esquerda,
        "margem_direita": margem_direita,
        "margem_superior": margem_superior,
        "pressao": pressao,
        "qualidade_tracado": qualidade_tracado,
        "organizacao": organizacao,
        "qualidade_rendimento": qualidade,
        "forma_curva": shape,
    }
    tracos_personalidade = build_personality_traits(
        classes=classes,
        total=total,
        order_pattern=order_pattern,
        reasoning_level=reasoning_level,
    )
    regras_aplicadas = sorted(
        {
            c.get("regra_id")
            for c in [
                produtividade,
                ritmo,
                distancia,
                tamanho_palos,
                dist_linhas,
                direcao_linhas,
                inclinacao_palos,
                margem_esquerda,
                margem_direita,
                margem_superior,
                pressao,
                qualidade_tracado,
                organizacao,
            ]
            if isinstance(c, dict) and c.get("regra_id")
        }
        | {x.get("regra_id") for x in irregularidades_avaliadas if x.get("regra_id")}
    )

    return {
        "total": total,
        "linhas": linhas,
        "media_por_linha": round(avg, 4),
        "desvio_padrao": round(std, 4),
        "variabilidade_cv": round(cv, 4),
        "velocidade_linha_seg": round(velocidade, 4),
        "erros": int(error_count),
        "score_final": round(float(score), 4),
        "nor": round(nor, 4),
        "blocos": blocks,
        "espacamento_medio_mm": round(avg_spacing_mm, 4) if avg_spacing_mm is not None else None,
        "altura_media_palos_mm": round(avg_height_mm, 4) if avg_height_mm is not None else None,
        "distancia_entre_linhas_mm": round(line_spacing_mm, 4) if line_spacing_mm is not None else None,
        "angulo_direcao_linhas_graus": round(line_direction_angle_deg, 4) if line_direction_angle_deg is not None else None,
        "angulo_inclinacao_palos_graus": round(stroke_inclination_angle_deg, 4) if stroke_inclination_angle_deg is not None else None,
        "margem_esquerda_mm": round(margin_left_mm, 4) if margin_left_mm is not None else None,
        "margem_direita_mm": round(margin_right_mm, 4) if margin_right_mm is not None else None,
        "margem_superior_mm": round(margin_top_mm, 4) if margin_top_mm is not None else None,
        "classificacoes": classes,
        "tracos_personalidade": tracos_personalidade,
        "irregularidades_avaliadas": irregularidades_avaliadas,
        "regras_aplicadas": regras_aplicadas,
        "observacoes": observacoes,
    }
