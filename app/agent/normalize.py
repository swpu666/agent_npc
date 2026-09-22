"""文本归一化、注入检测、指代检测与情境条件提取。

该模块只做纯文本处理，不依赖网络与知识库文件，便于单元测试。
"""

from __future__ import annotations

import re
import unicodedata

from app.agent.synonyms import (
    GAME_ALIASES,
    HERO_HINTS,
    INJECTION_PATTERNS,
    PHASE_HINTS,
    POSITION_HINTS,
    SYNONYM_LOOKUP,
    TOPIC_TERMS,
)

_WS_RE = re.compile(r"\s+")
# 归一化时统一丢弃的标点（中英文），保留字母数字与汉字
_PUNCT_RE = re.compile(r"[\u3000-\u303f\uff00-\uffef!-/:-@\[-`{-~]+")

# 指代标记：出现时说明该问句依赖上文
ANAPHORA_MARKERS = [
    "这个", "那个", "这些", "那些", "这俩", "它", "该", "此", "上面", "刚才", "之前说的",
    "继续说", "还有呢", "然后呢", "接着", "同样的", "一样",
]

# 说明：早期版本这里有一组 ELLIPTICAL_PATTERNS，用来识别"那这个呢"这类省略式追问。
# 现在这条链路改成了「上一轮状态 + 指代表达」两件事分别在 intent / retrieval 里判断
# （见 intent.classify 的 prev_route_state 继承、retrieval 的 detect_anaphora），
# 独立的句式表已经不再被任何地方使用，于是删除——留着会让人误以为它还在生效。

# 主题词的扩展集合：把主题词的等价俗称也算作主题词
_TOPIC_TERM_POOL: set[str] = set(TOPIC_TERMS)
for _term in list(TOPIC_TERMS):
    _TOPIC_TERM_POOL.update(SYNONYM_LOOKUP.get(_term, set()))
_TOPIC_TERM_POOL.update(GAME_ALIASES.keys())


def normalize(text: str) -> str:
    """全角转半角、统一小写、压缩空白、去除标点噪声。"""
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = text.lower()
    text = _PUNCT_RE.sub("", text)
    text = _WS_RE.sub("", text)
    return text.strip()


def has_topic_term(text: str) -> bool:
    """是否包含强游戏主题词（含常见俗称）。"""
    t = normalize(text)
    if not t:
        return False
    return any(term and term in t for term in _TOPIC_TERM_POOL)


def detect_injection(text: str) -> list[str]:
    """检测"尝试覆盖系统约束"的指令模式。

    只做标记，不做内容清洗：原始文本仍会作为数据传给模型，
    由系统提示词声明其为数据而非指令（对应设计文档 §2.4）。
    """
    t = normalize(text)
    if not t:
        return []
    hits: list[str] = []
    for pattern in INJECTION_PATTERNS:
        if re.search(pattern, t):
            hits.append(pattern)
    return hits


def detect_anaphora(text: str) -> bool:
    """是否出现指代/承接表达，需要借助历史补全检索意图。"""
    t = normalize(text)
    return any(m in t for m in ANAPHORA_MARKERS)


def extract_conditions(text: str) -> dict[str, list[str]]:
    """提取情境条件：位置 / 阶段 / 英雄。

    用于判断情境类问题是否缺关键条件（缺则反问补充），
    依赖历史时会由调用方把历史文本一起传入。
    """
    t = normalize(text)
    return {
        "position": sorted({w for w in POSITION_HINTS if normalize(w) in t}),
        "phase": sorted({w for w in PHASE_HINTS if normalize(w) in t}),
        "hero": sorted({w for w in HERO_HINTS if normalize(w) in t}),
    }


def has_any_condition(text: str) -> bool:
    cond = extract_conditions(text)
    return bool(cond["position"] or cond["phase"] or cond["hero"])


def truncate(text: str, limit: int) -> str:
    if text is None:
        return ""
    return text if len(text) <= limit else text[:limit]


_URL_RE = re.compile(r"https?://[^\s，。、）)】\"'<>]+")


def extract_urls(text: str) -> list[str]:
    return _URL_RE.findall(text or "")


# 回答自述"没有收录"的措辞。
_NO_COVERAGE_RE = re.compile(
    r"(没有收录|未收录|没有覆盖|没有相关资料|没法给你|没法回答|无法回答|答不了|答不上|帮不了你|帮不上)"
)


def has_meaningful_text(raw_text: str) -> bool:
    """消息里是否有实质内容（汉字或字母数字）。

    纯符号输入（"。。。。"、"？？？"）在归一化后会变成空串，此时所有规则都没有信号，
    只能靠上下文继承硬猜——实测会把一串句号当成上一轮问题的延续去作答。
    这类输入应被明确拦下，而不是让模型猜。
    """
    return bool(re.search(r"[\u4e00-\u9fff0-9a-zA-Z]", raw_text or ""))
_SENTENCE_END_RE = re.compile(r"[。！？!?\n]")
_FIRST_SENTENCE_LIMIT = 60


def first_sentence(text: str, limit: int = _FIRST_SENTENCE_LIMIT) -> str:
    head = (text or "").lstrip()
    match = _SENTENCE_END_RE.search(head)
    return head[: match.start()] if match else head[:limit]


def is_no_coverage_leading(answer: str) -> bool:
    """回答是否**在第一句话里**就声明"这个我没收录"。

    只认第一句，不能放宽到"开头若干字符"：实测有回答第一句就在正常作答
    （"前期主要是清理野区资源……"），第三句才补一句"更细的我的知识库里没有收录"。
    按"前 80 字"匹配会把这种正常回答误判成未覆盖，把本来正确的来源清空——
    这正是关键词判断的典型失效方式，所以判定边界必须收紧到"开门见山就是拒答"。
    """
    return bool(_NO_COVERAGE_RE.search(first_sentence(answer)))
