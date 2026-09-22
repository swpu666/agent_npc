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

# 纯追问句式（省略主语，必须依赖上文才能理解）
ELLIPTICAL_PATTERNS = [
    r"^那(个|这个|它)?呢",
    r"^那(这个|那个|它|么)?(怎么|如何|为什么|多久|多少|是什?么|有什?么)",
    r"(那|这)(这个|那个|些)?呢",
    r"^还有(呢|别的|其他|吗|没有|什么)",
    r"^然后呢|^接着呢|^继续说|^再说说|^详细(说|讲)",
    r"^为什么呢?$|^怎么说呢?$|^为什么呀?$",
]

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


def is_elliptical(text: str) -> bool:
    """是否是省略式追问（"那这个呢"），这类问句单独无法判定意图。"""
    t = normalize(text)
    if not t or len(t) > 16:
        return False
    return any(re.search(p, t) for p in ELLIPTICAL_PATTERNS)


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
