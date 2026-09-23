"""文本归一化、注入检测、指代检测与情境条件提取。

该模块只做纯文本处理，不依赖网络与知识库文件，便于单元测试。
"""

from __future__ import annotations

import re
import unicodedata

from app.agent.synonyms import (
    CONVERSATION_RECALL_PATTERNS,
    GAME_ALIASES,
    HERO_HINTS,
    INJECTION_PATTERNS,
    PHASE_HINTS,
    POSITION_HINTS,
    QUESTION_PATTERNS,
    SYNONYM_GROUPS,
    SYNONYM_LOOKUP,
    TOPIC_TERMS,
    canonical_hero,
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


_RECALL_RES = [re.compile(p) for p in CONVERSATION_RECALL_PATTERNS]

# 回顾类问句的长度上限：它是一句元提问（"我前面问了什么"），不会很长。
# 放宽这个界限只会把"我前面问的暴君刷新时间是多少"这类真问题卷进来。
_RECALL_MAX_CHARS = 20


def is_conversation_recall(text: str) -> bool:
    """是否在问**对话本身**（"我前面问了什么""刚才说到哪了"）。

    这类问题属于对话元信息，不属于游戏知识：把它当知识问题去检索，
    只会白白检不到、再回一句"知识库里没收录"——而答案就在 short-term
    history 里。判定必须在规则层完成，且不消耗模型调用。

    三条护栏（缺一不可）：
    1. 是短句——元提问不会长；
    2. 不含游戏主题词——「我刚才说的连招是什么」问的是连招，不是回顾；
    3. 不含情境条件（位置/阶段/英雄）——同上。
    """
    t = normalize(text)
    if not t or len(t) > _RECALL_MAX_CHARS:
        return False
    if has_topic_term(t):
        return False
    if any(extract_conditions(t).values()):
        return False
    return any(r.search(t) for r in _RECALL_RES)


# "那 XX 呢 / XX 呢"：把上一轮的话题换个对象再问一遍。
# 这类句子既没有疑问词也没有主题词（"那赵铁柱呢"），规则层完全没有信号，
# 但它明确是"同一类问题换个对象"，应当继承上一轮的意图与路由状态。
_TOPIC_FOLLOWUP_RE = re.compile(r"^那?.{1,8}呢$")


def is_topic_followup(text: str) -> bool:
    """是否是"换个对象再问一遍"式的追问。

    刻意排除带疑问词的句子（"暴君什么时候打呢"）：那种句子有明确的问题结构，
    应当独立判定，不能因为结尾带个"呢"就当成追问。
    """
    t = normalize(text)
    if not t or len(t) > 12 or not t.endswith("呢"):
        return False
    if any(word in t for word in QUESTION_PATTERNS):
        return False
    return bool(_TOPIC_FOLLOWUP_RE.match(t))


def extract_conditions(text: str) -> dict[str, list[str]]:
    """提取情境条件：位置 / 阶段 / 英雄。

    用于判断情境类问题是否缺关键条件（缺则反问补充），
    依赖历史时会由调用方把历史文本一起传入。
    """
    t = normalize(text)
    return {
        "position": sorted({w for w in POSITION_HINTS if normalize(w) in t}),
        "phase": sorted({w for w in PHASE_HINTS if normalize(w) in t}),
        # 错别字还原成规范写法后再上报：下游（提示词、追问复述）应当说「安琪拉」，
        # 而不是把玩家的错写「安其拉」再念回去。
        "hero": sorted({canonical_hero(w) for w in HERO_HINTS if normalize(w) in t}),
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
    """消息里是否有任何字符（用于区分"完全空"与"有内容"）。"""
    return bool(re.search(r"[\u4e00-\u9fff0-9a-zA-Z]", raw_text or ""))


# 已知的游戏拉丁术语（buff / adc / gank / combo …）：出现这些才算"有内容"，
# 否则一串字母和纯数字一样无法理解。从同义词表里收集，避免另维护一份。
_LATIN_HINTS: set[str] = {
    t for group in SYNONYM_GROUPS for t in group if t.isascii() and t.isalpha() and len(t) >= 2
}
_LATIN_HINTS |= {k for k in GAME_ALIASES if k.isascii()}
_CJK_RE = re.compile(r"[\u4e00-\u9fff]")


def is_meaningless_input(raw_text: str) -> bool:
    """输入是否没有任何可理解的内容（纯数字、纯符号、未知字母串）。

    这类输入**既不该当成"越界请求"**（越界是"要求了未接入的能力"），
    也不该交给模型兜底分类。实测「123456」会因此花掉 1.3 秒、被判成
    `out_of_scope`，然后回一句冷冰冰的"这个请求超出我的能力范围了"——
    判断是错的，语气也不像人。正确做法是：承认"没看懂"，并拟人化地引导回可答范围。
    """
    raw = raw_text or ""
    if _CJK_RE.search(raw):
        return False
    t = normalize(raw)
    if not t:
        return True
    return not any(hint in t for hint in _LATIN_HINTS)
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
