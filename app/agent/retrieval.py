"""知识检索：查询改写 → 同义词扩展 → BM25 + 字段加权打分 → 版本折扣 → 阈值判定 → Top-K。

打分公式（对应设计文档 §3.4）：

    score = 2.0 * BM25_norm(content)
          + Σ_t weight(t) * [3.0·title + 2.5·keywords + 2.0·topic + 1.0·entity]
    final = score * version_factor

    weight(t) = 1.0   直接命中查询词
                0.8   同义词 / 别称 / 简称扩展命中

BM25 分量按候选集内的最大值归一化到 [0, 2]，这样"是否命中"主要由人工提炼的
标题/关键词决定，而正文长文本只提供兜底召回；阈值 RETRIEVAL_MIN_SCORE 因此
具有跨查询的稳定含义：没有任何字段词命中时，总分最多 2.0，必然低于阈值。
"""

from __future__ import annotations

import json
import math
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from app.agent.normalize import normalize
from app.agent.synonyms import ALIAS_TERMS, SYNONYM_LOOKUP

# 人工补充的通用主题词（不放在 synonyms 里是为了强调它们只用于检索，不用于意图判定）。
# 刻意排除「作用 / 机制 / 规则」这类泛化词：它们几乎出现在所有条目的标题里，
# 会给标题 3.0 的权重带来大量噪声，而具体实体词已经足够定位。
EXTRA_TERMS = {
    "职责", "分工", "顺序", "刷新", "收益", "打法", "思路", "技巧",
    "反野", "抓人", "支援", "发育", "血量", "蓝量",
}

# 允许作为词条的单字（人工确认过不会造成大面积误命中）
SINGLE_CHAR_TERMS = {"塔", "赢"}

FIELD_WEIGHTS = {"title": 3.0, "keywords": 2.5, "topic": 2.0, "entity": 1.0}
BM25_SCALE = 2.0
BM25_K1 = 1.5
BM25_B = 0.75
SYNONYM_WEIGHT = 0.8
TOPIC_BOOST = 1.5
VERSION_MISMATCH_FACTOR = 0.6

_S_VERSION_RANGE = re.compile(r"s(\d+)\s*[-–~至到]\s*s(\d+)")
_S_VERSION_PLUS = re.compile(r"s(\d+)\s*\+")


@dataclass
class RetrievedDoc:
    id: str
    topic: str
    subtopic: str
    title: str
    content: str
    version: str
    confidence: str
    source_title: str
    source_url: str
    collected_at: str
    # 来源强度：direct = 该页直接写明本条内容；topic = 覆盖该主题但非逐句对应
    source_support: str = "direct"
    source_support_note: str = ""
    score: float = 0.0
    base_score: float = 0.0
    version_factor: float = 1.0
    match_reason: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "topic": self.topic,
            "subtopic": self.subtopic,
            "title": self.title,
            "version": self.version,
            "confidence": self.confidence,
            "score": round(self.score, 3),
            "match_reason": self.match_reason,
        }


# 高频虚词：由它们组成的二元组（"才能""怎么""可以"）没有检索价值，
# 不过滤会让 BM25 分量在"实词完全没命中"时也给出分数，干扰阈值判定与排序。
_STOP_CHARS = set("怎么才能的了吗呢是在有和与就都也还把被给对我你他它们这那个要会可及为从到而并但很太多少几个上下中前后")


def _bigrams(text: str) -> list[str]:
    t = normalize(text)
    if len(t) < 2:
        return [t] if t else []
    grams = [t[i:i + 2] for i in range(len(t) - 1)]
    return [g for g in grams if not all(ch in _STOP_CHARS for ch in g)]


def _build_term_dict(entries: list[dict]) -> set[str]:
    terms: set[str] = set()
    for e in entries:
        for item in e.get("keywords", []) or []:
            terms.add(normalize(item))
        for item in e.get("entities", []) or []:
            terms.add(normalize(item))
        terms.add(normalize(e.get("topic", "")))
        terms.add(normalize(e.get("subtopic", "")))
    terms |= {normalize(t) for t in EXTRA_TERMS}
    terms |= {normalize(t) for t in ALIAS_TERMS}
    # SINGLE_CHAR_TERMS 是显式白名单：它既是"允许单字词条"的开关，也是这些词的来源。
    # 否则像「赢」这种只出现在「怎么赢」「赢的条件」里的单字，会因为从未被独立登记而永远召不回。
    terms |= {normalize(t) for t in SINGLE_CHAR_TERMS}
    return {t for t in terms if t and (len(t) >= 2 or t in SINGLE_CHAR_TERMS)}


def extract_terms(text: str, term_dict: set[str], dedupe: bool = True) -> list[str]:
    """基于词表的命中抽取：取出所有出现在文本中的词条。

    相比"最长匹配分词"，这里用"包含判定"：知识库关键词本身就是完整的词
    （如「怎么赢」「防御塔」），包含判定能保证单字词（如「赢」）不会因为被长词
    吞掉而丢失召回。

    dedupe=True（查询侧）：消掉相互包含的冗余，避免同一处文本被长短两个词重复计分。
    dedupe=False（文档侧）：保留全部命中，索引集合是"是否包含"的查询表，
      多留几个词只会提高召回，不会有重复计分问题。
    """
    t = normalize(text)
    if not t:
        return []
    matched = sorted((term for term in term_dict if term in t), key=len, reverse=True)
    if not dedupe:
        return matched
    kept: list[str] = []
    for term in matched:
        if not any(term in longer for longer in kept):
            kept.append(term)
    return kept


def _expand_with_synonyms(direct_terms: list[str], term_dict: set[str]) -> dict[str, float]:
    weighted: dict[str, float] = {}
    for term in direct_terms:
        weighted[term] = 1.0
    for term in direct_terms:
        for syn in SYNONYM_LOOKUP.get(term, set()):
            s = normalize(syn)
            if s and s in term_dict and s not in weighted:
                weighted[s] = SYNONYM_WEIGHT
    return weighted


def _light_normalize(text: str) -> str:
    """版本号专用归一化：只做全角转半角与小写，保留 `+` `-` 等区间符号。

    不能用通用的 normalize()：它会连 `+`、`-` 一起当标点抹掉，
    导致 "S36+" 变成 "s36"，版本区间判定随之失效。
    """
    return unicodedata.normalize("NFKC", text or "").strip().lower()


def _version_factor(version: str, current_version: str) -> tuple[float, str | None]:
    v = _light_normalize(version)
    cur_match = re.search(r"s?(\d+)", _light_normalize(current_version))
    current = int(cur_match.group(1)) if cur_match else None

    if v.startswith("全版本"):
        return 1.0, None
    if current is None:
        return 1.0, None

    m = _S_VERSION_RANGE.search(v)
    if m:
        lo, hi = int(m.group(1)), int(m.group(2))
        if lo <= current <= hi:
            return 1.0, None
        return VERSION_MISMATCH_FACTOR, f"该条目适用 {version}，当前版本 {current_version} 可能已调整"

    m = _S_VERSION_PLUS.search(v)
    if m:
        lo = int(m.group(1))
        if current >= lo:
            return 1.0, None
        return VERSION_MISMATCH_FACTOR, f"该条目适用于 {version} 之后，当前版本 {current_version} 可能尚未生效"

    # "版本相关""数值随版本调整"等描述性标注：不做折扣，交由回答层提示以客户端为准
    return 1.0, None


class KnowledgeBase:
    def __init__(self, entries: list[dict], current_version: str = "S47"):
        self.entries = entries
        self.current_version = current_version
        self.term_dict = _build_term_dict(entries)
        self.docs: list[RetrievedDoc] = [self._to_doc(e) for e in entries]
        self.field_terms: list[dict[str, set[str]]] = []
        self.bigram_lists: list[list[str]] = []
        self.df: Counter = Counter()
        self.avg_len = 1.0
        self._build_index()

    # ------------------------------------------------------------------ 索引
    @staticmethod
    def _to_doc(e: dict) -> RetrievedDoc:
        src = e.get("source", {}) or {}
        return RetrievedDoc(
            id=e.get("id", ""),
            topic=e.get("topic", ""),
            subtopic=e.get("subtopic", ""),
            title=e.get("title", ""),
            content=e.get("content", ""),
            version=e.get("version", ""),
            confidence=e.get("confidence", "high"),
            source_title=src.get("title", ""),
            source_url=src.get("url", ""),
            collected_at=src.get("collected_at", ""),
            source_support=src.get("support", "direct"),
            source_support_note=src.get("support_note", ""),
        )

    def _build_index(self) -> None:
        lengths = []
        for e in self.entries:
            title_text = e.get("title", "")
            kw_text = " ".join(e.get("keywords", []) or [])
            topic_text = f"{e.get('topic', '')} {e.get('subtopic', '')}"
            entity_text = " ".join(e.get("entities", []) or [])
            self.field_terms.append({
                "title": set(extract_terms(title_text, self.term_dict, dedupe=False)),
                "keywords": set(extract_terms(kw_text, self.term_dict, dedupe=False)),
                "topic": set(extract_terms(topic_text, self.term_dict, dedupe=False)),
                "entity": set(extract_terms(entity_text, self.term_dict, dedupe=False)),
            })
            grams = _bigrams(e.get("content", ""))
            self.bigram_lists.append(grams)
            lengths.append(len(grams) or 1)
            for g in set(grams):
                self.df[g] += 1
        self.avg_len = sum(lengths) / max(1, len(lengths))

    # ------------------------------------------------------------------ 检索
    def _bm25_raw(self, query_terms_bigrams: list[str]) -> list[float]:
        n_docs = max(1, len(self.docs))
        scores = []
        for grams in self.bigram_lists:
            tf = Counter(grams)
            length = len(grams) or 1
            s = 0.0
            for g in query_terms_bigrams:
                f = tf.get(g, 0)
                if not f:
                    continue
                idf = math.log(1 + (n_docs - self.df.get(g, 0) + 0.5) / (self.df.get(g, 0) + 0.5))
                denom = f + BM25_K1 * (1 - BM25_B + BM25_B * length / self.avg_len)
                s += idf * f * (BM25_K1 + 1) / denom
            scores.append(s)
        return scores

    def search(
        self,
        query: str,
        top_k: int = 4,
        min_score: float = 3.0,
        boost_topic: str | None = None,
        explain: bool = True,
    ) -> tuple[list[RetrievedDoc], bool]:
        """返回 (命中的知识列表, 是否达到阈值)。"""
        direct_terms = extract_terms(query, self.term_dict)
        weighted_terms = _expand_with_synonyms(direct_terms, self.term_dict)

        bm25_raw = self._bm25_raw(_bigrams(query))
        bm25_max = max(bm25_raw) if bm25_raw else 0.0

        results: list[RetrievedDoc] = []
        for idx, doc in enumerate(self.docs):
            field_hits: dict[str, list[str]] = {}
            field_score = 0.0
            for field, weight in FIELD_WEIGHTS.items():
                hits = [t for t in weighted_terms if t in self.field_terms[idx][field]]
                if hits:
                    field_hits[field] = hits
                    field_score += weight * sum(weighted_terms[t] for t in hits)

            bm25_component = BM25_SCALE * (bm25_raw[idx] / bm25_max) if bm25_max > 0 else 0.0
            base = field_score + bm25_component
            if base <= 0.05:
                continue

            factor, version_note = _version_factor(doc.version, self.current_version)
            score = base * factor
            reason: list[str] = []
            if explain:
                for field in ("title", "keywords", "topic", "entity"):
                    if field in field_hits:
                        reason.append(f"{field}:{','.join(sorted(field_hits[field]))}")
                if bm25_component > 0:
                    reason.append(f"content_bm25:{bm25_component:.2f}")

            boosted = False
            if boost_topic and doc.topic == boost_topic:
                score *= TOPIC_BOOST
                boosted = True
            if explain and boosted:
                reason.append(f"topic_boost:承接上一轮话题「{doc.topic}」")
            if explain and version_note:
                reason.append(f"version:{version_note}")

            results.append(
                RetrievedDoc(
                    **{**doc.__dict__, "score": score, "base_score": base,
                       "version_factor": factor, "match_reason": reason}
                )
            )

        results.sort(key=lambda d: -d.score)
        if not results:
            return [], False

        top_score = results[0].score
        # 边际相关性裁剪：明显弱于首条的结果不注入上下文，避免诱导模型过度发挥
        cutoff = top_score * 0.4
        kept = [d for d in results[:top_k] if d.score >= cutoff]
        hit = top_score >= min_score
        return (kept if hit else []), hit

    # ------------------------------------------------------------------ 辅助
    def stats(self) -> dict:
        topics = Counter(d.topic for d in self.docs)
        return {
            "entry_count": len(self.docs),
            "topic_distribution": dict(topics),
            "term_count": len(self.term_dict),
            "current_version": self.current_version,
        }


@lru_cache(maxsize=4)
def load_knowledge_base(path: str, current_version: str) -> KnowledgeBase:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return KnowledgeBase(data.get("entries", []), current_version=current_version)


def previous_user_message(history: list[dict] | None) -> str | None:
    """取历史中最近一条用户消息。"""
    if not history:
        return None
    for m in reversed(history):
        if isinstance(m, dict) and m.get("role") == "user" and m.get("content"):
            return str(m["content"])
    return None


def augment_query(text: str, prev_user: str | None) -> str:
    """用上一轮问题补全当前问句的检索意图。

    调用方只要发现存在上一轮用户消息，就会一律调用它把上下文拼进查询一起检索；
    拼接后仍检不到才算未命中——是否补上下文不再取决于当前句"是否含主题词"。

    早期版本用 `has_topic_term` 做判据（含主题词则判定不必补上文），在
    「那我被对面反野了怎么办」上失效：`反野` 是主题词（于是不补历史），
    但知识库里并没有以它命名的条目（于是也检不到），两个判断互相抵消。
    """
    if not prev_user:
        return text
    return f"{prev_user} {text}"
