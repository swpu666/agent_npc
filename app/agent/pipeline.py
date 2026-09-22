"""Agent 编排链路：归一化 → 意图路由 → 按需检索 → 生成 → 后处理。

对应设计文档 §2.2 的六步流程。耗时按 意图 / 检索 / 生成 三段分别测量，
总耗时 = 三者之和 + 极小的编排开销。
"""

from __future__ import annotations

import itertools
import time
from dataclasses import dataclass, field
from datetime import datetime

from app.agent import prompt as prompt_mod
from app.agent import templates
from app.agent.intent import (
    INTENT_CHITCHAT,
    INTENT_KNOWLEDGE,
    INTENT_OUT_OF_SCOPE,
    INTENT_SITUATIONAL,
    ROUTE_KB_MISS,
    ROUTE_NEED_MORE_INFO,
    IntentResult,
    classify,
)
from app.agent.llm import LLMClient, LLMError
from app.agent.normalize import detect_anaphora, extract_urls, is_elliptical
from app.agent.retrieval import (
    KnowledgeBase,
    RetrievedDoc,
    augment_query,
    load_knowledge_base,
    previous_user_message,
)
from app.config import KNOWLEDGE_FILE, Settings, get_settings

_request_counter = itertools.count(1)


def next_request_id() -> str:
    return f"r-{datetime.now().strftime('%Y%m%d')}-{next(_request_counter):04d}"


@dataclass
class AgentResult:
    answer: str = ""
    intent: str = INTENT_KNOWLEDGE
    route_state: str | None = None
    citations: list[dict] = field(default_factory=list)
    retrieved: list[RetrievedDoc] = field(default_factory=list)
    timings: dict = field(default_factory=dict)
    model: str = ""
    mock: bool = False
    retryable: bool = False
    guard: dict = field(default_factory=dict)
    intent_detail: dict = field(default_factory=dict)
    request_id: str = ""
    error: dict | None = None
    debug: dict | None = None


def _citation(doc: RetrievedDoc) -> dict:
    return {
        "id": doc.id,
        "title": doc.title,
        "url": doc.source_url,
        "collected_at": doc.collected_at,
        "version": doc.version,
        "score": round(doc.score, 3),
        "topic": doc.topic,
    }


def _sanitize_answer(answer: str, allowed_urls: set[str]) -> tuple[str, list[str]]:
    """剔除回答中不属于本次检索来源的链接，避免模型自行编造来源。"""
    removed: list[str] = []
    for url in extract_urls(answer):
        if url.rstrip(".,;，。") not in allowed_urls:
            removed.append(url)
            answer = answer.replace(url, "")
    if removed:
        answer = "\n".join(line.rstrip() for line in answer.splitlines()).strip()
    return answer, removed


class Agent:
    def __init__(self, settings: Settings | None = None, llm: LLMClient | None = None, kb: KnowledgeBase | None = None):
        self.settings = settings or get_settings()
        self.llm = llm or LLMClient(self.settings)
        self.kb = kb or load_knowledge_base(str(KNOWLEDGE_FILE), self.settings.game_version)

    # ------------------------------------------------------------------ 主链路
    def answer(self, message: str, history: list[dict] | None = None) -> AgentResult:
        result = AgentResult(request_id=next_request_id(), model=self.llm.model, mock=self.settings.mock_llm)
        t_start = time.perf_counter()

        # [1] 意图路由（本地规则，通常 < 1ms）
        t0 = time.perf_counter()
        prev_intent = None
        if history and is_elliptical(message):
            prev_user = self._last_user_message(history)
            if prev_user:
                prev_intent = classify(prev_user, history=None).intent
        intent_result: IntentResult = classify(
            message,
            history=history,
            llm_classifier=lambda text, hist: self.llm.classify_intent(text),
            inherit_intent=prev_intent,
        )
        intent_ms = round((time.perf_counter() - t0) * 1000, 2)

        result.intent = intent_result.intent
        result.route_state = intent_result.route_state
        result.intent_detail = intent_result.to_debug()
        result.guard = {
            "injection_suspected": intent_result.is_injection_suspected,
            "matched_patterns": intent_result.injection,
            "sanitized_urls": [],
            "notes": [],
        }

        # [2] 按意图决定是否检索 / 是否调用模型
        t1 = time.perf_counter()
        docs: list[RetrievedDoc] = []
        near_titles: list[str] = []
        used_history = False
        if intent_result.intent in (INTENT_KNOWLEDGE, INTENT_SITUATIONAL) and intent_result.route_state != ROUTE_NEED_MORE_INFO:
            docs, near_titles, used_history = self._retrieve(message, history, intent_result)
        retrieval_ms = round((time.perf_counter() - t1) * 1000, 2)

        if intent_result.intent == INTENT_KNOWLEDGE and not docs:
            result.route_state = ROUTE_KB_MISS

        # [3] 生成：能确定回答的直接走模板，不需要模型
        llm_ms = 0
        if intent_result.intent == INTENT_OUT_OF_SCOPE:
            result.answer = templates.out_of_scope_answer(intent_result.oos_reason)
        elif result.route_state == ROUTE_KB_MISS:
            result.answer = templates.kb_miss_answer(near_titles)
        elif result.route_state == ROUTE_NEED_MORE_INFO:
            result.answer = templates.need_more_info_answer()
        else:
            messages = prompt_mod.build_messages(
                intent=intent_result.intent,
                route_state=result.route_state,
                user_text=message,
                history=history,
                docs=docs,
                history_max_turns=self.settings.history_max_turns,
                history_max_chars=self.settings.history_max_chars,
                injection_suspected=intent_result.is_injection_suspected,
            )
            llm_result = self.llm.chat(messages)
            result.answer = llm_result.text
            llm_ms = llm_result.latency_ms

        # [4] 后处理：引用校验 + 来源链接清洗
        allowed = {d.source_url for d in docs}
        result.answer, removed = _sanitize_answer(result.answer, allowed)
        if removed:
            result.guard["sanitized_urls"] = removed
            result.guard["notes"].append("回答中出现了非本次检索来源的链接，已剔除")

        result.citations = [_citation(d) for d in docs]
        result.retrieved = docs

        version_mismatch = [d for d in docs if d.version_factor < 1]
        if version_mismatch:
            result.guard["notes"].append("存在版本适用性折扣的知识条目")
            result.answer += "\n\n（说明：部分内容涉及版本调整，具体数值请以客户端内说明为准。）"

        result.timings = {
            "total_ms": int((time.perf_counter() - t_start) * 1000),
            "intent_ms": intent_ms,
            "retrieval_ms": retrieval_ms,
            "llm_ms": llm_ms,
        }
        if self.settings.mock_llm:
            result.guard["notes"].append("MOCK 模式：未调用真实模型，结果不代表真实回答质量")

        result.debug = {
            "intent": result.intent_detail,
            "retrieval_used_history": used_history,
            "retrieved": [d.to_dict() for d in docs] if docs else [],
            "near_miss_titles": near_titles,
        }
        return result

    # ------------------------------------------------------------------ 内部
    @staticmethod
    def _last_user_message(history: list[dict] | None) -> str | None:
        return previous_user_message(history)

    def _retrieve(
        self,
        message: str,
        history: list[dict] | None,
        intent_result: IntentResult,
    ) -> tuple[list[RetrievedDoc], list[str]]:
        """返回 (命中的知识, 弱相关标题, 是否借助历史补全才命中)。

        检索策略：先查当前问句，检不到再用上一轮问题补全后重试。

        "先查再补"的顺序很重要：自带主题词的问题（"暴君什么时候打"）不应该拼历史，
        否则会引入无关话题；而依赖上文的问法（"那这个呢""被反野了怎么办"）
        单独检索必然检不到，自然触发补全。
        """
        prev_user = previous_user_message(history)

        boost_topic = None
        if prev_user:
            prev_docs, prev_hit = self.kb.search(
                prev_user, top_k=1, min_score=self.settings.retrieval_min_score, explain=False
            )
            if prev_hit and prev_docs:
                boost_topic = prev_docs[0].topic

        top_k = self.settings.retrieval_top_k
        min_score = self.settings.retrieval_min_score

        # 规则一：出现指代表达（"那这个""它"）时，当前问句没有自洽的指代对象，必须补上下文。
        # 否则「那这个位置前期该干嘛」会被泛化的「五个位置的分工与配合」抢先命中（15.75 分），
        # 而玩家真正问的是上一轮提到的那个位置。
        prefer_context = bool(prev_user) and detect_anaphora(message)
        query = augment_query(message, prev_user) if prefer_context else message

        docs, hit = self.kb.search(query, top_k=top_k, min_score=min_score, boost_topic=boost_topic)
        used_history = hit and prefer_context

        # 规则二：没有指代词时先单独检索（自带主题词的问题不该拼历史，避免话题漂移），
        # 检不到再用上一轮问题补全后重试——覆盖「被反野了怎么办」这类省略主语的追问。
        if not hit and prev_user and not prefer_context:
            docs, hit = self.kb.search(
                augment_query(message, prev_user), top_k=top_k, min_score=min_score, boost_topic=boost_topic
            )
            used_history = hit

        if hit:
            return docs, [], used_history

        # 未命中时取"弱相关"条目，仅用于给玩家指路，绝不作为来源展示。
        # 关键约束：只有**真的命中了标题/关键词/主题/实体**的条目才算相关。
        # 仅靠 BM25 字面重叠上榜的条目属于噪声（例如「巅峰赛的积分怎么算」会命中
        # 「补刀的含义与作用」），把它当成建议列表反而会误导玩家。
        relaxed_query = augment_query(message, prev_user) if prev_user else message
        relaxed, _ = self.kb.search(
            relaxed_query, top_k=3, min_score=min_score * 0.6, boost_topic=boost_topic
        )
        field_prefixes = ("title:", "keywords:", "topic:", "entity:")
        near_titles = [
            d.title
            for d in relaxed
            if d.score < min_score and any(r.startswith(field_prefixes) for r in d.match_reason)
        ]
        return [], near_titles, False


_agent: Agent | None = None


def get_agent(settings: Settings | None = None) -> Agent:
    global _agent
    if settings is not None:
        return Agent(settings)
    if _agent is None:
        _agent = Agent()
    return _agent


__all__ = ["Agent", "AgentResult", "get_agent", "next_request_id", "LLMError"]
