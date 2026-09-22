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
    ROUTE_KB_GAP,
    ROUTE_KB_MISS,
    ROUTE_NEED_MORE_INFO,
    ROUTE_UNCLEAR_INPUT,
    IntentResult,
    classify,
)
from app.agent.llm import LLMClient, LLMError
from app import memory
from app.agent.normalize import extract_urls, is_no_coverage_leading
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
    # 长期记忆的当前状态摘要（仅在带 session_id 调用时非空），用于页面展示"系统记住了什么"
    memory: dict | None = None


def _citation(doc: RetrievedDoc) -> dict:
    return {
        "id": doc.id,
        "title": doc.title,
        "url": doc.source_url,
        "collected_at": doc.collected_at,
        "version": doc.version,
        "score": round(doc.score, 3),
        "topic": doc.topic,
        "support": doc.source_support,
        "support_note": doc.source_support_note,
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
        self.memory = memory.MemoryStore(self.settings)

    # ------------------------------------------------------------------ 主链路
    def answer(
        self,
        message: str,
        history: list[dict] | None = None,
        session_id: str | None = None,
    ) -> AgentResult:
        result = AgentResult(request_id=next_request_id(), model=self.llm.model, mock=self.settings.mock_llm)
        t_start = time.perf_counter()

        # [0] 读取长期记忆（本地文件，实测 < 0.5ms）。放在最前面是因为它只影响后续
        #     提示词的展开方向，不参与路由判定——记忆不该改变"这句话该走哪条路"。
        profile = self.memory.load(session_id) if session_id else None
        profile_note = memory.render_profile_note(profile)

        # [1] 意图路由（本地规则，通常 < 1ms）
        t0 = time.perf_counter()
        prev_intent, prev_route_state = self._previous_turn_state(history)
        intent_result: IntentResult = classify(
            message,
            history=history,
            llm_classifier=lambda text, hist: self.llm.classify_intent(text),
            inherit_intent=prev_intent,
            prev_route_state=prev_route_state,
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

        # kb_gap 的判定与检索无关（问题本身就落在知识库覆盖缺口上），
        # 因此不能被"未命中"覆盖成 kb_miss：这里只在路由状态还是空的时候才置 kb_miss。
        if intent_result.intent == INTENT_KNOWLEDGE and not docs and result.route_state is None:
            result.route_state = ROUTE_KB_MISS

        # [3] 生成：能确定回答的直接走模板，不需要模型
        llm_ms = 0
        if intent_result.intent == INTENT_OUT_OF_SCOPE:
            result.answer = templates.out_of_scope_answer(intent_result.oos_reason)
        elif result.route_state == ROUTE_UNCLEAR_INPUT:
            # "没听懂"走模板：0 次模型调用、毫秒级返回，语气可以精心打磨成拟人化表达，
            # 不必让模型临场发挥（也更省一次调用）。
            result.answer = templates.unclear_input_answer(message)
        elif result.route_state == ROUTE_KB_MISS:
            result.answer = templates.kb_miss_answer(near_titles)
        elif result.route_state == ROUTE_NEED_MORE_INFO:
            result.answer = templates.need_more_info_answer(intent_result.missing_conditions)
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
                conditions=intent_result.conditions,
                missing_conditions=intent_result.missing_conditions,
                already_asked_conditions=intent_result.already_asked_conditions,
                profile_note=profile_note,
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

        result.retrieved = docs
        if result.route_state == ROUTE_KB_GAP:
            # kb_gap：检索到的条目与问题核心诉求并不匹配，不展示任何来源。
            # 早期版本会把这类"半相关"条目当来源展示，玩家会误以为它支撑了回答。
            result.citations = []
            result.guard["notes"].append(
                "本条没有直接对应的知识条目，回答基于通用理解，因此不展示引用来源"
                + (f"（检索到的 {len(docs)} 条材料仅作背景）" if docs else "")
            )
        else:
            result.citations = [_citation(d) for d in docs]
            # 兜底：万一还有漏网的"半相关命中"，只要模型开门见山就说没收录，
            # 就统一按"覆盖不足"处理——既清空来源，也把路由状态标上。
            # 只清来源不改状态会导致界面自相矛盾：标着"正常回答"却一个来源都没有。
            if is_no_coverage_leading(result.answer):
                result.citations = []
                if result.route_state is None:
                    result.route_state = ROUTE_KB_GAP
                result.guard["notes"].append("回答自述该问题未收录，已按覆盖不足处理并清空来源")

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

        # [5] 更新长期记忆：只记录可观测的提问行为（问了什么、属于哪类、命中了哪个主题）。
        #     写入失败不能影响回答，MemoryStore 内部已吞掉 OSError。
        if session_id:
            updated = self.memory.update(
                session_id,
                message=message,
                intent=result.intent,
                route_state=result.route_state,
                doc_topics=[d.topic for d in docs],
                conditions=intent_result.conditions,
                answer_had_kb=bool(result.citations),
            )
            if updated:
                result.memory = {
                    "turns": updated.turns,
                    "topics": dict(updated.topics),
                    "profile_injected": bool(profile_note),
                }

        result.debug = {
            "intent": result.intent_detail,
            "retrieval_used_history": used_history,
            "retrieved": [d.to_dict() for d in docs] if docs else [],
            "near_miss_titles": near_titles,
            # kb_gap 时检索到的"半相关"条目仍作为背景传给模型，但不展示为来源；
            # 放在 debug 里方便评审核对"到底检到了什么、为什么没展示"。
            "withheld_materials": [d.id for d in docs] if result.route_state == ROUTE_KB_GAP else [],
            "profile_note": profile_note,
        }
        return result

    # ------------------------------------------------------------------ 内部
    @staticmethod
    def _last_user_message(history: list[dict] | None) -> str | None:
        return previous_user_message(history)

    @staticmethod
    def _previous_turn_state(history: list[dict] | None) -> tuple[str | None, str | None]:
        """规则层复算上一轮用户消息的意图与路由状态（本地、0 次模型调用）。

        上一轮的状态是理解本轮的关键：上一条刚问过"补充条件"，这一条几乎必然在回答条件。
        复算只走规则（不传 llm_classifier），因此即使上一轮本身需要兜底，
        这里也只得到一个确定性的近似值——用于上下文继承已经足够，不会引入额外延迟。
        """
        if not history:
            return None, None
        for idx in range(len(history) - 1, -1, -1):
            item = history[idx]
            if isinstance(item, dict) and item.get("role") == "user" and item.get("content"):
                prev_result = classify(str(item["content"]), history=history[:idx])
                return prev_result.intent, prev_result.route_state
        return None, None

    def _retrieve(
        self,
        message: str,
        history: list[dict] | None,
        intent_result: IntentResult,
    ) -> tuple[list[RetrievedDoc], list[str]]:
        """返回 (命中的知识, 弱相关标题, 是否借助历史补全)。

        检索策略：只要存在上一轮用户消息，就把"上一轮问题 + 当前问句"拼在一起检索，
        让上下文始终参与召回，不再对"自带主题词的问题"单独检索命中后跳过历史。

        说明：早期版本先单独检索、检不到才补历史，目的是让自带主题词的问句避免话题漂移；
        但那会让"既能单独命中、又确实依赖上文"的问句丢掉上下文。现改为有历史就一律拼接。
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

        # 有上一轮用户消息时一律拼进查询一起检索，让上下文始终参与召回。
        query = augment_query(message, prev_user) if prev_user else message
        docs, hit = self.kb.search(query, top_k=top_k, min_score=min_score, boost_topic=boost_topic)
        used_history = bool(prev_user)

        if hit:
            return docs, [], used_history

        # 未命中时取"弱相关"条目，仅用于给玩家指路，绝不作为来源展示。
        # 关键约束：只有**真的命中了标题/关键词/主题/实体**的条目才算相关。
        # 仅靠 BM25 字面重叠上榜的条目属于噪声（例如「巅峰赛的积分怎么算」会命中
        # 「补刀的含义与作用」），把它当成建议列表反而会误导玩家。
        relaxed, _ = self.kb.search(
            query, top_k=3, min_score=min_score * 0.6, boost_topic=boost_topic
        )
        field_prefixes = ("title:", "keywords:", "topic:", "entity:")
        near_titles = [
            d.title
            for d in relaxed
            if d.score < min_score and any(r.startswith(field_prefixes) for r in d.match_reason)
        ]
        return [], near_titles, used_history


_agent: Agent | None = None


def get_agent(settings: Settings | None = None) -> Agent:
    global _agent
    if settings is not None:
        return Agent(settings)
    if _agent is None:
        _agent = Agent()
    return _agent


__all__ = ["Agent", "AgentResult", "get_agent", "next_request_id", "LLMError"]
