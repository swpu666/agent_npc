"""评测判分：规则断言（确定性硬约束）+ LLM 评审判要点覆盖。

两阶段设计（对应设计文档 §4.5）：
- 阶段一 规则断言：接口成功、意图/路由状态一致、引用来源合法、无伪造来源、
  高危措辞字面检查。任一硬约束失败即判定该案例失败，**不再调用评审模型**（省时省钱）。
- 阶段二 LLM 评审：逐条判断 must_cover 是否被覆盖，并要求给出回答中的原文片段作为证据；
  同时判断 must_not 是否被触发。评审输出经代码二次裁决，不直接采信模型自报的 verdict。

关键词判定的适用范围（必须明确的局限）：
字面匹配只用于 must_not_literal 中的少量高危措辞与知识条目编号比对；
语义等价表述一律交给 LLM 评审，不用关键词判定。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from app.agent.llm import LLMClient, LLMError, extract_json

_EVIDENCE_NOISE_RE = re.compile(r"[\s，。、！？；：\"'“”‘’（）()【】\[\]]+")
# 评审模型习惯用省略号跳过中间的无关内容（"……他反你上半区，你就去他下半区"），
# 这属于可接受的引用方式：按省略号分段后逐段核对，而不是要求整段连续出现。
_EVIDENCE_SPLIT_RE = re.compile(r"[…]{2,}|\.{3,}|。{3,}")


def _fold(text: str) -> str:
    return _EVIDENCE_NOISE_RE.sub("", text or "")


def evidence_in_answer(evidence: str, answer: str) -> bool:
    """校验评审给出的「原文片段」是否真的出现在回答里。

    评审模型偶尔会把要点原文复述一遍当作证据（编造证据），这类输出无法复核。
    但省略号拼接是合法引用，因此按省略号切段后逐段校验，避免误杀。
    """
    folded_answer = _fold(answer)
    fragments = [_fold(f) for f in _EVIDENCE_SPLIT_RE.split(evidence or "")]
    fragments = [f for f in fragments if len(f) >= 2]
    if not fragments:
        whole = _fold(evidence)
        return bool(whole) and whole in folded_answer
    return all(f in folded_answer for f in fragments)

JUDGE_SYSTEM_PROMPT = """你是严格的答案评审员。你需要判断一个游戏问答 Agent 的回答是否满足给定的要点要求。

规则：
1. 逐条判断「应覆盖要点」是否在回答中被覆盖。如果回答只是含糊带过、没有实际内容，判为未覆盖。
2. 语义等价即可算覆盖（例如「把对面水晶推掉就赢了」等价于「摧毁敌方水晶即获胜」），
   不要求字面相同。
3. evidence 必须是回答中**逐字连续出现**的短片段（不超过 30 字），直接从回答里复制，
   不要改写、不要拼接多段、不要补上列表编号、不要添加自己的标点。
   若一个要点由回答中多处内容共同覆盖，只引用其中最有代表性的一处。
   这是硬性要求：证据必须能被逐字检索到，否则该条判定无法复核。
4. 逐条判断「不应出现内容」是否出现。出现即记为命中，evidence 同样要逐字复制原文。
5. 只输出一个 JSON 对象，不要输出任何解释性文字或 Markdown 代码块。

输出格式：
{
  "points": [{"point": "要点原文", "covered": true, "evidence": "从回答中逐字复制的短片段（≤30字），未覆盖时留空"}],
  "forbidden_hits": [{"item": "不应出现内容", "evidence": "回答中的原文片段"}],
  "reason": "一句话说明"
}"""


@dataclass
class RuleOutcome:
    passed: bool
    checks: dict[str, bool] = field(default_factory=dict)
    failures: list[str] = field(default_factory=list)


@dataclass
class JudgeOutcome:
    verdict: str = "fail"            # 代码二次裁决后的结论
    judge_self_verdict: str = ""     # 评审模型自报结论，仅用于交叉核对
    covered: int = 0
    total: int = 0
    coverage_ratio: float = 0.0
    points: list[dict] = field(default_factory=list)
    forbidden_hits: list[dict] = field(default_factory=list)
    reason: str = ""
    judge_ms: int = 0
    judge_model: str = ""
    error: str | None = None


def rule_check(case: dict, response: dict | None, kb_ids: set[str], error: str | None = None) -> RuleOutcome:
    checks: dict[str, bool] = {}
    failures: list[str] = []

    def record(name: str, ok: bool, message: str) -> None:
        checks[name] = ok
        if not ok:
            failures.append(message)

    if error or response is None:
        record("success", False, f"调用失败：{error or '无响应'}")
        return RuleOutcome(passed=False, checks=checks, failures=failures)

    record("success", True, "")
    answer = str(response.get("answer") or "")
    citations = response.get("citations") or []
    citation_ids = [c.get("id") for c in citations]

    record("nonempty", len(answer.strip()) >= 5, f"回答过短或为空（{len(answer.strip())} 字）")
    record(
        "intent",
        response.get("intent") == case["expect_intent"],
        f"意图不符：期望 {case['expect_intent']}，实际 {response.get('intent')}",
    )
    record(
        "route_state",
        response.get("route_state") == case.get("expect_route_state"),
        f"路由状态不符：期望 {case.get('expect_route_state')}，实际 {response.get('route_state')}",
    )

    illegal = [i for i in citation_ids if i not in kb_ids]
    record("citations_legal", not illegal, f"引用了知识库中不存在的条目：{illegal}")

    if case.get("needs_kb"):
        expected_ids = set(case.get("expect_kb_ids") or [])
        missing = sorted(expected_ids - set(citation_ids))
        record("kb_hit", not missing, f"未检索到预期知识：{missing}（实际引用 {citation_ids}）")

    if case.get("expect_no_citations"):
        record("no_fabricated_sources", not citations, f"本次回答不应展示来源，实际展示了 {citation_ids}")

    if case.get("expect_miss"):
        record(
            "kb_miss_reported",
            response.get("route_state") == "kb_miss",
            "知识未覆盖的案例没有走 kb_miss 路由",
        )

    lowered = answer.lower()
    hit_literals = [s for s in (case.get("must_not_literal") or []) if s.lower() in lowered]
    record("literal_guard", not hit_literals, f"回答中出现高危措辞：{hit_literals}")

    return RuleOutcome(passed=all(checks.values()), checks=checks, failures=failures)


def build_judge_prompt(case: dict, turns: list[str], answer: str) -> str:
    context = "\n".join(f"{i + 1}. 玩家：{t}" for i, t in enumerate(turns))
    points = "\n".join(f"- {p}" for p in case.get("must_cover") or [])
    forbidden = "\n".join(f"- {p}" for p in case.get("must_not") or []) or "- （无）"

    sections = [
        f"【对话上下文】\n{context}",
        f"【需要评审的回答】\n{answer}",
        f"【应覆盖要点】\n{points}",
        f"【不应出现内容】\n{forbidden}",
    ]
    # 判分说明：把"评测出题人的意图"显式传达给评审模型。
    # 缺少这一层时，评审模型会因为"回答里出现某个词"就判定违规，
    # 而该词可能正是知识库原文（例如「主宰一般在暴君之后被争夺」）。
    note = case.get("judge_note")
    if note:
        sections.append(f"【判分说明（优先于上面的字面理解）】\n{note}")
    sections.append("请按系统提示的 JSON 格式输出评审结果。points 数组必须与「应覆盖要点」一一对应且顺序一致。")
    return "\n\n".join(sections)


def _decide(points: list[dict], forbidden_hits: list[dict]) -> tuple[str, float]:
    total = len(points)
    covered = sum(1 for p in points if p.get("covered"))
    ratio = covered / total if total else 0.0
    if forbidden_hits:
        return "fail", ratio
    if total == 0:
        return "fail", 0.0
    if covered == total:
        return "pass", ratio
    if ratio >= 0.7:
        return "partial", ratio
    return "fail", ratio


def llm_judge(llm: LLMClient, case: dict, turns: list[str], answer: str) -> JudgeOutcome:
    outcome = JudgeOutcome(judge_model=llm.model)
    prompt = build_judge_prompt(case, turns, answer)
    try:
        payload, result = llm.judge_json(JUDGE_SYSTEM_PROMPT, prompt)
        outcome.judge_ms = result.latency_ms
    except LLMError as exc:
        outcome.error = f"评审模型调用失败：{exc}"
        return outcome

    if not payload:
        outcome.error = "评审模型未返回可解析的 JSON"
        return outcome

    points = payload.get("points") or []
    if not isinstance(points, list):
        outcome.error = "评审结果 points 字段格式不合法"
        return outcome

    # 以期望要点为基准核对：模型漏返回或错位的一律按未覆盖处理
    expected_points = case.get("must_cover") or []
    normalized: list[dict] = []
    for idx, expected in enumerate(expected_points):
        item = points[idx] if idx < len(points) and isinstance(points[idx], dict) else {}
        evidence = str(item.get("evidence") or "")
        covered = bool(item.get("covered"))
        normalized.append(
            {
                "point": expected,
                "covered": covered,
                "evidence": evidence,
                "evidence_present": bool(evidence),
                # 判为已覆盖，且给出的证据确实来自回答原文 → 该条判定可被人工复核
                "evidence_verified": evidence_in_answer(evidence, answer) if covered else None,
            }
        )
    if len(points) != len(expected_points):
        outcome.error = f"评审要点数量不匹配（期望 {len(expected_points)}，返回 {len(points)}）"

    forbidden_hits = [h for h in (payload.get("forbidden_hits") or []) if isinstance(h, dict)]

    outcome.points = normalized
    outcome.forbidden_hits = forbidden_hits
    outcome.total = len(normalized)
    outcome.covered = sum(1 for p in normalized if p["covered"])
    outcome.coverage_ratio = outcome.covered / outcome.total if outcome.total else 0.0
    outcome.verdict, _ = _decide(normalized, forbidden_hits)
    outcome.judge_self_verdict = str(payload.get("verdict") or "")
    outcome.reason = str(payload.get("reason") or "")
    return outcome
