"""指标聚合与分母口径。

所有分母都显式写出，避免出现"看起来很高但分母不可解释"的指标：
- 意图准确率 / 回答通过率 / 成功率：分母是全部案例；
- 知识检索命中率：分母是"需要知识支撑"的案例，且只看检索层，不掺入生成质量；
- 耗时：只在成功调用的案例上统计，评审模型耗时单列，不混入 Agent 响应耗时。
"""

from __future__ import annotations

from collections import defaultdict


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    rank = (len(ordered) - 1) * p
    low = int(rank)
    high = min(low + 1, len(ordered) - 1)
    frac = rank - low
    return float(ordered[low] * (1 - frac) + ordered[high] * frac)


def _latency(values: list[float]) -> dict:
    if not values:
        return {"count": 0, "mean": 0.0, "p50": 0.0, "p90": 0.0, "max": 0.0}
    return {
        "count": len(values),
        "mean": round(sum(values) / len(values), 1),
        "p50": round(percentile(values, 0.5), 1),
        "p90": round(percentile(values, 0.9), 1),
        "max": round(max(values), 1),
    }


def summarize(records: list[dict]) -> dict:
    total = len(records)
    if total == 0:
        return {"total_cases": 0}

    success = [r for r in records if r["success"]]
    needs_kb = [r for r in records if r.get("needs_kb")]
    judged = [r for r in records if r.get("judged")]

    intent_ok = sum(1 for r in records if r.get("intent_ok"))
    kb_hit = sum(1 for r in needs_kb if r.get("kb_hit"))
    rule_pass = sum(1 for r in records if r.get("rule_pass"))
    judge_pass = sum(1 for r in judged if r.get("judge_verdict") == "pass")
    full_pass = sum(1 for r in records if r.get("passed"))

    def rate(num: int, den: int) -> float:
        return round(num / den, 4) if den else 0.0

    by_intent: dict[str, dict] = defaultdict(lambda: {"cases": 0, "passed": 0, "intent_ok": 0})
    for r in records:
        bucket = by_intent[r["expect_intent"]]
        bucket["cases"] += 1
        bucket["passed"] += 1 if r.get("passed") else 0
        bucket["intent_ok"] += 1 if r.get("intent_ok") else 0

    return {
        "total_cases": total,
        "metrics": {
            "意图准确率": {
                "value": rate(intent_ok, total),
                "numerator": intent_ok,
                "denominator": total,
                "applicable": "全部案例（含多轮、注入、能力范围外）",
            },
            "知识检索命中率": {
                "value": rate(kb_hit, len(needs_kb)),
                "numerator": kb_hit,
                "denominator": len(needs_kb),
                "applicable": "仅统计需要知识支撑的案例；只看检索层是否拿到预期知识，不计生成质量",
            },
            "回答通过率": {
                "value": rate(full_pass, total),
                "numerator": full_pass,
                "denominator": total,
                "applicable": "全部案例；需规则断言通过且评审判定 pass",
            },
            "规则断言通过率": {
                "value": rate(rule_pass, total),
                "numerator": rule_pass,
                "denominator": total,
                "applicable": "全部案例；确定性硬约束，失败者不进入评审阶段",
            },
            "评审通过率": {
                "value": rate(judge_pass, len(judged)),
                "numerator": judge_pass,
                "denominator": len(judged),
                "applicable": "仅统计实际进入评审阶段的案例",
            },
            "调用成功率": {
                "value": rate(len(success), total),
                "numerator": len(success),
                "denominator": total,
                "applicable": "全部案例；无错误码返回即视为成功",
            },
            "评审证据可复核率": {
                "value": rate(
                    sum(r.get("evidence_verified", 0) for r in judged),
                    sum(r.get("evidence_checked", 0) for r in judged),
                ),
                "numerator": sum(r.get("evidence_verified", 0) for r in judged),
                "denominator": sum(r.get("evidence_checked", 0) for r in judged),
                "applicable": "评审判定为「已覆盖」的要点数；衡量评审给出的原文证据是否真的来自回答，用于识别评审模型编造证据",
            },
        },
        "latency_ms": {
            "说明": "仅统计调用成功的案例；评审模型耗时单列，不计入 Agent 响应耗时",
            "total": _latency([r["total_ms"] for r in success]),
            # 分开统计"真的调用了模型"和"走确定性模板"的案例：
            # 模板路径（能力范围外 / 知识未命中 / 需补充条件）为 0 次模型调用，
            # 混在一起算总耗时均值会得到"均值小于 P50"这种不可解释的数字。
            "total_with_llm": _latency([r["total_ms"] for r in success if r.get("llm_ms")]),
            "total_template_only": _latency([r["total_ms"] for r in success if not r.get("llm_ms")]),
            "model_call_share": round(
                sum(1 for r in success if r.get("llm_ms")) / max(1, len(success)), 4
            ),
            "intent": _latency([r["intent_ms"] for r in success]),
            "retrieval": _latency([r["retrieval_ms"] for r in success]),
            "llm": _latency([r["llm_ms"] for r in success if r.get("llm_ms") is not None]),
            "judge": _latency([r["judge_ms"] for r in judged if r.get("judge_ms")]),
        },
        "by_intent": {k: v for k, v in sorted(by_intent.items())},
    }
