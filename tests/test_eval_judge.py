"""评测判分逻辑单测：规则断言、证据可复核性、结论裁决。"""

from __future__ import annotations

from eval.judge import _decide, evidence_in_answer, rule_check

KB_IDS = {"MAP-OVERLORD-001", "OPS-FARM-001"}


def make_case(**overrides) -> dict:
    case = {
        "id": "TX",
        "expect_intent": "knowledge_qa",
        "expect_route_state": None,
        "needs_kb": True,
        "expect_kb_ids": ["OPS-FARM-001"],
        "expect_no_citations": False,
        "expect_miss": False,
        "must_not_literal": [],
    }
    case.update(overrides)
    return case


def make_response(**overrides) -> dict:
    resp = {
        "answer": "补刀就是由自己打出对最后一个小兵的最后一击。",
        "intent": "knowledge_qa",
        "route_state": None,
        "citations": [{"id": "OPS-FARM-001"}],
    }
    resp.update(overrides)
    return resp


# --------------------------------------------------------------- 规则断言
def test_rule_check_passes_on_expected_response() -> None:
    outcome = rule_check(make_case(), make_response(), KB_IDS)
    assert outcome.passed, outcome.failures


def test_rule_check_flags_wrong_intent() -> None:
    outcome = rule_check(make_case(), make_response(intent="chitchat"), KB_IDS)
    assert not outcome.passed
    assert any("意图不符" in f for f in outcome.failures)


def test_rule_check_flags_missing_expected_knowledge() -> None:
    outcome = rule_check(make_case(), make_response(citations=[{"id": "MAP-OVERLORD-001"}]), KB_IDS)
    assert not outcome.passed
    assert any("未检索到预期知识" in f for f in outcome.failures)


def test_rule_check_flags_illegal_citation() -> None:
    outcome = rule_check(make_case(), make_response(citations=[{"id": "FAKE-999"}]), KB_IDS)
    assert not outcome.passed
    assert any("不存在" in f for f in outcome.failures)


def test_rule_check_flags_fabricated_sources_when_not_needed() -> None:
    case = make_case(needs_kb=False, expect_kb_ids=[], expect_no_citations=True)
    outcome = rule_check(case, make_response(), KB_IDS)
    assert not outcome.passed
    assert any("不应展示来源" in f for f in outcome.failures)


def test_rule_check_flags_high_risk_phrasing() -> None:
    case = make_case(must_not_literal=["已为你查询"])
    outcome = rule_check(case, make_response(answer="已为你查询到你的段位是钻石。"), KB_IDS)
    assert not outcome.passed
    assert any("高危措辞" in f for f in outcome.failures)


def test_rule_check_handles_call_failure() -> None:
    outcome = rule_check(make_case(), None, KB_IDS, error="timeout: 模型调用超时")
    assert not outcome.passed
    assert outcome.checks["success"] is False
    assert any("调用失败" in f for f in outcome.failures)


# --------------------------------------------------------------- 证据可复核
def test_evidence_in_answer_ignores_punctuation_and_spaces() -> None:
    answer = "补刀，就是由你自己打出\n对敌方小兵的最后一击。"
    assert evidence_in_answer("补刀就是由你自己打出对敌方小兵的最后一击", answer)
    assert evidence_in_answer("敌方小兵", answer)


def test_evidence_in_answer_accepts_ellipsis_spliced_quotes() -> None:
    """评审用省略号跳过中间内容属于合法引用，不应被判为编造证据。"""
    answer = (
        "被反野，按这个顺序处理：\n"
        "1. 先判断他走没走。如果对面打野还在你野区，硬进就是送。\n"
        "2. 换野，别死守。他反你上半区，你就去他下半区。\n"
    )
    evidence = "被反野，按这个顺序处理：……如果对面打野还在你野区……他反你上半区，你就去他下半区"
    assert evidence_in_answer(evidence, answer)


def test_report_html_is_self_contained_and_renders_key_parts() -> None:
    """报告 HTML 必须是自包含单文件（不依赖外部资源），评审才能离线打开。"""
    from eval.report_html import render_report_html

    payload = {
        "config": {"run_at": "2026-09-22 18:00:00", "model": "deepseek-chat", "tag": "unit"},
        "summary": {
            "metrics": {
                "意图准确率": {"value": 1.0, "numerator": 2, "denominator": 2},
                "回答通过率": {"value": 0.5, "numerator": 1, "denominator": 2},
            },
            "latency_ms": {
                "total": {"count": 2, "mean": 10, "p50": 10, "p90": 12, "max": 12},
                "model_call_share": 0.5,
            },
        },
        "records": [
            {
                "id": "T01",
                "turns": ["红BUFF有什么用？"],
                "intent": "knowledge_qa",
                "passed": True,
                "answer": "红BUFF让普攻附带伤害并减速。",
                "citations": [{"id": "MAP-BUFF-001"}],
                "timings": {"total_ms": 1200, "intent_ms": 0.2, "retrieval_ms": 0.3, "llm_ms": 1199},
                "judge_points": [
                    {"point": "说明作用", "covered": True, "evidence": "普攻附带伤害", "evidence_verified": True}
                ],
            },
            {
                "id": "T02",
                "turns": ["巅峰赛积分怎么算？"],
                "intent": "knowledge_qa",
                "passed": False,
                "answer": "知识库未收录。",
                "fail_reason": "要点未覆盖",
                "judge_points": [],
            },
        ],
    }
    out = render_report_html(payload, title="单测报告")
    assert out.startswith("<!DOCTYPE html>")
    assert "<style>" in out and "<script>" in out
    assert "http://" not in out.replace("http://www.w3.org", ""), "不应引入外部资源"
    assert "100.00%" in out and "50.00%" in out
    assert "红BUFF让普攻附带伤害并减速。" in out
    assert "要点未覆盖" in out
    assert "普攻附带伤害" in out  # 评审证据要能一眼对照
    assert "实际发起模型调用的案例占比：50%" in out


def test_report_html_escapes_untrusted_content() -> None:
    """回答内容会进 HTML，必须转义，避免脚本注入。"""
    from eval.report_html import render_report_html

    payload = {
        "config": {},
        "summary": {},
        "records": [{"id": "T01", "turns": ["x"], "answer": "<img src=x onerror=alert(1)>", "passed": True}],
    }
    out = render_report_html(payload)
    # html.escape 会同时转义 < > & " '
    assert "&lt;img src=x onerror=alert(1)&gt;" in out
    assert "<img src=x" not in out


def test_evidence_in_answer_rejects_fabricated_evidence() -> None:
    answer = "补刀可以拿到更多金币。"
    # 评审把要点原文抄回来当证据，属于编造证据
    assert not evidence_in_answer("补刀指由自己完成对敌方小兵的最后一击", answer)
    assert not evidence_in_answer("", answer)


def test_evidence_in_answer_rejects_partially_spliced_fabrication() -> None:
    """只要有一段对不上，就说明该条判定无法复核。"""
    answer = "先判断他走没走，硬进就是送。"
    evidence = "先判断他走没走……叫队友帮忙看视野"
    assert not evidence_in_answer(evidence, answer)


# --------------------------------------------------------------- 结论裁决
def test_decide_pass_when_all_points_covered() -> None:
    verdict, ratio = _decide([{"covered": True}, {"covered": True}], [])
    assert verdict == "pass" and ratio == 1.0


def test_decide_partial_when_seventy_percent_covered() -> None:
    points = [{"covered": True}, {"covered": True}, {"covered": True}, {"covered": False}]
    verdict, ratio = _decide(points, [])
    assert verdict == "partial" and ratio == 0.75


def test_decide_fail_on_forbidden_hit_even_if_all_points_covered() -> None:
    verdict, _ = _decide([{"covered": True}], [{"item": "x", "evidence": "y"}])
    assert verdict == "fail"


def test_decide_fail_below_threshold() -> None:
    verdict, ratio = _decide([{"covered": True}, {"covered": False}], [])
    assert verdict == "fail" and ratio == 0.5
