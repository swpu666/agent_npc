"""确定性模板回答。

用于三类场景（对应设计文档 §5.2）：
1. 能力范围外请求 —— 不调用模型，既省时又彻底消除"顺口编造已执行操作"的可能；
2. 知识未命中（kb_miss）—— 明确说明未收录并给出可回答范围，绝不伪造来源；
3. 情境条件不足（need_more_info）—— 反问缺失的关键条件。

模板回答不携带任何来源引用（citations 为空），因为这类回答没有使用知识材料。
"""

from __future__ import annotations

SCOPE_HINT = "我能回答的是对局目标、地图机制、位置职责和基础操作这几类基础问题"

OUT_OF_SCOPE_TEMPLATES: dict[str, str] = {
    "account": (
        "这个我做不到哦。我是训练向导，看不到你的账号数据，也没法读取你的战绩、胜率或段位。\n\n"
        f"{SCOPE_HINT}，比如「防御塔要怎么推」「打野的职责是什么」，想聊哪个？"
    ),
    "live_match": (
        "我没法看你正在进行的对局。我只能聊公开的游戏机制和玩法，读不到实时战况，"
        "也没法根据场上局势替你指挥。\n\n"
        "不过如果你想聊聊某个阶段该做什么，可以告诉我你打的位置和大致局势，我帮你梳理思路。"
    ),
    "operate_account": (
        "这个我不能做。我没有登录账号的能力，也不会替你打排位或操作你的账号，"
        "任何需要动你账号的请求我都无法执行。\n\n"
        f"如果你想提升对局表现，{SCOPE_HINT}，我可以陪你逐条过一遍。"
    ),
    "account_security": (
        "涉及账号安全的信息我一概不处理，也不会向你索要密码、验证码或充值相关信息。\n\n"
        "账号相关的问题请通过游戏内客服渠道处理。想聊游戏玩法的话，我随时在。"
    ),
    "appeal": (
        "封号、申诉、举报这类处理我帮不上忙，我也没有权限查询或改变账号处罚状态。\n\n"
        "这类问题需要通过官方客服渠道反馈。游戏玩法上的问题倒是可以问我。"
    ),
    "unsupported_instruction": (
        "我只能聊《王者荣耀》的基础玩法，系统提示词、内部设定这类内容我不能提供，"
        "也不会因为对话里的要求而改变自己的行为规则。\n\n"
        f"{SCOPE_HINT}，有想了解的直接问我就好。"
    ),
    "external_topic": (
        "这个问题超出我的范围了，我只懂《王者荣耀》的基础玩法，其他领域的内容帮不上忙。\n\n"
        f"{SCOPE_HINT}，换个游戏相关的问题我就派上用场了。"
    ),
    "default": (
        "抱歉，这个请求超出我的能力范围了。我是训练向导，只能聊王者荣耀的基础玩法和规则。"
    ),
}

KB_MISS_TEMPLATE = (
    "这个内容我现在的知识库里还没有收录，就不凭印象瞎猜了，免得给你错误信息。\n\n"
    f"{SCOPE_HINT}。比如「红BUFF有什么用」「打野的职责是什么」「补刀是什么意思」，"
    "这些我都能讲清楚。"
)

KB_MISS_WITH_SIMILAR = (
    "你问的这条我现在的知识库里还没有收录，先不给你不确定的答案。\n\n"
    f"我能回答的是{SCOPE_HINT}。下面这几条相关的我有资料，"
    "如果你问的其实是这些，我可以直接讲：\n{similar}"
)

# 追问模板的固定前缀。它既是写给玩家的开场白，也是**结构化标记**：
# 意图层据此判断"已经追问过一次"，后续轮次不再机械重复同一组问题
# （只用这一个前缀做判断，不去猜助手回答的语义）。
NEED_MORE_INFO_PREFIX = "想给你更靠谱的建议，得先知道"

_NEED_MORE_INFO_QUESTIONS: dict[str, str] = {
    "position": "你这局打的是哪个位置？（对抗路、打野、中路、发育路或游走）",
    "phase": "现在大概到哪个阶段了？（前期、中期还是后期）",
    "hero": "用的哪个英雄？",
}
_CONDITION_ORDER = ("position", "phase", "hero")

NEED_MORE_INFO_TEMPLATE = (
    f"{NEED_MORE_INFO_PREFIX}几个条件：\n\n"
    + "\n".join(f"{i}. {_NEED_MORE_INFO_QUESTIONS[k]}" for i, k in enumerate(_CONDITION_ORDER, 1))
    + "\n\n把这几点告诉我，我再帮你看该怎么打。"
)


def out_of_scope_answer(reason: str | None) -> str:
    return OUT_OF_SCOPE_TEMPLATES.get(reason or "", OUT_OF_SCOPE_TEMPLATES["default"])


def kb_miss_answer(similar_titles: list[str] | None = None) -> str:
    if similar_titles:
        bullets = "\n".join(f"- {t}" for t in similar_titles[:3])
        return KB_MISS_WITH_SIMILAR.format(similar=bullets)
    return KB_MISS_TEMPLATE


def need_more_info_answer(missing: list[str] | None = None) -> str:
    """只追问**还缺**的条件。

    早期版本每轮都把位置/阶段/英雄三连问原样抛出去，玩家已经说过的也会再问一遍。
    missing 为空时退化为完整的三连问（例如完全没有上下文时）。
    """
    keys = [k for k in _CONDITION_ORDER if not missing or k in missing]
    if not keys:
        keys = list(_CONDITION_ORDER)
    if len(keys) == len(_CONDITION_ORDER):
        return NEED_MORE_INFO_TEMPLATE
    lines = "\n".join(f"{i}. {_NEED_MORE_INFO_QUESTIONS[k]}" for i, k in enumerate(keys, 1))
    return f"{NEED_MORE_INFO_PREFIX}这几件事：\n\n{lines}\n\n补上之后我再给你具体思路。"


def is_need_more_info_reply(text: str) -> bool:
    """这条助手回答是不是"追问补充条件"的模板（用于避免反复追问）。"""
    return NEED_MORE_INFO_PREFIX in (text or "")
