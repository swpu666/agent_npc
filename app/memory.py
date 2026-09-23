"""长期记忆：会话级用户画像 + 跨会话知识缺口聚合。

## 它做什么

1. **用户画像**（按 `session_id` 隔离）：累计提问次数、意图分布、关注的知识主题、
   提到过的位置与英雄、最近问过的原话。用于生成回答时**方向性地**更贴合玩家关注点。
2. **知识缺口聚合**（跨会话）：把所有会话里"没答上来"的问题汇总排序——
   这是**用真实使用数据驱动知识库迭代**的入口，比拍脑袋猜玩家要问什么靠谱。
   注意：**页面上已撤下这份清单，只保留 `GET /api/knowledge-gaps` 接口**。
   原因见 DESIGN.md §2.4——未命中判据放宽后，聚合里会混进"你有病"这类非知识输入，
   按被问次数排序看着像结论、其实不准；要让数据可信，用之前得先按意图过滤。

## 刻意不做的事

- **不做"喜好结论"**。统计到"他这 5 次提问里有 3 次问地图机制"是事实；
  但推论"他喜欢玩打野"就是猜了——从"问了什么"推"喜欢玩什么"这一步没有依据，
  而且一旦猜错会持续影响后续每一次回答。所以画像只记录**可观测的提问行为**，
  且在注入提示词时明确标注"这是统计，不是玩家的自我声明，不要生硬复述"。
- **不跨会话共享个人画像**。`session_id` 不同就是两个人（本 Demo 没有账号体系，
  这个隔离是它能提供的最强保证）。

## 隐私

画像文件包含玩家问过的**原始问题文本**，与使用记录同级别。
存在 `data/memory/<session_id>.json`，已加入 .gitignore；提供 `DELETE` 接口可清除。
上线前同样需要脱敏、告知与保留期，详见 README「使用记录与隐私」。
"""

from __future__ import annotations

import json
import re
import threading
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from app.config import BASE_DIR, Settings, get_settings

_LOCK = threading.Lock()

# session_id 来自前端，属于不可信输入：只允许字母数字与 -_，防目录穿越
_SESSION_RE = re.compile(r"^[A-Za-z0-9_\-]{1,64}$")

# 注入画像的最低门槛：问得还太少时，统计没有意义，硬讲反而像在监视玩家
MIN_TURNS_FOR_PROFILE = 3
MAX_RECENT_QUESTIONS = 8
MAX_UNRESOLVED = 10

_POSITION_LABELS = {
    "打野": "打野",
    "对抗路": "对抗路",
    "上路": "对抗路",
    "中单": "中路",
    "中路": "中路",
    "法师": "中路",
    "射手": "发育路",
    "发育路": "发育路",
    "下路": "发育路",
    "辅助": "游走",
    "游走": "游走",
}


def valid_session_id(session_id: str | None) -> str | None:
    s = (session_id or "").strip()
    return s if s and _SESSION_RE.match(s) else None


@dataclass
class UserProfile:
    session_id: str
    turns: int = 0
    intents: Counter = field(default_factory=Counter)
    topics: Counter = field(default_factory=Counter)
    positions: Counter = field(default_factory=Counter)
    heroes: Counter = field(default_factory=Counter)
    route_states: Counter = field(default_factory=Counter)
    recent_questions: list[str] = field(default_factory=list)
    unresolved: list[str] = field(default_factory=list)
    first_seen: str = ""
    last_seen: str = ""

    # ---------------------------------------------------------------- 序列化
    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "turns": self.turns,
            "intents": dict(self.intents),
            "topics": dict(self.topics),
            "positions": dict(self.positions),
            "heroes": dict(self.heroes),
            "route_states": dict(self.route_states),
            "recent_questions": self.recent_questions,
            "unresolved": self.unresolved,
            "first_seen": self.first_seen,
            "last_seen": self.last_seen,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "UserProfile":
        return cls(
            session_id=data.get("session_id", ""),
            turns=int(data.get("turns", 0)),
            intents=Counter(data.get("intents") or {}),
            topics=Counter(data.get("topics") or {}),
            positions=Counter(data.get("positions") or {}),
            heroes=Counter(data.get("heroes") or {}),
            route_states=Counter(data.get("route_states") or {}),
            recent_questions=list(data.get("recent_questions") or []),
            unresolved=list(data.get("unresolved") or []),
            first_seen=data.get("first_seen", ""),
            last_seen=data.get("last_seen", ""),
        )

    # ---------------------------------------------------------------- 派生视图
    def top_topics(self, n: int = 3) -> list[tuple[str, int]]:
        return self.topics.most_common(n)

    def summary_lines(self) -> list[str]:
        """给玩家/评审看的画像摘要（只陈述可观测事实）。"""
        lines = [f"累计提问 {self.turns} 次"]
        if self.intents:
            parts = "、".join(f"{k} {v} 次" for k, v in self.intents.most_common())
            lines.append(f"提问类型：{parts}")
        if self.topics:
            lines.append("关注主题：" + "、".join(f"{k}（{v}）" for k, v in self.top_topics(4)))
        if self.positions:
            lines.append("提到过的位置：" + "、".join(f"{k}（{v}）" for k, v in self.positions.most_common(3)))
        if self.heroes:
            lines.append("提到过的英雄：" + "、".join(f"{k}（{v}）" for k, v in self.heroes.most_common(5)))
        return lines


class MemoryStore:
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self.dir = Path(self.settings.memory_dir)
        if not self.dir.is_absolute():
            self.dir = BASE_DIR / self.dir

    def _path(self, session_id: str) -> Path | None:
        sid = valid_session_id(session_id)
        if not sid:
            return None
        path = (self.dir / f"{sid}.json").resolve()
        try:
            path.relative_to(self.dir.resolve())
        except ValueError:
            return None
        return path

    # ---------------------------------------------------------------- 读写
    def load(self, session_id: str | None) -> UserProfile | None:
        path = self._path(session_id or "")
        if path is None or not path.is_file():
            return None
        try:
            return UserProfile.from_dict(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError):
            return None

    def save(self, profile: UserProfile) -> None:
        if not self.settings.memory_enabled:
            return
        path = self._path(profile.session_id)
        if path is None:
            return
        try:
            self.dir.mkdir(parents=True, exist_ok=True)
            with _LOCK:
                path.write_text(
                    json.dumps(profile.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
                )
        except OSError:
            # 记忆写入失败绝不影响正常问答
            pass

    def clear(self, session_id: str) -> bool:
        path = self._path(session_id)
        if path is None or not path.is_file():
            return False
        try:
            path.unlink()
            return True
        except OSError:
            return False

    def list_sessions(self) -> list[str]:
        if not self.dir.is_dir():
            return []
        return sorted(p.stem for p in self.dir.glob("*.json") if _SESSION_RE.match(p.stem))

    # ---------------------------------------------------------------- 更新
    def update(
        self,
        session_id: str | None,
        *,
        message: str,
        intent: str,
        route_state: str | None,
        doc_topics: list[str] | None = None,
        conditions: dict[str, list[str]] | None = None,
        answer_had_kb: bool = False,
    ) -> UserProfile | None:
        """记录一轮问答。

        只记录**可观测的提问行为**：问了什么、属于哪类、命中了哪个主题、提到过什么。
        不做"他喜欢玩什么"这类推断——那一步没有依据，猜错还会持续影响后续回答。
        """
        sid = valid_session_id(session_id)
        if not sid or not self.settings.memory_enabled:
            return None

        now = datetime.now().astimezone().isoformat(timespec="seconds")
        profile = self.load(sid) or UserProfile(session_id=sid, first_seen=now)
        profile.turns += 1
        profile.last_seen = now
        profile.intents[intent] += 1
        profile.route_states[str(route_state or "null")] += 1

        for topic in doc_topics or []:
            if topic:
                profile.topics[topic] += 1
        for pos in (conditions or {}).get("position", []):
            label = _POSITION_LABELS.get(pos, pos)
            profile.positions[label] += 1
        for hero in (conditions or {}).get("hero", []):
            profile.heroes[hero] += 1

        question = (message or "").strip()[:120]
        if question and (not profile.recent_questions or profile.recent_questions[-1] != question):
            profile.recent_questions.append(question)
        profile.recent_questions = profile.recent_questions[-MAX_RECENT_QUESTIONS:]

        # 没答上来的问题单独留档：这是最值得回补知识库的信号。
        # kb_general 同样计入——它虽然用通用理解答了，但知识库确实缺这条材料。
        if not answer_had_kb and route_state in (None, "kb_miss", "kb_gap", "kb_general"):
            if question and question not in profile.unresolved:
                profile.unresolved.append(question)
            profile.unresolved = profile.unresolved[-MAX_UNRESOLVED:]

        self.save(profile)
        return profile


def render_profile_note(profile: UserProfile | None) -> str:
    """把画像渲染成注入提示词的说明。

    三条约束写进了文案本身，因为它们都是真实踩过的坑：
    1. 说明这是**统计**而非玩家自述，避免模型当成事实复述（"你上次说你喜欢打野"）；
    2. 只用于**调整展开方向**，不许拿来推断玩家身份或做人格评价；
    3. 不足 `MIN_TURNS_FOR_PROFILE` 轮时不注入——样本太小，讲了反而像在监视。
    """
    if profile is None or profile.turns < MIN_TURNS_FOR_PROFILE:
        return ""
    lines = ["【关于这位玩家的历史提问统计（自动累积，不是他本人说的）】"]
    if profile.top_topics(3):
        lines.append("常问方向：" + "、".join(f"{k}（{v} 次）" for k, v in profile.top_topics(3)))
    if profile.positions:
        lines.append("提到过的位置：" + "、".join(f"{k}（{v} 次）" for k, v in profile.positions.most_common(2)))
    if profile.heroes:
        lines.append("提到过的英雄：" + "、".join(f"{k}" for k, _ in profile.heroes.most_common(4)))
    lines.append(
        "使用方式：仅用于决定「相关的话可以多展开一点」的方向，不要复述这些统计"
        "（不要说「你经常问 XX」），也不要据此推断他喜欢玩什么位置或英雄——"
        "这是统计结果，不是他的自我说明。"
    )
    return "\n".join(lines)


def aggregate_knowledge_gaps(store: MemoryStore, limit: int = 20) -> list[dict]:
    """跨会话聚合"没答上来"的问题，按出现次数排序。

    相比拍脑袋猜玩家会问什么，这份清单来自**真实使用数据**：
    同类问题被反复问到却没收录，就是知识库最该补的地方。
    """
    counter: Counter = Counter()
    for sid in store.list_sessions():
        profile = store.load(sid)
        if not profile:
            continue
        for question in profile.unresolved:
            counter[question] += 1
    return [{"question": q, "count": c} for q, c in counter.most_common(limit)]


_store: MemoryStore | None = None


def get_memory_store(settings: Settings | None = None) -> MemoryStore:
    global _store
    if settings is not None:
        return MemoryStore(settings)
    if _store is None:
        _store = MemoryStore()
    return _store
