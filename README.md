# 训练向导「小玖」· 王者荣耀问答 Agent（最小可用 Demo）

面向玩家的游戏问答 Agent：**网页提问 → 意图识别 → 按需检索本地知识库 → 生成回答 → 网页展示结果与来源**，
并附带一套可重复运行的离线评测。设计说明见 [DESIGN.md](DESIGN.md)。

- 游戏：王者荣耀（原创向导 NPC「小玖」，不复刻官方角色）
- 技术栈：Python 3.11+ / FastAPI + 原生 HTML/CSS/JS
- 模型：任意 OpenAI 兼容接口（DeepSeek / 通义千问 / GPT 等），密钥只留在后端
- 返回方式：非流式 JSON，响应中带意图、引用来源与分环节耗时

---

## 一、目录速览

```
app/
  main.py                 FastAPI 装配入口
  config.py               配置（全部来自 .env）
  schemas.py              接口请求/响应模型
  api/chat.py             /api/chat、/api/health、/api/kb/stats
  agent/
    normalize.py          文本归一化、注入检测、指代与条件提取
    synonyms.py           同义词/简称/错别字词典 + 意图规则词表
    intent.py             意图识别与路由（可独立单测）
    retrieval.py          知识检索：同义词扩展 + BM25 + 字段加权 + 版本折扣
    prompt.py             系统提示、材料注入、历史裁剪
    llm.py                模型客户端（超时/重试/Mock）
    templates.py          确定性模板回答
    pipeline.py           Agent 编排链路
  data/knowledge/         知识库（31 条，5 类主题，含来源与版本）
  tools/kb_lint.py        知识库自检
  static/                 玩家使用页面
eval/
  run_eval.py             离线评测入口
  dataset/testset.jsonl   14 条测试集
  judge.py                规则断言 + LLM 评审
  metrics.py              指标聚合（分母口径显式）
  reports/                评测报告（json + md）
tests/                    97 个单测（意图 / 检索 / 编排 / 接口 / 判分）
```

---

## 二、快速开始

### 1. 安装依赖

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows
# source .venv/bin/activate       # macOS / Linux
pip install -r requirements.txt
```

### 2. 配置模型

```bash
copy .env.example .env            # Windows
# cp .env.example .env            # macOS / Linux
```

编辑 `.env`，至少填好这三项（示例为 DeepSeek，换成任意 OpenAI 兼容服务都可以）：

```ini
LLM_BASE_URL=https://api.deepseek.com/v1
LLM_API_KEY=sk-你的密钥
LLM_MODEL=deepseek-chat
```

`.env` 已被 `.gitignore` 忽略，不会进入仓库；前端页面与静态资源中不含任何密钥。

### 3. 启动服务

```bash
python -m app.tools.kb_lint        # 可选：知识库自检（零依赖）
uvicorn app.main:app --port 8000
```

浏览器打开 <http://127.0.0.1:8000/>。页面支持连续追问、清空对话、等待状态、失败重发，
知识类回答下方会展示实际使用的来源（标题 / 链接 / 采集日期 / 适用版本 / 相关度）。

### 4. 跑离线评测

```bash
python -m eval.run_eval                       # 默认：Agent 核心链路 + 测试集 14 例 + LLM 评审
python -m eval.run_eval --mode http           # 改为通过 HTTP 调用 /api/chat，验证真实接口链路
python -m eval.run_eval --no-judge            # 只跑规则断言，不消耗评审模型额度
python -m eval.run_eval --only T06,T09,T13    # 只跑指定案例
```

报告写入 `eval/reports/report_<tag>_<时间戳>.json` 与 `.md`，包含逐条结果、汇总指标、
耗时分布，以及未通过案例的原文与评审证据（便于人工抽查）。

仓库中已保留的报告（可复核）：

| 文件 | 说明 |
| --- | --- |
| `report_final1_*.{json,md}`、`report_final2_*.{json,md}` | 真实模型下的两次最终评测，各项质量指标均为 100% |
| `report_baseline_*.{json,md}` | 接入真实模型后的第 1 次全量运行，回答通过率 92.86%（T06 未通过） |
| `report_judge_rule_issue_*.{json,md}` | 暴露"判分规则过严导致 T05 假失败"的那次运行 |
| `report_smoke_*.{json,md}` | Mock 模式的链路验证结果，**不作为质量结论** |

问题定位与修正过程记录在设计文档 [DESIGN.md](DESIGN.md) 的 §4.5 与 §6。

---

## 三、Mock 模式（链路验证专用）

没有模型密钥时，可以用 Mock 模式先验证页面与接口是否连通：

```bash
# Windows PowerShell
$env:MOCK_LLM=1; uvicorn app.main:app --port 8000
$env:MOCK_LLM=1; python -m eval.run_eval --tag smoke --no-judge
```

Mock 模式**不会调用任何真实模型**，会在三处显著标识：页面顶部 `MOCK 模式` 徽标、
接口响应 `mock: true`、评测报告文件名前缀 `smoke` 与正文声明。
**Mock 结果只用于验证链路连通，不能替代真实模型调用及其质量评测。**

---

## 四、接口

### `POST /api/chat`

```json
{
  "message": "红BUFF多久刷新一次？",
  "history": [
    { "role": "user", "content": "打野的职责是什么？" },
    { "role": "assistant", "content": "打野负责……" }
  ]
}
```

响应（节选）：

```json
{
  "request_id": "r-20260922-0001",
  "answer": "……",
  "intent": "knowledge_qa",
  "route_state": null,
  "citations": [
    { "id": "MAP-BUFF-003", "title": "红 BUFF 与蓝 BUFF 的刷新节奏",
      "url": "https://pvp.qq.com/…", "collected_at": "2026-09-22",
      "version": "数值随版本调整，以客户端内说明为准", "score": 8.2 }
  ],
  "timings": { "total_ms": 1840, "intent_ms": 1, "retrieval_ms": 4, "llm_ms": 1780 },
  "model": "deepseek-chat",
  "mock": false,
  "retryable": false,
  "guard": { "injection_suspected": false, "notes": [] }
}
```

| 字段 | 说明 |
| --- | --- |
| `intent` | `knowledge_qa` / `situational_advice` / `chitchat` / `out_of_scope` |
| `route_state` | `need_more_info`（情境条件不足，已反问）、`kb_miss`（知识未收录）、`null` |
| `citations` | 只包含本次实际检索到的知识条目；无资料支持时为空数组，绝不伪造来源 |
| `timings` | 意图 / 检索 / 生成三段分别实测，单位毫秒 |

错误约定：

| 场景 | 状态码 | `error` | 前端行为 |
| --- | --- | --- | --- |
| 空输入 | 400 | `empty_input` | 明确提示 |
| 超过 1000 字 | 400 | `too_long` | 明确提示并保留输入 |
| 模型超时 | 504 | `timeout` | 展示「重新发送」 |
| 未配置密钥 / 模型 4xx | 502 | `config` / `http_4xx` | 展示原因，不重试 |
| 模型 5xx / 连接失败 | 502 | `http_5xx` / `connection` | 已按配置重试，仍失败可手动重发 |

另有 `GET /api/health`（服务状态、模型名、是否 Mock）与 `GET /api/kb/stats`（知识库统计）。

---

## 五、测试

```bash
python -m pytest tests -q
```

覆盖：意图四分类与优先级、注入防护、省略式追问继承、多轮检索改写、检索召回与阈值、
版本折扣、编排链路的模板短路与引用校验、接口参数校验与错误码、静态资源不泄露密钥、
评测判分逻辑（规则断言 / 证据可复核性 / 结论裁决）。

---

## 六、已验证环境与实测结果

| 项 | 值 |
| --- | --- |
| 操作系统 | Windows 11（10.0.22621） |
| Python | 3.12.0（代码兼容 3.11+） |
| 被测 / 评审模型 | `deepseek-chat`（评审 `temperature=0`，独立调用、独立计时） |
| 单测 | `pytest tests -q` → **97 passed** |
| 知识库自检 | `python -m app.tools.kb_lint` → 31 条，5 类主题，**0 错误 0 警告** |
| Mock 链路 | 服务可启动、页面可访问、`/api/chat` 返回完整字段 |

真实模型离线评测（14 个案例，连续 2 次运行，报告见 `eval/reports/report_final*.{json,md}`）：

| 指标 | 结果 | 分母 |
| --- | --- | --- |
| 意图准确率 | 100.00% | 全部 14 例 |
| 知识检索命中率 | 100.00% | 需要资料的 6 例 |
| 回答通过率 | 100.00% | 全部 14 例 |
| 评审证据可复核率 | 100.00% | 评审判为"已覆盖"的 31 个要点 |
| 调用成功率 | 100.00% | 全部 14 例 |

响应耗时（毫秒）：

| 环节 | 均值 | P50 | P90 |
| --- | --- | --- | --- |
| 意图路由（本地） | 0.2–0.4 | 0.1–0.2 | 0.5 |
| 知识检索（本地） | 0.3 | 0.2–0.3 | 0.6–0.7 |
| **模型生成** | **746–805** | 1007–1094 | 1329–1723 |
| 总耗时（调用模型的 8 例） | 1307–1410 | 1262–1357 | 1585–1822 |
| 总耗时（走模板的 6 例） | < 1ms | < 1ms | < 1ms |
| 评审模型（单列，不计入上面） | 1399–1509 | 1368–1513 | 1653–1774 |

一次问答里只有 **57.1%（8/14）** 的案例真正调用模型，其余走确定性模板；
调用模型的案例中，**等待时间的 99.9% 花在模型生成上**，本地意图与检索合计不到 1ms。

---

## 七、知识库

`app/data/knowledge/kb_v1.json`，31 条，覆盖 5 类主题：

| 主题 | 条目数 |
| --- | --- |
| 地图机制 | 9 |
| 位置职责 | 6 |
| 基础操作 | 6 |
| 对局目标 | 5 |
| 资源与经济 | 5 |

每条包含：唯一编号、主题、子主题、事实键、标题、正文、关键词、实体、适用版本、
来源（标题 / 链接 / 采集日期）、置信度。自检命令：

```bash
python -m app.tools.kb_lint                    # 字段完整性、id 唯一、主题合法、版本标注
python -m app.tools.kb_lint --check-urls       # 额外联网校验来源链接可达性
python -m app.tools.kb_lint --write-meta       # 刷新 kb_meta.json 统计
```

---

## 八、已知限制

- 非流式返回，首字感知延迟较高（用页面等待状态与阶段文案缓解）。
- 未引入向量检索，极端口语化表述可能漏召；漏召时如实回答「知识库未收录」。
- 会话不持久化，刷新页面即清空（题目未要求）。
- 数值类知识（如 BUFF 刷新间隔）随版本调整，知识库只标注版本范围，回答中会提示以客户端内说明为准。
