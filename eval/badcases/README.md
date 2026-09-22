# BadCase 收纳：前端「玩家画像」抽屉报错「无法读取画像」

现象：点页面右上角「玩家画像」→ 抽屉里出现一张红色卡片：

> **读取失败** —— 无法读取画像，请确认服务仍在运行。

本文把这一条报错按**真实根因**拆成若干 badcase，并给出复现、诊断与修复建议。
结论先行：**这条报错几乎都不是「画像读取逻辑」的问题，而是「请求没成功拿到合法 JSON」**。

---

## 一、为什么后端不会真的「读不到画像」

已用 `fastapi.testclient` 直接打接口验证：

| 场景 | 请求 | 真实返回 |
| --- | --- | --- |
| 画像不存在 | `GET /api/memory/<sid>` | `200`，`{"enabled":true,"turns":0,"summary":["这个会话还没有记录…"],"profile":null}` |
| 任意 session_id | `GET /api/memory/anything` | `200`，同上 |
| 知识缺口 | `GET /api/knowledge-gaps?limit=10` | `200` |

后端 `app/api/chat.py:get_memory` 把「文件不存在 / JSON 损坏 / 非法 session_id」全部兜底成
`200 + profile:null`（`app/memory.py:load` 吞掉了 `OSError / JSONDecodeError`）。
真正的「没有画像」会被渲染成 *"这个会话还没有记录（至少问 3 次后才会用于调整回答）"*，
**而不是**「读取失败」。

所以：一旦看到「读取失败」，一定是前端 `fetch` 链路在网络/传输层失败，或拿到的响应体不是 JSON。

---

## 二、BadCase 分类

### BC-1　服务没起 / 已崩溃（网络层失败）　【最常见】

- **现象**：点「玩家画像」立刻红卡；同时左上角模型徽标显示「服务状态未知」。
- **根因**：`fetch("api/memory/…")` 直接 reject（如 `ERR_CONNECTION_REFUSED`），进入 `.catch`。
- **诊断**：看 health 徽标。health 走的是同一条相对路径，若它也失败，必是后端没运行。
- **修复**：先起服务（`uvicorn app.main:app --port 8000`），再访问页面。

### BC-2　用 `file://` 直接双击打开 `index.html`

- **现象**：聊天、health、画像**全部**失败。
- **根因**：相对路径 `api/memory/…` 在 `file://` 下被解析成 `file:///…/api/memory/…`，fetch 失败进 catch。
- **修复**：必须通过 `http(s)` 访问（起 uvicorn 或 nginx），不能本地双击。

### BC-3　反向代理 / 网关返回 HTML 错误页（502 / 504）

- **现象**：服务本身正常，但前面有 nginx / 网关；网关超时或 502 时回一个 HTML 页面。
  前端 `.json()` 解析 HTML 失败 → 进 `.catch`，但错误被笼统说成「无法读取画像」。
- **根因**：前端没区分「网络错误」和「非 JSON 响应」，也没区分**两次 fetch 到底哪次挂了**。
- **复现**：让上游对 `/api/memory/` 或 `/api/knowledge-gaps` 返回 `Content-Type: text/html` 的 502 页。
- **修复**：见第三节代码改动（区分错误类型 + 两次 fetch 分别报错）。

### BC-4　相对 / 绝对路径不一致，部署在子路径下 404

- **现象**：`api/memory/`（行 435/489）用相对路径，而 `/api/knowledge-gaps`（行 457）用绝对路径。
  若部署在子路径（如 `https://host/app/`），相对路径解析成 `/app/api/memory/…`，后端只在 `/api/…`，导致 404 / 网关错误。
- **根因**：同一个抽屉里两种写法混用，子路径部署下必有一边错。
- **修复**：统一成绝对路径 `/api/…`（或加 `<base href>`）。见第三节。

### BC-5（潜在）错误消息有误导性：可能是「知识缺口」先挂

- **现象 / 根因**：`openMemory()` 用一条 `.catch` 包住了**两次** fetch（先 `memory`，再 `knowledge-gaps`）。
  任一次失败都会显示「无法读取画像」——但实际上可能是第二段「知识库未覆盖问题」接口挂了，
  画像本身是好好的。这会导致排错方向错。
- **修复**：两次 fetch 各自 try/catch，分开提示（见第三节）。

### BC-6（理论，当前不会发生）后端非预期 500

- 当前 `load()` 已吞掉文件/JSON 异常；`to_dict()/summary_lines()` 也都是纯内存操作，
  不会抛。但 FastAPI 的 500 本身返回的是 JSON（`{"detail":…}`），前端 `.json()` 仍能成功，
  **不会**进 catch——只会渲染成空。属于「失败不透明」，列为待观察项，不是当前报错来源。

---

## 三、诊断清单（看到红卡先查这几项）

1. 左上角模型徽标是「服务状态未知」吗？→ 是 = BC-1 / BC-2。
2. 地址栏是 `http(s)://…` 还是 `file://…`？→ `file://` = BC-2。
3. 打开浏览器 DevTools → Network，看 `/api/memory/<sid>` 与 `/api/knowledge-gaps`：
   - 状态是 `(failed) net::ERR_*` → BC-1 / BC-3 网络层；
   - 状态 502/504 且响应是 HTML → BC-3；
   - 状态 404 且路径带多余前缀 → BC-4。
4. 单独 `curl http://127.0.0.1:8000/api/memory/<sid>` 是否能拿到 JSON？能 = 后端没问题，问题在前端/代理。

---

## 四、修复建议（已落地在 `app/static/app.js`）

- 抽一个 `getJson(url)`，区分三种失败：网络错误 / HTTP 非 2xx / 非 JSON 响应。
- 两次 fetch 分别捕获，画像失败说「画像读取失败」，知识缺口失败说「知识缺口读取失败」，不再张冠李戴。
- `api/memory/…` 改为绝对路径 `/api/memory/…`，与 `knowledge-gaps` 保持一致（BC-4）。

---

## 五、怎么把新 badcase 收纳进来

每遇到一条「无法读取画像」或同类前端报错，按上面表格补齐：
`BC 编号 / 现象 / 根因 / 复现步骤 / 诊断方法 / 是否修复`。
优先固化成可复现的复现步骤（curl 或 DevTools 截图），再决定是否值得补进
`eval/dialogues/cases.jsonl` 做回归。
