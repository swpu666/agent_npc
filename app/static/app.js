/* 训练向导「小玖」前端交互
 * - 多轮上下文：本地维护 turns，按最近 6 轮回传（后端不持久化会话）
 * - 等待状态：请求期间的阶段性文案 + 已用时间
 * - 失败处理：明确错误提示，并允许原样重新发送
 * - 来源展示：只渲染后端实际检索到的知识条目，无来源时不显示来源区
 */
(function () {
  "use strict";

  var MAX_INPUT = 1000;
  var HISTORY_TURNS = 6;

  var chatInner = document.getElementById("chat-inner");
  var chatBox = document.getElementById("chat");
  var emptyState = document.getElementById("empty-state");
  var input = document.getElementById("input");
  var composer = document.getElementById("composer");
  var btnSend = document.getElementById("btn-send");
  var btnClear = document.getElementById("btn-clear");
  var btnRetry = document.getElementById("btn-retry");
  var notice = document.getElementById("notice");
  var noticeText = document.getElementById("notice-text");
  var counter = document.getElementById("counter");
  var badgeModel = document.getElementById("badge-model");
  var badgeMock = document.getElementById("badge-mock");
  var emptyStateHTML = emptyState ? emptyState.outerHTML : "";
  var btnLogs = document.getElementById("btn-logs");
  var drawer = document.getElementById("drawer");
  var drawerBody = document.getElementById("drawer-body");
  var drawerMeta = document.getElementById("drawer-meta");
  var btnDrawerClose = document.getElementById("btn-drawer-close");

  var turns = [];        // 已完成的对话 [{role, content}]
  var busy = false;
  var lastFailed = null;

  // 会话标识：长期记忆按它隔离。没有账号体系，所以在浏览器本地生成并复用，
  // 换浏览器/清缓存就是"换一个人"。放在 localStorage 而不是内存里，
  // 是为了刷新页面后仍能接着累积（否则记忆每刷新一次就断了）。
  var SESSION_KEY = "wzry-tutor-session";
  var sessionId = (function () {
    try {
      var saved = window.localStorage.getItem(SESSION_KEY);
      if (saved) return saved;
      var made = "s" + Date.now().toString(36) + Math.random().toString(36).slice(2, 8);
      window.localStorage.setItem(SESSION_KEY, made);
      return made;
    } catch (e) {
      // 隐私模式等禁用了 localStorage：退化成"单次会话记忆"，不影响主流程
      return "s" + Date.now().toString(36);
    }
  })();

  var INTENT_LABEL = {
    knowledge_qa: "知识问答",
    situational_advice: "情境建议",
    chitchat: "闲聊",
    out_of_scope: "能力范围外"
  };
  var ROUTE_LABEL = {
    need_more_info: "需要补充条件",
    kb_miss: "知识库未收录",
    kb_gap: "知识库未覆盖",
    kb_general: "通用理解作答",
    unclear_input: "没听明白",
    conversation_recall: "回顾之前的问题"
  };
  // 意图判定的来源：规则命中是毫秒级，走模型兜底则要 1 秒上下。
  // 把它显示出来，玩家/评审才能解释"为什么这一轮慢"——否则那个数字只是一个谜。
  var INTENT_SOURCE_LABEL = {
    rule: "规则命中",
    inherit: "继承上下文",
    llm: "模型兜底分类",
    fallback: "默认兜底"
  };

  /* ------------------------------------------------------------ 启动 */
  fetch("api/health")
    .then(function (r) { return r.json(); })
    .then(function (d) {
      badgeModel.textContent = d.model + " · 知识库就绪";
      if (d.mock) {
        badgeMock.classList.remove("hidden");
        badgeModel.classList.add("badge-warn");
        badgeModel.textContent = d.model + "（未调用真实模型）";
      } else {
        badgeModel.classList.add("badge-live");
      }
      if (!d.has_api_key && !d.mock) {
        showNotice("后端未检测到有效模型密钥，请求会失败。请在 .env 中配置 LLM_API_KEY，或使用 MOCK_LLM=1 先验证页面。", false);
      }
    })
    .catch(function () { badgeModel.textContent = "服务状态未知"; });

  /* ------------------------------------------------------------ 工具 */
  function scrollToBottom() {
    chatBox.scrollTop = chatBox.scrollHeight;
  }

  function hideEmpty() {
    if (emptyState && emptyState.parentNode) {
      emptyState.parentNode.removeChild(emptyState);
      emptyState = null;
    }
  }

  function el(tag, className, text) {
    var node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined && text !== null) node.textContent = text;
    return node;
  }

  function showNotice(text, canRetry) {
    noticeText.textContent = text;
    notice.hidden = false;
    btnRetry.style.display = canRetry ? "" : "none";
  }

  function clearNotice() {
    notice.hidden = true;
    btnRetry.style.display = "";
  }

  /* ------------------------------------------------------------ 渲染 */
  function renderUser(text) {
    var msg = el("div", "msg msg-user");
    msg.appendChild(el("div", "msg-avatar", "我"));
    var wrap = el("div", "bubble-wrap");
    var bubble = el("div", "bubble", text);
    wrap.appendChild(bubble);
    msg.appendChild(wrap);
    chatInner.appendChild(msg);
    scrollToBottom();
  }

  function renderAssistant(payload) {
    var msg = el("div", "msg msg-npc");
    msg.appendChild(el("div", "msg-avatar", "玖"));
    var wrap = el("div", "bubble-wrap");
    wrap.appendChild(el("div", "bubble", payload.answer || "（空回答）"));

    // 元信息：意图 / 路由状态 / 耗时
    var meta = el("div", "meta");
    var intentTag = el("span", "tag tag-intent", "意图：" + (INTENT_LABEL[payload.intent] || payload.intent));
    meta.appendChild(intentTag);

    if (payload.route_state && ROUTE_LABEL[payload.route_state]) {
      meta.appendChild(el("span", "tag tag-route", ROUTE_LABEL[payload.route_state]));
    }
    // 只有"走了模型兜底"才标出来：这是意图耗时的唯一变因，标出来才解释得通
    var intentSource = payload.debug && payload.debug.intent && payload.debug.intent.source;
    if (intentSource === "llm" || intentSource === "fallback") {
      meta.appendChild(el("span", "tag tag-note", "意图：" + (INTENT_SOURCE_LABEL[intentSource] || intentSource)));
    }
    if (payload.mock) {
      meta.appendChild(el("span", "tag tag-note", "MOCK 结果"));
    }
    (payload.guard && payload.guard.notes ? payload.guard.notes : []).forEach(function (n) {
      if (n.indexOf("MOCK") === -1) meta.appendChild(el("span", "tag tag-note", n));
    });

    var t = payload.timings || {};
    meta.appendChild(el(
      "span", "tag tag-time",
      "耗时 " + (t.total_ms || 0) + "ms（意图 " + (t.intent_ms || 0) +
      " / 检索 " + (t.retrieval_ms || 0) + " / 生成 " + (t.llm_ms || 0) + "）"
    ));
    wrap.appendChild(meta);

    // 知识库覆盖不到的问题：明确告诉玩家这段是通用理解，且本次没有可引用的来源
    var GAP_NOTICE = {
      kb_gap: "本条没有直接对应的知识条目（知识库只收录通用玩法规则，没有英雄资料与使用率数据）。" +
        "以上为通用理解，仅供参考，因此不展示引用来源。",
      kb_general: "知识库里没有收录这条内容，以上是用通用理解作答，仅供参考，" +
        "具体数值与细节以客户端内说明为准，因此不展示引用来源。"
    };
    if (GAP_NOTICE[payload.route_state]) {
      wrap.appendChild(el("div", "gap-notice", GAP_NOTICE[payload.route_state]));
    }

    // 来源：只展示后端真实检索到的条目，没有就不显示
    var cits = payload.citations || [];
    if (cits.length) {
      var box = el("div", "citations");
      box.appendChild(el("div", "citations-title", "资料来源（" + cits.length + "）"));
      cits.forEach(function (c) {
        var a = el("a", "citation");
        a.href = c.url;
        a.target = "_blank";
        a.rel = "noopener noreferrer";
        var head = el("div", "citation-head");
        head.appendChild(el("span", "citation-id", c.id));
        head.appendChild(el("span", "citation-title", c.title));
        // 来源强度必须显式标出：topic 档只表示"该页面覆盖这个主题"，
        // 不等于逐句对应，不能让它看起来和直接来源一样权威
        if (c.support === "topic") {
          head.appendChild(el("span", "citation-support", "主题来源"));
        } else {
          head.appendChild(el("span", "citation-support citation-support-direct", "直接来源"));
        }
        a.appendChild(head);
        a.appendChild(el("div", "citation-meta",
          "采集日期 " + c.collected_at + " · 适用版本 " + c.version + " · 相关度 " + c.score));
        if (c.support_note) {
          a.appendChild(el("div", "citation-note", "说明：" + c.support_note));
        }
        box.appendChild(a);
      });
      wrap.appendChild(box);
    }

    msg.appendChild(wrap);
    chatInner.appendChild(msg);
    scrollToBottom();
  }

  function renderTyping() {
    var msg = el("div", "msg msg-npc");
    msg.appendChild(el("div", "msg-avatar", "玖"));
    var wrap = el("div", "bubble-wrap");
    var typing = el("div", "typing");
    var dots = el("div", "dots");
    dots.appendChild(el("span", "dot"));
    dots.appendChild(el("span", "dot"));
    dots.appendChild(el("span", "dot"));
    typing.appendChild(dots);
    var text = el("span", "typing-text", "正在识别意图…");
    typing.appendChild(text);
    wrap.appendChild(typing);
    msg.appendChild(wrap);
    chatInner.appendChild(msg);
    scrollToBottom();

    // 非流式接口：用阶段文案 + 计时让等待过程可感知
    var startedAt = Date.now();
    var stages = [
      [0, "正在识别意图…"],
      [220, "正在检索知识库…"],
      [700, "正在生成回答…"]
    ];
    var timer = setInterval(function () {
      var elapsed = Date.now() - startedAt;
      var label = stages[0][1];
      for (var i = 0; i < stages.length; i++) if (elapsed >= stages[i][0]) label = stages[i][1];
      text.textContent = label + "（已等待 " + (elapsed / 1000).toFixed(1) + "s）";
    }, 120);

    return function () {
      clearInterval(timer);
      if (msg.parentNode) msg.parentNode.removeChild(msg);
    };
  }

  /* ------------------------------------------------------------ 发送 */
  function buildHistory() {
    return turns.slice(-HISTORY_TURNS * 2).map(function (t) {
      return { role: t.role, content: t.content };
    });
  }

  function setBusy(value) {
    busy = value;
    btnSend.disabled = value;
    btnClear.disabled = value;
    btnSend.textContent = value ? "等待中…" : "发送";
  }

  function send(text) {
    text = (text || "").replace(/\s+$/, "");
    if (!text) {
      showNotice("请输入你的问题后再发送。", false);
      return;
    }
    if (text.length > MAX_INPUT) {
      showNotice("问题太长了，请控制在 " + MAX_INPUT + " 字以内。", false);
      return;
    }

    clearNotice();
    hideEmpty();
    renderUser(text);
    input.value = "";
    autoGrow();
    updateCounter();

    var history = buildHistory();
    var stopTyping = renderTyping();
    setBusy(true);
    lastFailed = { text: text, history: history };

    fetch("api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: text, history: history, session_id: sessionId })
    })
      .then(function (resp) {
        return resp.json().catch(function () { return {}; }).then(function (data) {
          return { ok: resp.ok, status: resp.status, data: data };
        });
      })
      .then(function (res) {
        stopTyping();
        setBusy(false);
        if (!res.ok) {
          var msg = (res.data && res.data.message) || ("请求失败（HTTP " + res.status + "）");
          showNotice(msg + " 点击「重新发送」可重试。", true);
          return;
        }
        renderAssistant(res.data);
        turns.push({ role: "user", content: text });
        turns.push({ role: "assistant", content: res.data.answer || "" });
        lastFailed = null;
      })
      .catch(function (err) {
        stopTyping();
        setBusy(false);
        showNotice("网络请求失败：" + (err && err.message ? err.message : "未知错误") + "。点击「重新发送」可重试。", true);
      });
  }

  /* ------------------------------------------------------------ 事件 */
  composer.addEventListener("submit", function (e) {
    e.preventDefault();
    if (busy) return;
    send(input.value.trim());
  });

  input.addEventListener("keydown", function (e) {
    if (e.key === "Enter" && !e.shiftKey && !e.isComposing) {
      e.preventDefault();
      if (!busy) send(input.value.trim());
    }
  });

  btnRetry.addEventListener("click", function () {
    if (busy) return;
    // 没有待重发的提问时要给出反馈，而不是静默 return（那样玩家只会觉得按钮坏了）
    if (!lastFailed) {
      showNotice("当前没有失败待重发的提问。", false);
      return;
    }
    var pending = lastFailed;
    // 失败的请求没有写入 turns，这里只需移除界面上的那条提问气泡后原样重发
    var userMsgs = chatInner.querySelectorAll(".msg-user");
    if (userMsgs.length) chatInner.removeChild(userMsgs[userMsgs.length - 1]);
    send(pending.text);
  });

  btnClear.addEventListener("click", function () {
    turns = [];
    lastFailed = null;
    clearNotice();
    chatInner.innerHTML = "";
    if (emptyStateHTML) {
      chatInner.insertAdjacentHTML("afterbegin", emptyStateHTML);
      emptyState = document.getElementById("empty-state");
    }
  });

  /* ------------------------------------------------------------ 使用记录抽屉 */
  var INTENT_SHORT = {
    knowledge_qa: "知识",
    situational_advice: "情境",
    chitchat: "闲聊",
    out_of_scope: "越界"
  };

  function renderLogItem(r) {
    var item = el("div", "log-item");
    var head = el("div", "log-head");
    head.appendChild(el("span", "log-time", String(r.ts || "").replace("T", " ").slice(0, 19)));
    head.appendChild(el("span", "log-ip", String(r.ip || "-")));
    head.appendChild(el("span", "tag tag-intent", INTENT_SHORT[r.intent] || r.intent || "?"));
    if (r.route_state) head.appendChild(el("span", "tag tag-route", ROUTE_LABEL[r.route_state] || r.route_state));
    var t = r.timings || {};
    head.appendChild(el("span", "tag tag-time", (t.total_ms || 0) + "ms"));
    item.appendChild(head);
    item.appendChild(el("div", "log-q", "问：" + (r.message || "")));
    if (r.answer) item.appendChild(el("div", "log-a", "答：" + r.answer));
    if (r.error) item.appendChild(el("div", "log-err", "错误：" + r.error));
    return item;
  }

  function openDrawer() {
    drawer.hidden = false;
    drawerBody.innerHTML = "";
    drawerMeta.textContent = "加载中…";
    // 只请求自己的记录：按 session_id 过滤，而不是拉取全部。
    fetch("api/logs?limit=60&session_id=" + encodeURIComponent(sessionId))
      .then(function (r) { return r.json(); })
      .then(function (d) {
        if (!d.enabled) {
          drawerMeta.textContent = "使用记录已关闭（LOG_ENABLED=0）";
          drawerBody.appendChild(el("div", "log-empty", "后端已关闭使用记录写入。"));
          return;
        }
        var records = d.records || [];
        drawerMeta.textContent = "我的记录 " + records.length + " 条 · 会话 " + sessionId;
        if (!records.length) {
          drawerBody.appendChild(el("div", "log-empty", "这个会话还没有记录。发几条问题后再来看。"));
          return;
        }
        records.forEach(function (r) { drawerBody.appendChild(renderLogItem(r)); });
        drawerBody.appendChild(el(
          "div", "log-empty",
          "这里只显示你自己的记录。查看所有会话的记录请访问 /agentnpc/allNpc/（管理视角）。"
        ));
      })
      .catch(function () {
        drawerMeta.textContent = "读取失败";
        drawerBody.appendChild(el("div", "log-empty", "无法读取使用记录，请确认服务仍在运行。"));
      });
  }

  function closeDrawer() { drawer.hidden = true; }

  /* ------------------------------------------------------------ 玩家画像 */
  var btnMemory = document.getElementById("btn-memory");
  var drawerMemory = document.getElementById("drawer-memory");
  var memoryBody = document.getElementById("memory-body");
  var memoryMeta = document.getElementById("memory-meta");
  var btnMemoryClose = document.getElementById("btn-memory-close");
  var btnMemoryClear = document.getElementById("btn-memory-clear");

  function memoryCard(title, lines) {
    var card = el("div", "log-item");
    card.appendChild(el("div", "log-head", title));
    lines.forEach(function (line) {
      card.appendChild(el("div", "log-a", line));
    });
    return card;
  }

  // 统一拿 JSON：区分「网络失败 / HTTP 非 2xx / 非 JSON 响应」三种失败，便于精确报错。
  // 后端的 memory / knowledge-gaps 接口在任何情况下都返回 200 + 合法 JSON，
  // 所以“读取失败”几乎一定是请求没到达或网关回了 HTML 错误页（见 eval/badcases/README.md）。
  //
  // 注意：url 必须是**相对地址**（"api/…"），由页面里的 <base href="/agentnpc/"> 决定最终前缀。
  // 写成绝对路径 "/api/…" 会被解析到站点根目录、绕过子路径部署，网关直接回 404 —
  // 实测「画像读取失败 / 知识缺口读取失败」就是这么来的。
  function getJson(url) {
    return fetch(url).then(function (r) {
      if (!r.ok) {
        var httpErr = new Error("HTTP " + r.status);
        httpErr.kind = "http";
        httpErr.status = r.status;
        throw httpErr;
      }
      return r.text().then(function (text) {
        try {
          return JSON.parse(text);
        } catch (e) {
          var jsonErr = new Error("non-json");
          jsonErr.kind = "nonjson";
          throw jsonErr;
        }
      });
    });
  }

  // 把一次 fetch 失败翻译成给玩家看的中文提示（区分根因，不再笼统说“无法读取画像”）。
  function memoryFailCard(what, err) {
    var hint;
    if (!err) {
      hint = "网络请求失败，请确认服务仍在运行（或页面是通过 http(s) 打开的，而非本地文件）。";
    } else if (err.kind === "http") {
      hint = "服务返回了错误码 " + err.status + "，请联系管理员或稍后重试。";
    } else if (err.kind === "nonjson") {
      hint = "服务返回的内容不是预期的数据（可能是网关/代理的错误页）。请确认后端服务正常。";
    } else {
      hint = "无法读取" + what + "，请确认服务仍在运行。";
    }
    memoryBody.appendChild(memoryCard(what + "读取失败", [hint]));
  }

  function openMemory() {
    drawerMemory.hidden = false;
    memoryBody.innerHTML = "";
    memoryMeta.textContent = "会话 " + sessionId;
    // 画像接口失败：只报画像，不连坐知识缺口（两者失败原因可能不同）。
    getJson("api/memory/" + encodeURIComponent(sessionId))
      .then(function (d) {
        var s = d.profile || {};
        memoryBody.appendChild(memoryCard("系统记住的内容（本会话）", d.summary || []));
        var detail = [];
        if (s.turns) detail.push("提问次数：" + s.turns);
        if (s.intents) {
          detail.push("提问类型：" + Object.keys(s.intents).map(function (k) {
            return INTENT_SHORT[k] || k;
          }).join("、"));
        }
        if (s.heroes && Object.keys(s.heroes).length) {
          detail.push("提到过的英雄：" + Object.keys(s.heroes).join("、"));
        }
        if (s.recent_questions && s.recent_questions.length) {
          detail.push("最近问过：" + s.recent_questions.slice(-3).join(" ／ "));
        }
        if (detail.length) memoryBody.appendChild(memoryCard("明细", detail));
        if (!d.enabled) {
          memoryBody.appendChild(memoryCard("状态", ["长期记忆已在后端关闭（MEMORY_ENABLED=0）"]));
        }
      })
      .catch(function (err) { memoryFailCard("画像", err); });
    // 知识缺口接口单独请求、单独报错，避免把它的失败误报成“画像失败”。
    getJson("api/knowledge-gaps?limit=10")
      .then(function (g) {
        var lines = (g.gaps || []).map(function (item) {
          return item.count > 1 ? item.question + "（被问 " + item.count + " 次）" : item.question;
        });
        memoryBody.appendChild(memoryCard(
          "知识库未覆盖的问题（跨会话聚合，按被问次数排序）",
          lines.length
            ? lines
            : ["暂时没有：最近问过的问题知识库都接住了。"]
        ));
        memoryBody.appendChild(el(
          "div", "log-empty",
          "说明：这里统计的是「知识库没有直接资料、只能靠通用理解回答」的问题，"
          + "按被问次数排序——反复出现的就该补进知识库。"
        ));
      })
      .catch(function (err) { memoryFailCard("知识缺口", err); });
  }

  function closeMemory() { drawerMemory.hidden = true; }

  if (btnMemory) btnMemory.addEventListener("click", openMemory);
  if (btnMemoryClose) btnMemoryClose.addEventListener("click", closeMemory);
  if (drawerMemory) {
    drawerMemory.addEventListener("click", function (e) { if (e.target === drawerMemory) closeMemory(); });
  }
  if (btnMemoryClear) {
    btnMemoryClear.addEventListener("click", function () {
      fetch("api/memory/" + encodeURIComponent(sessionId), { method: "DELETE" })
        .then(function (r) { return r.json(); })
        .then(function (d) {
          memoryMeta.textContent = d.cleared ? "已清除该会话记忆" : "本来就没有记录";
          memoryBody.innerHTML = "";
          memoryBody.appendChild(memoryCard("已清除", ["这个会话的画像已被删除，后续会重新开始累积。"]));
        })
        .catch(function () {
          memoryBody.appendChild(memoryCard("清除失败", ["无法连接到服务，请确认服务仍在运行。"]));
        });
    });
  }

  /* ------------------------------------------------------------ 设计文档 */
  var btnDesign = document.getElementById("btn-design");
  if (btnDesign) {
    btnDesign.addEventListener("click", function () {
      window.open("design", "_blank");
    });
  }

  /* ------------------------------------------------------------ 评测报告 */
  // 直接在服务端渲染的报告页打开；报告是自包含 HTML，也可以离线双击查看
  var btnReports = document.getElementById("btn-reports");
  if (btnReports) {
    btnReports.addEventListener("click", function () {
      window.open("report", "_blank");
    });
  }

  if (btnLogs) btnLogs.addEventListener("click", openDrawer);
  if (btnDrawerClose) btnDrawerClose.addEventListener("click", closeDrawer);
  if (drawer) {
    drawer.addEventListener("click", function (e) { if (e.target === drawer) closeDrawer(); });
  }
  document.addEventListener("keydown", function (e) {
    if (e.key !== "Escape") return;
    closeDrawer();
    closeMemory();
  });

  // 示例问题使用事件委托，清空对话重建空状态后依然有效
  chatInner.addEventListener("click", function (e) {
    var target = e.target;
    if (target && target.classList && target.classList.contains("chip")) {
      send(target.textContent.trim());
    }
  });

  function autoGrow() {
    input.style.height = "auto";
    input.style.height = Math.min(input.scrollHeight, 160) + "px";
  }

  function updateCounter() {
    var n = input.value.length;
    counter.textContent = n + "/" + MAX_INPUT;
    counter.classList.toggle("over", n > MAX_INPUT);
  }

  input.addEventListener("input", function () {
    autoGrow();
    updateCounter();
  });

  input.focus();
  updateCounter();
})();
