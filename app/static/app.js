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

  var INTENT_LABEL = {
    knowledge_qa: "知识问答",
    situational_advice: "情境建议",
    chitchat: "闲聊",
    out_of_scope: "能力范围外"
  };
  var ROUTE_LABEL = {
    need_more_info: "需要补充条件",
    kb_miss: "知识库未收录",
    kb_gap: "知识库未覆盖"
  };

  /* ------------------------------------------------------------ 启动 */
  fetch("/api/health")
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
    if (payload.route_state === "kb_gap") {
      wrap.appendChild(el(
        "div", "gap-notice",
        "本条问题超出知识库覆盖范围（知识库只收录通用规则，没有英雄与使用率数据）。" +
        "以上为通用理解，仅供参考，本次未引用任何知识库来源。"
      ));
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

    fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: text, history: history })
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
    fetch("/api/logs?limit=60")
      .then(function (r) { return r.json(); })
      .then(function (d) {
        if (!d.enabled) {
          drawerMeta.textContent = "使用记录已关闭（LOG_ENABLED=0）";
          drawerBody.appendChild(el("div", "log-empty", "后端已关闭使用记录写入。"));
          return;
        }
        var records = d.records || [];
        drawerMeta.textContent = records.length + " 条 · " + d.log_dir;
        if (!records.length) {
          drawerBody.appendChild(el("div", "log-empty", "暂无记录。发几条问题后再来看。"));
          return;
        }
        records.forEach(function (r) { drawerBody.appendChild(renderLogItem(r)); });
      })
      .catch(function () {
        drawerMeta.textContent = "读取失败";
        drawerBody.appendChild(el("div", "log-empty", "无法读取使用记录，请确认服务仍在运行。"));
      });
  }

  function closeDrawer() { drawer.hidden = true; }

  if (btnLogs) btnLogs.addEventListener("click", openDrawer);
  if (btnDrawerClose) btnDrawerClose.addEventListener("click", closeDrawer);
  if (drawer) {
    drawer.addEventListener("click", function (e) { if (e.target === drawer) closeDrawer(); });
  }
  document.addEventListener("keydown", function (e) { if (e.key === "Escape") closeDrawer(); });

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
