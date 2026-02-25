(function () {
  "use strict";

  const config = window.APP_CONFIG || {};
  const BRANCHES = Array.isArray(config.branches) ? config.branches : [];
  const LABEL_CONFIG =
    typeof config.labelConfig === "string"
      ? config.labelConfig
      : [
          "<View>",
          '  <Labels name="label" toName="text">',
          '    <Label value="TERM" background="#c49b2c"/>',
          "  </Labels>",
          '  <Text name="text" value="$text"/>',
          "</View>",
        ].join("\n");

  const branchInput = document.getElementById("branch-input");
  const branchList = document.getElementById("branch-list");
  const textInput = document.getElementById("text-input");
  const predictButton = document.getElementById("predict-btn");
  const errorBox = document.getElementById("error-msg");
  const charCount = document.getElementById("char-count");
  const requestInfo = document.getElementById("request-info");

  const emptyState = document.getElementById("empty-state");
  const resultContent = document.getElementById("result-content");
  const resultText = document.getElementById("result-text");
  const termTableBody = document.getElementById("term-table-body");

  /* ── About modal toggle ─────────────────────── */
  const aboutToggle = document.getElementById("about-toggle");
  const aboutOverlay = document.getElementById("about-overlay");
  const aboutClose = document.getElementById("about-close");

  if (aboutToggle && aboutOverlay) {
    function openAbout() {
      aboutOverlay.hidden = false;
      requestAnimationFrame(() => { aboutOverlay.classList.add("is-open"); });
      aboutToggle.setAttribute("aria-expanded", "true");
    }
    function closeAbout() {
      aboutOverlay.classList.remove("is-open");
      aboutToggle.setAttribute("aria-expanded", "false");
      setTimeout(() => { aboutOverlay.hidden = true; }, 300);
    }
    aboutToggle.addEventListener("click", openAbout);
    if (aboutClose) aboutClose.addEventListener("click", closeAbout);
    aboutOverlay.addEventListener("click", function (e) {
      if (e.target === aboutOverlay) closeAbout();
    });
    document.addEventListener("keydown", function (e) {
      if (e.key === "Escape" && !aboutOverlay.hidden) closeAbout();
    });
  }

  const statCount = document.getElementById("stat-count");
  const statScore = document.getElementById("stat-score");
  const statCoverage = document.getElementById("stat-coverage");

  const state = {
    filtered: [],
    activeIndex: -1,
  };

  let _twGeneration = 0;

  const TR_MAP = {
    "\u0131": "i",
    "\u0130": "i",
    "\u00fc": "u",
    "\u00dc": "u",
    "\u00f6": "o",
    "\u00d6": "o",
    "\u015f": "s",
    "\u015e": "s",
    "\u00e7": "c",
    "\u00c7": "c",
    "\u011f": "g",
    "\u011e": "g",
    "\u00e2": "a",
    "\u00c2": "a",
  };

  const TR_PATTERN = /[\u0131\u0130\u00fc\u00dc\u00f6\u00d6\u015f\u015e\u00e7\u00c7\u011f\u011e\u00e2\u00c2]/g;

  function normalizeTurkish(value) {
    return value.replace(TR_PATTERN, (char) => TR_MAP[char] || char).toLowerCase();
  }

  function escapeHtml(value) {
    const div = document.createElement("div");
    div.textContent = value;
    return div.innerHTML;
  }

  function formatPercent(value) {
    return `${(Math.max(0, value) * 100).toFixed(1)}%`;
  }

  function updateCharCount() {
    const length = textInput.value.length;
    charCount.textContent = `${length} karakter`;
  }

  function showError(message) {
    errorBox.textContent = message;
    errorBox.classList.add("is-visible");
  }

  function clearError() {
    errorBox.textContent = "";
    errorBox.classList.remove("is-visible");
  }

  function setLoading(isLoading) {
    predictButton.disabled = isLoading;
    predictButton.classList.toggle("is-loading", isLoading);
  }

  function setRequestInfo(message) {
    requestInfo.textContent = message;
  }

  function filterBranches(query) {
    const normalizedQuery = normalizeTurkish(query.trim());
    if (!normalizedQuery) {
      return [];
    }

    return BRANCHES.filter((branch) => normalizeTurkish(branch).includes(normalizedQuery)).slice(0, 10);
  }

  function highlightMatch(text, query) {
    if (!query) {
      return escapeHtml(text);
    }

    const normalizedText = normalizeTurkish(text);
    const matchStart = normalizedText.indexOf(query);
    if (matchStart === -1) {
      return escapeHtml(text);
    }

    const before = text.slice(0, matchStart);
    const middle = text.slice(matchStart, matchStart + query.length);
    const after = text.slice(matchStart + query.length);

    return `${escapeHtml(before)}<strong>${escapeHtml(middle)}</strong>${escapeHtml(after)}`;
  }

  function closeTypeahead() {
    branchList.classList.remove("is-open");
    branchInput.setAttribute("aria-expanded", "false");
    branchInput.removeAttribute("aria-activedescendant");
    state.activeIndex = -1;
  }

  function renderTypeahead(items, rawQuery) {
    branchList.innerHTML = "";
    state.filtered = items;

    if (items.length === 0) {
      closeTypeahead();
      return;
    }

    const normalizedQuery = normalizeTurkish(rawQuery.trim());

    items.forEach((item, index) => {
      const option = document.createElement("div");
      option.className = "typeahead-item";
      option.id = `branch-option-${index}`;
      option.setAttribute("role", "option");
      option.setAttribute("data-value", item);
      option.innerHTML = highlightMatch(item, normalizedQuery);

      option.addEventListener("mousedown", (event) => {
        event.preventDefault();
        selectBranch(index);
      });

      branchList.appendChild(option);
    });

    state.activeIndex = -1;
    branchList.classList.add("is-open");
    branchInput.setAttribute("aria-expanded", "true");
  }

  function setActiveTypeahead(index) {
    const options = branchList.querySelectorAll(".typeahead-item");
    options.forEach((option, optionIndex) => {
      option.classList.toggle("is-active", optionIndex === index);
    });

    if (index >= 0 && options[index]) {
      branchInput.setAttribute("aria-activedescendant", options[index].id);
      options[index].scrollIntoView({ block: "nearest" });
    } else {
      branchInput.removeAttribute("aria-activedescendant");
    }
  }

  function selectBranch(index) {
    const branch = state.filtered[index];
    if (!branch) {
      return;
    }

    branchInput.value = branch;
    closeTypeahead();
    textInput.focus();
  }

  function updateTypeaheadFromInput() {
    const query = branchInput.value;
    const matches = filterBranches(query);
    renderTypeahead(matches, query);
  }

  function sanitizeEntities(entities, originalText) {
    const sorted = entities
      .map((entity) => ({
        start: Number(entity.value && entity.value.start),
        end: Number(entity.value && entity.value.end),
        score: Number(entity.score) || 0,
      }))
      .filter((entity) => Number.isFinite(entity.start) && Number.isFinite(entity.end))
      .sort((a, b) => a.start - b.start || a.end - b.end);

    const result = [];
    let cursor = 0;

    sorted.forEach((entity) => {
      const boundedStart = Math.max(0, entity.start);
      const boundedEnd = Math.min(originalText.length, entity.end);

      if (boundedEnd <= boundedStart || boundedStart < cursor) {
        return;
      }

      result.push({
        start: boundedStart,
        end: boundedEnd,
        score: entity.score,
      });
      cursor = boundedEnd;
    });

    return result;
  }

  function renderTextHighlights(text, entities) {
    if (!entities.length) {
      resultText.textContent = text;
      return;
    }

    let cursor = 0;
    const chunks = [];

    entities.forEach((entity) => {
      if (entity.start > cursor) {
        chunks.push(escapeHtml(text.slice(cursor, entity.start)));
      }

      const term = text.slice(entity.start, entity.end);
      chunks.push(
        `<mark data-score="Skor: ${formatPercent(entity.score)}">${escapeHtml(term)}</mark>`
      );
      cursor = entity.end;
    });

    if (cursor < text.length) {
      chunks.push(escapeHtml(text.slice(cursor)));
    }

    resultText.innerHTML = chunks.join("");
  }

  function renderTable(text, entities) {
    if (!entities.length) {
      termTableBody.innerHTML =
        '<tr><td colspan="4">Bu metin için terim tespit edilemedi.</td></tr>';
      return;
    }

    const rows = entities.map((entity, index) => {
      const term = escapeHtml(text.slice(entity.start, entity.end));
      return `<tr>
        <td>${index + 1}</td>
        <td>${term}</td>
        <td>${entity.start}-${entity.end}</td>
        <td>${formatPercent(entity.score)}</td>
      </tr>`;
    });

    termTableBody.innerHTML = rows.join("");
  }

  function renderStats(text, entities, fallbackScore) {
    const totalChars = entities.reduce((sum, entity) => sum + (entity.end - entity.start), 0);
    const averageScore = entities.length
      ? entities.reduce((sum, entity) => sum + entity.score, 0) / entities.length
      : fallbackScore;

    statCount.textContent = String(entities.length);
    statScore.textContent = formatPercent(averageScore || 0);
    statCoverage.textContent = formatPercent(text.length ? totalChars / text.length : 0);
  }

  // ── Typewriter animation (anime.js) ──────────────

  function splitWords(str) {
    return str.match(/\s*\S+\s*/g) || [];
  }

  function buildTypewriterDOM(text, entities) {
    const fragment = document.createDocumentFragment();
    const allWords = [];
    let cursor = 0;

    function addPlain(str) {
      if (!str) return;
      const words = splitWords(str);
      if (!words.length) {
        fragment.appendChild(document.createTextNode(str));
        return;
      }
      for (const word of words) {
        const span = document.createElement("span");
        span.className = "tw";
        span.textContent = word;
        fragment.appendChild(span);
        allWords.push({ el: span, type: "plain" });
      }
    }

    for (const entity of entities) {
      if (entity.start > cursor) addPlain(text.slice(cursor, entity.start));

      const mark = document.createElement("mark");
      mark.className = "tw-entity";
      mark.setAttribute("data-score", `Skor: ${formatPercent(entity.score)}`);

      const words = splitWords(text.slice(entity.start, entity.end));
      if (!words.length) {
        mark.textContent = text.slice(entity.start, entity.end);
      } else {
        words.forEach((word, i) => {
          const span = document.createElement("span");
          span.className = "tw";
          span.textContent = word;
          mark.appendChild(span);
          allWords.push({ el: span, type: "entity", mark, first: i === 0 });
        });
      }

      fragment.appendChild(mark);
      cursor = entity.end;
    }

    if (cursor < text.length) addPlain(text.slice(cursor));

    return { fragment, allWords };
  }

  function animateTypewriter(allWords, gen, onComplete) {
    const count = allWords.length;
    if (!count) { onComplete(); return; }

    // Dynamic per-word delay: cap total typing at ~5.5s
    const delay = Math.max(12, Math.min(65, Math.round(5500 / count)));

    allWords.forEach(w => { w.el.style.opacity = "0"; });

    let lastIdx = -1;

    anime({
      targets: allWords.map(w => w.el),
      opacity: [0, 1],
      easing: "easeOutQuad",
      duration: 250,
      delay: anime.stagger(delay),
      update(anim) {
        if (gen !== _twGeneration) return;
        const idx = Math.min(Math.floor((anim.progress / 100) * count), count - 1);
        if (idx === lastIdx) return;

        if (lastIdx >= 0) allWords[lastIdx].el.classList.remove("tw-current");
        allWords[idx].el.classList.add("tw-current");

        const w = allWords[idx];
        if (w.type === "entity" && w.first) w.mark.classList.add("tw-painted");

        lastIdx = idx;
      },
      complete() {
        if (lastIdx >= 0) allWords[lastIdx].el.classList.remove("tw-current");
        if (gen === _twGeneration) onComplete();
      },
    });
  }

  function animateStatsIn(termCount, avgScore, coverageVal) {
    anime({
      targets: ".stat-card",
      opacity: [0, 1],
      translateY: [16, 0],
      easing: "easeOutQuad",
      duration: 420,
      delay: anime.stagger(70),
    });

    const obj = { c: 0, s: 0, v: 0 };
    anime({
      targets: obj,
      c: termCount,
      s: avgScore * 100,
      v: coverageVal * 100,
      easing: "easeOutExpo",
      duration: 800,
      round: 10,
      update() {
        statCount.textContent = String(Math.round(obj.c));
        statScore.textContent = `${obj.s.toFixed(1)}%`;
        statCoverage.textContent = `${obj.v.toFixed(1)}%`;
      },
    });
  }

  function animateTableIn() {
    anime({
      targets: "#term-table-body tr",
      opacity: [0, 1],
      translateX: [-20, 0],
      easing: "easeOutQuad",
      duration: 280,
      delay: anime.stagger(30, { start: 120 }),
    });
  }

  function renderResults(text, responsePayload) {
    const firstResult =
      responsePayload &&
      Array.isArray(responsePayload.results) &&
      responsePayload.results.length > 0
        ? responsePayload.results[0]
        : {};

    const rawEntities = Array.isArray(firstResult.result) ? firstResult.result : [];
    const entities = sanitizeEntities(rawEntities, text);

    emptyState.style.display = "none";
    resultContent.classList.add("is-visible");

    // Fallback: anime.js yüklenemezse anında göster
    if (typeof anime === "undefined") {
      renderTextHighlights(text, entities);
      renderTable(text, entities);
      renderStats(text, entities, Number(firstResult.score) || 0);
      return;
    }

    // ── Animated path ──────────────────────────────
    _twGeneration++;
    const gen = _twGeneration;

    // Reset stats
    statCount.textContent = "\u2014";
    statScore.textContent = "\u2014";
    statCoverage.textContent = "\u2014";

    // Hide stats & table until typing completes
    const statsGrid = document.getElementById("stats-grid");
    const tableWrap = document.querySelector(".table-wrap");
    statsGrid.style.opacity = "0";
    tableWrap.style.opacity = "0";

    // Build typewriter DOM
    const { fragment, allWords } = buildTypewriterDOM(text, entities);
    resultText.innerHTML = "";
    resultText.appendChild(fragment);

    // Pre-render table (hidden)
    renderTable(text, entities);

    // Compute stats for later animation
    const totalChars = entities.reduce((s, e) => s + (e.end - e.start), 0);
    const avgScore = entities.length
      ? entities.reduce((s, e) => s + e.score, 0) / entities.length
      : Number(firstResult.score) || 0;
    const coverageVal = text.length ? totalChars / text.length : 0;

    // Start typewriter → then stats & table
    animateTypewriter(allWords, gen, () => {
      statsGrid.style.opacity = "1";
      tableWrap.style.opacity = "1";
      animateStatsIn(entities.length, avgScore, coverageVal);
      animateTableIn();
    });
  }

  async function runPrediction() {
    clearError();
    closeTypeahead();

    const text = textInput.value;
    const branch = branchInput.value.trim();

    if (!branch) {
      showError("Lütfen bir alan/branş seçiniz.");
      branchInput.focus();
      return;
    }

    if (!text.trim()) {
      showError("Lütfen analiz edilecek bir metin giriniz.");
      textInput.focus();
      return;
    }

    setLoading(true);
    setRequestInfo("");

    const startTime = performance.now();

    try {
      const response = await fetch("/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          tasks: [{ data: { text, branch } }],
          label_config: LABEL_CONFIG,
        }),
      });

      const payload = await response.json();

      if (!response.ok) {
        const message = payload && payload.error ? payload.error : `Sunucu hatası: ${response.status}`;
        throw new Error(message);
      }

      renderResults(text, payload);

      const elapsedMs = Math.round(performance.now() - startTime);
      setRequestInfo(`Son işlem süresi: ${elapsedMs} ms`);
    } catch (error) {
      showError(error.message || "Beklenmeyen bir hata oluştu.");
    } finally {
      setLoading(false);
    }
  }

  predictButton.addEventListener("click", runPrediction);

  textInput.addEventListener("input", updateCharCount);

  textInput.addEventListener("keydown", (event) => {
    if ((event.ctrlKey || event.metaKey) && event.key === "Enter") {
      event.preventDefault();
      runPrediction();
    }
  });

  branchInput.addEventListener("input", updateTypeaheadFromInput);

  branchInput.addEventListener("focus", () => {
    if (branchInput.value.trim()) {
      updateTypeaheadFromInput();
    }
  });

  branchInput.addEventListener("keydown", (event) => {
    const options = branchList.querySelectorAll(".typeahead-item");

    if (event.key === "Escape") {
      closeTypeahead();
      return;
    }

    if (event.key === "Enter" && !branchList.classList.contains("is-open")) {
      return;
    }

    if (!options.length && event.key !== "Enter") {
      return;
    }

    if (event.key === "ArrowDown") {
      event.preventDefault();
      state.activeIndex = Math.min(state.activeIndex + 1, options.length - 1);
      setActiveTypeahead(state.activeIndex);
      return;
    }

    if (event.key === "ArrowUp") {
      event.preventDefault();
      state.activeIndex = Math.max(state.activeIndex - 1, 0);
      setActiveTypeahead(state.activeIndex);
      return;
    }

    if (event.key === "Enter") {
      event.preventDefault();
      if (state.activeIndex >= 0) {
        selectBranch(state.activeIndex);
      } else {
        closeTypeahead();
      }
    }
  });

  document.addEventListener("click", (event) => {
    if (!event.target.closest(".autocomplete")) {
      closeTypeahead();
    }
  });

  updateCharCount();
})();
