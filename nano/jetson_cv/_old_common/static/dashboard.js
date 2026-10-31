// Jetson Nano CV dashboard — vanilla JS, zero dipendenze esterne.
(function () {
  "use strict";

  var STATS_POLL_MS = 700;
  var SLIDER_THROTTLE_MS = 180;

  var COLORS = {
    good: "#0ca30c", warning: "#fab219", serious: "#ec835a", critical: "#d03b3b",
    accent: "#3987e5", grid: "#2c2c2a", ink: "#ffffff", ink2: "#c3c2b7",
    muted: "#898781", baseline: "#383835",
  };

  var FILTER_HINTS = {
    none: "Nessuna trasformazione oltre a luminosità/contrasto: la scelta " +
      "migliore per la detection in condizioni normali.",
    gray: "Converte in scala di grigi. YOLOv5 è stato addestrato su " +
      "immagini a colori: aspettati un calo di precisione, specialmente su " +
      "classi che si distinguono anche per colore.",
    clahe: "Equalizzazione adattiva del contrasto locale (CLAHE): schiarisce " +
      "le zone in ombra senza bruciare quelle già chiare. Utile con " +
      "controluce forte o luce molto disomogenea; può introdurre rumore " +
      "visibile in zone uniformi.",
    sharpen: "Aumenta la nitidezza percepita (unsharp mask). Può aiutare su " +
      "immagini leggermente sfocate, ma amplifica anche il rumore del " +
      "sensore e può generare falsi bordi che confondono il rilevatore.",
    blur: "Sfocatura gaussiana: utile solo a scopo didattico per mostrare " +
      "quanto la detection dipenda dai dettagli ad alta frequenza. " +
      "Aspettati un netto calo di box rilevati e di confidenza.",
    edge: "Mostra solo i contorni (Canny), scartando colore e texture. Il " +
      "rilevatore quasi certamente fallisce: pensato per capire cosa " +
      "“vede” un edge detector classico, non per l'uso normale.",
  };

  var lastStats = null;
  var applyingModel = false;

  // ---------------- Tabs ----------------
  var tabButtons = document.querySelectorAll("nav.tabs button");
  tabButtons.forEach(function (btn) {
    btn.addEventListener("click", function () {
      tabButtons.forEach(function (b) { b.classList.remove("active"); });
      document.querySelectorAll(".tab-panel").forEach(function (p) {
        p.classList.remove("active");
      });
      btn.classList.add("active");
      document.getElementById("tab-" + btn.dataset.tab).classList.add("active");
    });
  });

  // ---------------- Helpers HTTP ----------------
  function postJson(url, body) {
    return fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body || {}),
    }).then(function (r) {
      if (!r.ok) return r.json().then(function (e) { throw new Error(e.error || r.statusText); });
      return r.json();
    });
  }

  function postParams(patch) { return postJson("/api/params", patch); }

  function throttle(fn, ms) {
    var t = null, pendingArgs = null;
    return function () {
      pendingArgs = arguments;
      if (t) return;
      t = setTimeout(function () {
        t = null;
        fn.apply(null, pendingArgs);
      }, ms);
    };
  }

  // ---------------- Statistiche: gauge generiche ----------------
  function statusColor(value, thresholds) {
    // thresholds: [{upTo, color}], ascendente; l'ultimo copre il resto
    for (var i = 0; i < thresholds.length; i++) {
      if (value < thresholds[i].upTo) return thresholds[i].color;
    }
    return thresholds[thresholds.length - 1].color;
  }

  var PCT_THRESHOLDS = [
    { upTo: 60, color: COLORS.good },
    { upTo: 80, color: COLORS.warning },
    { upTo: 90, color: COLORS.serious },
    { upTo: 101, color: COLORS.critical },
  ];
  var TEMP_THRESHOLDS = [
    { upTo: 50, color: COLORS.good },
    { upTo: 70, color: COLORS.warning },
    { upTo: 85, color: COLORS.serious },
    { upTo: 999, color: COLORS.critical },
  ];

  function drawGauge(canvas, value, max, color, valueText) {
    var ctx = canvas.getContext("2d");
    var w = canvas.width, h = canvas.height;
    var cx = w / 2, cy = h - 18, r = Math.min(w, h - 30) / 2 - 6;
    ctx.clearRect(0, 0, w, h);

    var pct = Math.max(0, Math.min(1, value / max));

    ctx.lineWidth = 10;
    ctx.lineCap = "round";

    // track
    ctx.beginPath();
    ctx.arc(cx, cy, r, Math.PI, 2 * Math.PI);
    ctx.strokeStyle = COLORS.grid;
    ctx.stroke();

    // valore
    if (pct > 0) {
      ctx.beginPath();
      ctx.arc(cx, cy, r, Math.PI, Math.PI + Math.PI * pct);
      ctx.strokeStyle = color;
      ctx.stroke();
    }

    ctx.fillStyle = COLORS.ink;
    ctx.font = "600 17px system-ui, sans-serif";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText(valueText, cx, cy - r * 0.35);
  }

  function ensureGaugeCard(id, label) {
    var row = document.getElementById("gauges-row");
    var existing = document.getElementById("gauge-" + id);
    if (existing) return existing;
    var card = document.createElement("div");
    card.className = "gauge-card";
    card.innerHTML =
      '<canvas id="gauge-' + id + '" width="140" height="100"></canvas>' +
      '<div class="gauge-label">' + label + "</div>";
    row.appendChild(card);
    return document.getElementById("gauge-" + id);
  }

  function updateSistemaTab(sys) {
    if (!sys || sys.jtop_ok === false) {
      var row = document.getElementById("gauges-row");
      if (!row.dataset.warned) {
        row.innerHTML = '<div class="card"><h3>jetson-stats</h3>' +
          '<div class="stat-sub">Non disponibile: ' +
          (sys && sys.error ? sys.error : "servizio jtop non raggiungibile") +
          "</div></div>";
        row.dataset.warned = "1";
      }
      return;
    }
    document.getElementById("gauges-row").dataset.warned = "";

    if (sys.cpu_percent !== null && sys.cpu_percent !== undefined) {
      var c = ensureGaugeCard("cpu", "CPU");
      drawGauge(c, sys.cpu_percent, 100, statusColor(sys.cpu_percent, PCT_THRESHOLDS),
                Math.round(sys.cpu_percent) + "%");
    }
    if (sys.gpu_percent !== null && sys.gpu_percent !== undefined) {
      var g = ensureGaugeCard("gpu", "GPU");
      drawGauge(g, sys.gpu_percent, 100, statusColor(sys.gpu_percent, PCT_THRESHOLDS),
                Math.round(sys.gpu_percent) + "%");
    }
    var temps = sys.temps || {};
    Object.keys(temps).sort().forEach(function (name) {
      var t = temps[name];
      var el = ensureGaugeCard("temp-" + name, name.replace("thermal", "SoC"));
      drawGauge(el, t, 100, statusColor(t, TEMP_THRESHOLDS), Math.round(t) + "°");
    });

    document.getElementById("sys-fan").innerHTML =
      (sys.fan_percent !== null ? Math.round(sys.fan_percent) : "--") + '<span class="stat-unit">%</span>';
    document.getElementById("sys-fan-rpm").textContent =
      (sys.fan_rpm !== null && sys.fan_rpm !== undefined ? sys.fan_rpm : "--") + " rpm";
    document.getElementById("sys-powermode").textContent = sys.power_mode || "--";
    document.getElementById("sys-power-mw").textContent =
      (sys.power_mw !== null && sys.power_mw !== undefined ? sys.power_mw : "--") + " mW";

    if (sys.ram_total_mb) {
      var pct = Math.round((sys.ram_used_mb / sys.ram_total_mb) * 100);
      document.getElementById("sys-ram-pct").innerHTML = pct + '<span class="stat-unit">%</span>';
      document.getElementById("sys-ram-mb").textContent =
        Math.round(sys.ram_used_mb) + " / " + Math.round(sys.ram_total_mb) + " MB";
    }

    syncPowerControlsFromSystem(sys);
  }

  // ---------------- Statistiche: grafico (tempo inferenza o fps) ----------------
  var chartSeries = "infer"; // "infer" | "fps"
  var chartHistory = [];

  function selectChartSeries(series) {
    chartSeries = series;
    document.getElementById("btn-chart-infer").classList.toggle("active", series === "infer");
    document.getElementById("btn-chart-fps").classList.toggle("active", series === "fps");
    if (lastStats) {
      chartHistory = series === "infer" ? lastStats.infer_ms.history : lastStats.fps_history;
      drawInferChart();
    }
  }
  document.getElementById("btn-chart-infer").addEventListener("click", function () { selectChartSeries("infer"); });
  document.getElementById("btn-chart-fps").addEventListener("click", function () { selectChartSeries("fps"); });

  function drawInferChart() {
    var canvas = document.getElementById("infer-chart");
    var ctx = canvas.getContext("2d");
    var w = canvas.width, h = canvas.height;
    var padL = 44, padR = 12, padT = 14, padB = 22;
    ctx.clearRect(0, 0, w, h);

    var hist = chartHistory;
    var unit = chartSeries === "fps" ? " fps" : " ms";
    if (!hist.length) return;
    var maxV = Math.max.apply(null, hist) * 1.15 || 1;
    var minV = 0;
    var plotW = w - padL - padR, plotH = h - padT - padB;

    function xAt(i) { return padL + (i / Math.max(1, hist.length - 1)) * plotW; }
    function yAt(v) { return padT + plotH - ((v - minV) / (maxV - minV)) * plotH; }

    // gridlines + etichette asse Y (3 righe)
    ctx.strokeStyle = COLORS.grid;
    ctx.fillStyle = COLORS.muted;
    ctx.font = "11px system-ui, sans-serif";
    ctx.textAlign = "right";
    ctx.textBaseline = "middle";
    ctx.lineWidth = 1;
    for (var g = 0; g <= 2; g++) {
      var v = (maxV / 2) * g;
      var y = yAt(v);
      ctx.beginPath();
      ctx.moveTo(padL, y);
      ctx.lineTo(w - padR, y);
      ctx.stroke();
      ctx.fillText(Math.round(v), padL - 8, y);
    }

    // baseline
    ctx.strokeStyle = COLORS.baseline;
    ctx.beginPath();
    ctx.moveTo(padL, padT + plotH);
    ctx.lineTo(w - padR, padT + plotH);
    ctx.stroke();

    // linea (2px, singola serie: niente legenda, il titolo la identifica)
    ctx.beginPath();
    hist.forEach(function (v, i) {
      var x = xAt(i), y = yAt(v);
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    });
    ctx.strokeStyle = COLORS.accent;
    ctx.lineWidth = 2;
    ctx.lineJoin = "round";
    ctx.stroke();

    // etichetta diretta sul valore corrente (fine linea)
    var lastX = xAt(hist.length - 1), lastY = yAt(hist[hist.length - 1]);
    ctx.beginPath();
    ctx.arc(lastX, lastY, 3, 0, 2 * Math.PI);
    ctx.fillStyle = COLORS.accent;
    ctx.fill();

    canvas._chartMeta = { hist: hist, xAt: xAt, yAt: yAt, padL: padL, padR: padR, unit: unit };
  }

  function setupChartHover() {
    var canvas = document.getElementById("infer-chart");
    var tooltip = document.getElementById("chart-tooltip");
    canvas.addEventListener("mousemove", function (ev) {
      var meta = canvas._chartMeta;
      if (!meta || !meta.hist.length) return;
      var rect = canvas.getBoundingClientRect();
      var scaleX = canvas.width / rect.width;
      var mx = (ev.clientX - rect.left) * scaleX;
      var n = meta.hist.length;
      var idx = Math.round(((mx - meta.padL) / (canvas.width - meta.padL - meta.padR)) * (n - 1));
      idx = Math.max(0, Math.min(n - 1, idx));
      var val = meta.hist[idx];
      tooltip.style.display = "block";
      tooltip.style.left = (ev.clientX - rect.left + 12) + "px";
      tooltip.style.top = (ev.clientY - rect.top - 10) + "px";
      tooltip.textContent = val.toFixed(1) + meta.unit;
    });
    canvas.addEventListener("mouseleave", function () {
      tooltip.style.display = "none";
    });
  }

  // ---------------- Tabelle conteggi ----------------
  function fillCountTable(tableId, counts) {
    var table = document.getElementById(tableId);
    while (table.rows.length > 1) table.deleteRow(1);
    var entries = Object.keys(counts || {}).map(function (k) { return [k, counts[k]]; });
    entries.sort(function (a, b) { return b[1] - a[1]; });
    if (!entries.length) {
      var row = table.insertRow();
      row.innerHTML = '<td colspan="2" style="color:var(--ink-muted)">nessuna detection</td>';
      return;
    }
    entries.forEach(function (e) {
      var row = table.insertRow();
      row.innerHTML = "<td>" + e[0] + '</td><td class="num">' + e[1] + "</td>";
    });
  }

  // ---------------- Barra di stato ROI (evidenza enter/exit vicino al video) ----------------
  var lastRoiCounts = null;

  function pulse(el, cls) {
    el.classList.remove("pulse-enter", "pulse-exit");
    void el.offsetWidth; // forza il reflow: riavvia la transizione anche se già in corso
    el.classList.add(cls);
    setTimeout(function () { el.classList.remove(cls); }, 900);
  }

  function updateRoiStatusBar(roi, serverTime) {
    var insideEl = document.getElementById("roi-bar-inside");
    var enterEl = document.getElementById("roi-bar-enter");
    var exitEl = document.getElementById("roi-bar-exit");
    insideEl.textContent = roi.inside_now;
    enterEl.textContent = roi.enter_total;
    exitEl.textContent = roi.exit_total;

    if (lastRoiCounts) {
      if (roi.enter_total > lastRoiCounts.enter_total) pulse(enterEl, "pulse-enter");
      if (roi.exit_total > lastRoiCounts.exit_total) pulse(exitEl, "pulse-exit");
    }
    lastRoiCounts = { enter_total: roi.enter_total, exit_total: roi.exit_total };

    var logEl = document.getElementById("roi-log");
    var entries = roi.log || [];
    if (!entries.length) {
      logEl.innerHTML = '<li class="roi-log-empty">Nessun evento ancora.</li>';
      return;
    }
    logEl.innerHTML = "";
    entries.forEach(function (e) {
      var ago = Math.max(0, Math.round(serverTime - e.t));
      var isEnter = e.type === "enter";
      var li = document.createElement("li");
      li.innerHTML =
        '<span class="' + (isEnter ? "evt-enter" : "evt-exit") + '">' +
        (isEnter ? "→ entrato: " : "← uscito: ") + e.label + "</span>" +
        '<span class="evt-time">' + ago + "s fa</span>";
      logEl.appendChild(li);
    });
  }

  // ---------------- Poll /api/stats ----------------
  var slidersTouched = false; // evita di sovrascrivere uno slider mentre l'utente lo trascina

  function applyStats(s) {
    lastStats = s;
    document.getElementById("conn-dot").className = "ok";
    document.getElementById("hdr-fps").textContent = s.fps.toFixed(1);
    document.getElementById("hdr-model").textContent =
      "modello: " + (s.model.name || "--") +
      (s.model.switching ? " (cambio in corso…)" : "");

    var modelLabel = (s.model.name || "--") + (s.model.switching ? " (cambio in corso…)" : "");
    document.getElementById("video-model-chip").textContent = modelLabel;
    document.getElementById("model-active-name").textContent = modelLabel;
    document.getElementById("model-active-meta").textContent = s.model.name
      ? "decoder " + (s.model.postprocess || "--") + " · input " +
        s.model.input_size + "×" + s.model.input_size
      : "--";
    document.getElementById("video-fps-chip").textContent = s.fps.toFixed(1) + " fps";
    document.getElementById("video-infer-chip").textContent =
      "inf " + s.infer_ms.current.toFixed(1) + " ms";

    updateSistemaTab(s.system);

    document.getElementById("sys-uptime").textContent = formatUptime(s.uptime_s);
    document.getElementById("sys-frames").textContent = s.frame_count + " frame";

    chartHistory = chartSeries === "infer" ? s.infer_ms.history : s.fps_history;
    drawInferChart();
    document.getElementById("st-current").textContent = s.infer_ms.current.toFixed(1);
    document.getElementById("st-mean").textContent = s.infer_ms.mean.toFixed(1);
    document.getElementById("st-p50").textContent = s.infer_ms.p50.toFixed(1);
    document.getElementById("st-p95").textContent = s.infer_ms.p95.toFixed(1);
    document.getElementById("st-min").textContent = s.infer_ms.min.toFixed(1);
    document.getElementById("st-max").textContent = s.infer_ms.max.toFixed(1);
    document.getElementById("st-std").textContent = s.infer_ms.std.toFixed(1);
    document.getElementById("st-mean-conf").textContent =
      (s.detections.mean_conf_current * 100).toFixed(1) + "%";

    fillCountTable("table-frame", s.detections.frame);
    fillCountTable("table-cumulative", s.detections.cumulative_unique);

    document.getElementById("roi-inside").textContent = s.roi.inside_now;
    document.getElementById("roi-enter").textContent = s.roi.enter_total;
    document.getElementById("roi-exit").textContent = s.roi.exit_total;
    document.getElementById("roi-peak").textContent = s.roi.inside_peak;
    document.getElementById("tracker-active").textContent = s.tracker.active_tracks;
    updateRoiStatusBar(s.roi, s.server_time);

    if (!slidersTouched) syncControlsFromParams(s.params);
    redrawRoiOverlay(s.params.roi);

    if (!applyingModel) {
      document.getElementById("model-status").textContent = "";
    }
  }

  function formatUptime(sec) {
    sec = Math.floor(sec);
    var h = Math.floor(sec / 3600), m = Math.floor((sec % 3600) / 60), s = sec % 60;
    return (h ? h + "h " : "") + m + "m " + s + "s";
  }

  var controlsInitialized = false;
  var lastKnownClassFilter = null;

  function syncControlsFromParams(p) {
    if (controlsInitialized) return; // dopo il primo giro l'utente guida i controlli
    controlsInitialized = true;
    setSlider("ctl-brightness", "out-brightness", p.brightness, 0);
    setSlider("ctl-contrast", "out-contrast", p.contrast, 2);
    setSlider("ctl-conf", "out-conf", p.conf, 2);
    setSlider("ctl-iou", "out-iou", p.iou, 2);
    setSlider("ctl-stride", "out-stride", p.infer_stride, 0);
    setSlider("ctl-jpeg", "out-jpeg", p.jpeg_quality, 0);
    document.getElementById("ctl-filter").value = p.filter;
    document.getElementById("filter-hint").textContent = FILTER_HINTS[p.filter] || "";
    document.getElementById("ctl-trajectories").checked = !!p.show_trajectories;
    document.getElementById("ctl-count-classes").checked = !!p.count_classes;
    document.getElementById("ctl-roi-enabled").checked = !!p.roi.enabled;
    lastKnownClassFilter = p.class_filter || [];
    applyClassFilterSelection();
    setDetectionToggle(!!p.detection_enabled);
  }

  function setDetectionToggle(enabled) {
    var btn = document.getElementById("btn-detection-toggle");
    btn.dataset.enabled = enabled ? "true" : "false";
    document.getElementById("detection-toggle-label").textContent =
      enabled ? "Detection attiva" : "Detection ferma";
  }

  document.getElementById("btn-detection-toggle").addEventListener("click", function () {
    var btn = document.getElementById("btn-detection-toggle");
    var next = btn.dataset.enabled !== "true";
    setDetectionToggle(next); // ottimistico, corretto al prossimo poll se il server rifiuta
    postParams({ detection_enabled: next })
      .catch(function (e) { console.error("detection_enabled:", e.message); });
  });

  function setSlider(inputId, outId, value, decimals) {
    document.getElementById(inputId).value = value;
    document.getElementById(outId).textContent = Number(value).toFixed(decimals);
  }

  function pollStats() {
    fetch("/api/stats").then(function (r) { return r.json(); }).then(applyStats)
      .catch(function () {
        document.getElementById("conn-dot").className = "bad";
      });
  }
  setInterval(pollStats, STATS_POLL_MS);
  pollStats();

  // ---------------- Controlli Immagine / Rilevamento ----------------
  var sendPatch = throttle(function (patch) {
    postParams(patch).catch(function (e) { console.error("params:", e.message); });
  }, SLIDER_THROTTLE_MS);

  function bindSlider(inputId, outId, key, decimals, isFloat) {
    var input = document.getElementById(inputId);
    var out = document.getElementById(outId);
    input.addEventListener("input", function () {
      slidersTouched = true;
      var v = isFloat ? parseFloat(input.value) : parseInt(input.value, 10);
      out.textContent = v.toFixed(decimals);
      var patch = {};
      patch[key] = v;
      sendPatch(patch);
    });
    input.addEventListener("mouseup", function () { setTimeout(function () { slidersTouched = false; }, 400); });
  }

  bindSlider("ctl-brightness", "out-brightness", "brightness", 0, true);
  bindSlider("ctl-contrast", "out-contrast", "contrast", 2, true);
  bindSlider("ctl-conf", "out-conf", "conf", 2, true);
  bindSlider("ctl-iou", "out-iou", "iou", 2, true);
  bindSlider("ctl-stride", "out-stride", "infer_stride", 0, false);
  bindSlider("ctl-jpeg", "out-jpeg", "jpeg_quality", 0, false);

  document.getElementById("ctl-filter").addEventListener("change", function (e) {
    postParams({ filter: e.target.value }).catch(function (err) { console.error("filter:", err.message); });
    document.getElementById("filter-hint").textContent = FILTER_HINTS[e.target.value] || "";
  });
  document.getElementById("ctl-trajectories").addEventListener("change", function (e) {
    postParams({ show_trajectories: e.target.checked });
  });
  document.getElementById("ctl-count-classes").addEventListener("change", function (e) {
    postParams({ count_classes: e.target.checked });
  });
  document.getElementById("ctl-roi-enabled").addEventListener("change", function (e) {
    postParams({ roi: { enabled: e.target.checked } });
  });

  // ---------------- Filtro classi ----------------
  function applyClassFilterSelection() {
    var sel = document.getElementById("ctl-class-filter");
    if (!sel.options.length || !lastKnownClassFilter) return;
    var wanted = {};
    lastKnownClassFilter.forEach(function (c) { wanted[c] = true; });
    Array.prototype.forEach.call(sel.options, function (opt) {
      opt.selected = !!wanted[opt.value];
    });
  }

  function loadClasses() {
    fetch("/api/classes").then(function (r) { return r.json(); }).then(function (classes) {
      var sel = document.getElementById("ctl-class-filter");
      sel.innerHTML = "";
      classes.forEach(function (name) {
        var opt = document.createElement("option");
        opt.value = name;
        opt.textContent = name;
        sel.appendChild(opt);
      });
      applyClassFilterSelection();
    }).catch(function () { /* dashboard_server non ancora pronto */ });
  }
  loadClasses();

  document.getElementById("ctl-class-filter").addEventListener("change", function (e) {
    var values = Array.prototype.filter.call(e.target.options, function (o) { return o.selected; })
      .map(function (o) { return o.value; });
    postParams({ class_filter: values }).catch(function (err) { console.error("class_filter:", err.message); });
  });

  // ---------------- Modelli ----------------
  function loadModels(selectCurrent) {
    fetch("/api/models").then(function (r) { return r.json(); }).then(function (models) {
      var sel = document.getElementById("ctl-model");
      sel.innerHTML = "";
      models.forEach(function (m) {
        var opt = document.createElement("option");
        opt.value = m.file;
        opt.textContent = m.label + (m.metadata_source === "guessed" ? " (dedotto)" : "");
        if (m.current) opt.selected = true;
        sel.appendChild(opt);
      });
    }).catch(function () { /* dashboard_server non ancora pronto */ });
  }
  loadModels();

  document.getElementById("btn-apply-model").addEventListener("click", function () {
    var file = document.getElementById("ctl-model").value;
    if (!file) return;
    applyingModel = true;
    var statusEl = document.getElementById("model-status");
    statusEl.textContent = "Cambio modello in corso…";
    postJson("/api/model", { file: file }).then(function () {
      var check = setInterval(function () {
        if (!lastStats || !lastStats.model.switching) {
          clearInterval(check);
          applyingModel = false;
          statusEl.textContent = "Modello attivo: " + (lastStats ? lastStats.model.name : file);
          loadModels();
        }
      }, 500);
    }).catch(function (e) {
      applyingModel = false;
      statusEl.textContent = "Errore: " + e.message;
    });
  });

  // ---------------- Alimentazione: power mode + ventola ----------------
  var powerControlsInitialized = false;

  function updateFanSpeedRowVisibility() {
    var manual = document.getElementById("ctl-fan-profile").value === "manual";
    document.getElementById("fan-speed-row").classList.toggle("disabled", !manual);
  }

  function syncPowerControlsFromSystem(sys) {
    if (powerControlsInitialized) return;
    if (!sys || !sys.jtop_ok || !sys.power_modes || !sys.power_modes.length) return;
    powerControlsInitialized = true;

    var sel = document.getElementById("ctl-powermode");
    sel.innerHTML = "";
    sys.power_modes.forEach(function (name, idx) {
      var opt = document.createElement("option");
      opt.value = idx;
      opt.textContent = name;
      if (idx === sys.power_mode_id) opt.selected = true;
      sel.appendChild(opt);
    });

    document.getElementById("ctl-fan-profile").value = sys.fan_profile || "temp_control";
    var speed = sys.fan_percent !== null && sys.fan_percent !== undefined ? sys.fan_percent : 100;
    setSlider("ctl-fan-speed", "out-fan-speed", speed, 0);
    document.getElementById("out-fan-speed").textContent = Math.round(speed) + "%";
    updateFanSpeedRowVisibility();
  }

  document.getElementById("ctl-powermode").addEventListener("change", function (e) {
    postJson("/api/power", { mode_id: parseInt(e.target.value, 10) })
      .catch(function (err) { console.error("power:", err.message); });
  });

  document.getElementById("ctl-fan-profile").addEventListener("change", function (e) {
    updateFanSpeedRowVisibility();
    postJson("/api/fan", { profile: e.target.value })
      .catch(function (err) { console.error("fan:", err.message); });
  });

  var sendFanSpeed = throttle(function (speed) {
    postJson("/api/fan", { speed: speed }).catch(function (err) { console.error("fan:", err.message); });
  }, SLIDER_THROTTLE_MS);

  document.getElementById("ctl-fan-speed").addEventListener("input", function (e) {
    var v = parseInt(e.target.value, 10);
    document.getElementById("out-fan-speed").textContent = v + "%";
    document.getElementById("ctl-fan-profile").value = "manual"; // set_speed forza sempre manual lato server
    updateFanSpeedRowVisibility();
    sendFanSpeed(v);
  });

  document.getElementById("btn-max-performance").addEventListener("click", function () {
    postJson("/api/power/max", {}).then(function () {
      var sel = document.getElementById("ctl-powermode");
      for (var i = 0; i < sel.options.length; i++) {
        if (sel.options[i].textContent === "MAXN") { sel.value = sel.options[i].value; break; }
      }
      document.getElementById("ctl-fan-profile").value = "manual";
      setSlider("ctl-fan-speed", "out-fan-speed", 100, 0);
      document.getElementById("out-fan-speed").textContent = "100%";
      updateFanSpeedRowVisibility();
    }).catch(function (e) { console.error("power/max:", e.message); });
  });

  // ---------------- Ventola automatica di sistema (automagic-fan) ----------------
  var fanServiceInitialized = false;

  function syncFanServiceControls(st) {
    var statusEl = document.getElementById("fansvc-status");
    if (!st || st.available === false) {
      statusEl.textContent = "non disponibile" + (st && st.error ? " (" + st.error + ")" : "");
      return;
    }
    statusEl.textContent = st.active ? "attivo" : "fermo";
    if (fanServiceInitialized || !st.config) return;
    fanServiceInitialized = true;
    var c = st.config;
    setSlider("fansvc-off", "fansvc-off-out", c.FAN_OFF_TEMP, 0);
    setSlider("fansvc-max", "fansvc-max-out", c.FAN_MAX_TEMP, 0);
    setSlider("fansvc-interval", "fansvc-interval-out", c.UPDATE_INTERVAL, 0);
    document.getElementById("fansvc-maxperf").checked = !!c.MAX_PERF;
  }

  function pollFanService() {
    fetch("/api/fan-service").then(function (r) { return r.json(); })
      .then(syncFanServiceControls).catch(function () {});
  }
  setInterval(pollFanService, 3000);
  pollFanService();

  ["fansvc-off", "fansvc-max", "fansvc-interval"].forEach(function (id) {
    document.getElementById(id).addEventListener("input", function (e) {
      document.getElementById(id + "-out").textContent = e.target.value;
    });
  });

  document.getElementById("btn-fansvc-start").addEventListener("click", function () {
    postJson("/api/fan-service", { action: "start" }).then(pollFanService)
      .catch(function (e) { document.getElementById("fansvc-status").textContent = "errore: " + e.message; });
  });
  document.getElementById("btn-fansvc-stop").addEventListener("click", function () {
    postJson("/api/fan-service", { action: "stop" }).then(pollFanService)
      .catch(function (e) { document.getElementById("fansvc-status").textContent = "errore: " + e.message; });
  });

  document.getElementById("btn-fansvc-save").addEventListener("click", function () {
    var patch = {
      FAN_OFF_TEMP: parseInt(document.getElementById("fansvc-off").value, 10),
      FAN_MAX_TEMP: parseInt(document.getElementById("fansvc-max").value, 10),
      UPDATE_INTERVAL: parseInt(document.getElementById("fansvc-interval").value, 10),
      MAX_PERF: document.getElementById("fansvc-maxperf").checked ? 1 : 0,
    };
    var statusEl = document.getElementById("fansvc-save-status");
    statusEl.textContent = "Salvataggio…";
    postJson("/api/fan-service/config", patch).then(function (st) {
      statusEl.textContent = "Salvato" +
        (st.active ? " e servizio riavviato per applicarlo." : " (servizio fermo: si applica al prossimo avvio).");
      document.getElementById("fansvc-status").textContent = st.active ? "attivo" : "fermo";
    }).catch(function (e) { statusEl.textContent = "Errore: " + e.message; });
  });

  // ---------------- ROI drag-to-draw ----------------
  var roiCanvas = document.getElementById("roi-canvas");
  var roiCtx = roiCanvas.getContext("2d");
  var streamImg = document.getElementById("stream");
  var dragStart = null, dragCurrent = null;
  var currentRoiParams = null;

  function resizeRoiCanvas() {
    roiCanvas.width = streamImg.clientWidth || 640;
    roiCanvas.height = streamImg.clientHeight || 480;
    redrawRoiOverlay(currentRoiParams);
  }
  window.addEventListener("resize", resizeRoiCanvas);
  streamImg.addEventListener("load", resizeRoiCanvas);
  setTimeout(resizeRoiCanvas, 300);

  function redrawRoiOverlay(roi) {
    if (roi) currentRoiParams = roi;
    roiCtx.clearRect(0, 0, roiCanvas.width, roiCanvas.height);
    if (currentRoiParams && currentRoiParams.enabled && !dragStart) {
      drawRoiRect(currentRoiParams.x1, currentRoiParams.y1, currentRoiParams.x2, currentRoiParams.y2, COLORS.accent);
    }
    if (dragStart && dragCurrent) {
      var x1 = Math.min(dragStart.x, dragCurrent.x) / roiCanvas.width;
      var y1 = Math.min(dragStart.y, dragCurrent.y) / roiCanvas.height;
      var x2 = Math.max(dragStart.x, dragCurrent.x) / roiCanvas.width;
      var y2 = Math.max(dragStart.y, dragCurrent.y) / roiCanvas.height;
      drawRoiRect(x1, y1, x2, y2, COLORS.warning);
    }
  }

  function drawRoiRect(x1, y1, x2, y2, color) {
    var px1 = x1 * roiCanvas.width, py1 = y1 * roiCanvas.height;
    var px2 = x2 * roiCanvas.width, py2 = y2 * roiCanvas.height;
    roiCtx.strokeStyle = color;
    roiCtx.lineWidth = 2;
    roiCtx.setLineDash([6, 4]);
    roiCtx.strokeRect(px1, py1, px2 - px1, py2 - py1);
    roiCtx.setLineDash([]);
  }

  function canvasPos(ev) {
    var rect = roiCanvas.getBoundingClientRect();
    var clientX = ev.touches ? ev.touches[0].clientX : ev.clientX;
    var clientY = ev.touches ? ev.touches[0].clientY : ev.clientY;
    return { x: clientX - rect.left, y: clientY - rect.top };
  }

  roiCanvas.addEventListener("mousedown", function (ev) {
    dragStart = canvasPos(ev);
    dragCurrent = dragStart;
  });
  roiCanvas.addEventListener("mousemove", function (ev) {
    if (!dragStart) return;
    dragCurrent = canvasPos(ev);
    redrawRoiOverlay();
  });
  window.addEventListener("mouseup", function () {
    if (!dragStart || !dragCurrent) { dragStart = null; return; }
    var dx = Math.abs(dragCurrent.x - dragStart.x), dy = Math.abs(dragCurrent.y - dragStart.y);
    if (dx > 8 && dy > 8) {
      var x1 = Math.min(dragStart.x, dragCurrent.x) / roiCanvas.width;
      var y1 = Math.min(dragStart.y, dragCurrent.y) / roiCanvas.height;
      var x2 = Math.max(dragStart.x, dragCurrent.x) / roiCanvas.width;
      var y2 = Math.max(dragStart.y, dragCurrent.y) / roiCanvas.height;
      postParams({ roi: { enabled: true, x1: x1, y1: y1, x2: x2, y2: y2 } })
        .then(function () { document.getElementById("ctl-roi-enabled").checked = true; })
        .catch(function (e) { console.error("roi:", e.message); });
    }
    dragStart = null; dragCurrent = null;
    redrawRoiOverlay();
  });

  document.getElementById("btn-clear-roi").addEventListener("click", function () {
    postParams({ roi: { enabled: false, x1: 0.25, y1: 0.25, x2: 0.75, y2: 0.75 } })
      .then(function () { document.getElementById("ctl-roi-enabled").checked = false; })
      .catch(function (e) { console.error("roi:", e.message); });
  });

  setupChartHover();
})();
