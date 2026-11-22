"use strict";

(() => {
  const get = (id) => document.getElementById(`geometry-${id}`);
  const root = "geometry/v1";
  const state = {catalog: null, condition: null, summary: null, examples: null, sample: 0, position: 0};
  let conditionVersion = 0;
  let matchVersion = 0;
  let shard = {url: null, promise: null};

  function node(tag, text) {
    const element = document.createElement(tag);
    if (text !== undefined) element.textContent = text;
    return element;
  }

  function options(id, entries, preferred = get(id).value) {
    const select = get(id);
    select.replaceChildren(...entries.map(([value, label]) => new Option(label, value)));
    if (entries.some(([value]) => String(value) === String(preferred))) select.value = preferred;
  }

  async function request(path) {
    const response = await fetch(path);
    if (!response.ok) throw new Error(`HTTP ${response.status} on ${path}`);
    return response;
  }

  async function json(path) {
    return (await request(path)).json();
  }

  function conditionPath() {
    return `${root}/${state.condition.backend}/${state.condition.id}`;
  }

  function fillLoops() {
    const loops = Array.from({length: state.condition.passes}, (_, index) => [index, index]);
    options("from", loops);
    options("to", loops.filter(([index]) => index !== Number(get("from").value)));
  }

  const gaussianColors = [
    "#176b51", "#276aa2", "#a04a20", "#96408b", "#5f6718", "#ab2856", "#078183", "#6952b7",
    "#834c26", "#577540", "#323b83", "#ad5627", "#80364f", "#227591", "#6b5493", "#2f5031",
  ];
  let gaussianChart = null;

  function renderGaussianity() {
    if (!state.summary) return;
    const layer = get("gaussian-layer").value;
    const loop = get("gaussian-loop").value;
    const metric = get("gaussian-metric").value;
    const allLayers = layer === "all";
    get("gaussian-loop-field").hidden = !allLayers;
    const series = new Map();
    for (const visit of state.summary.gaussianity) {
      if (!allLayers && visit.layer !== Number(layer)) continue;
      if (allLayers && loop !== "all" && visit.loop !== null && visit.loop !== Number(loop)) continue;
      const key = allLayers ? visit.loop ?? (visit.layer < 6 ? "prelude" : "final") : "layer";
      if (!series.has(key)) {
        const color = typeof key === "number" ? gaussianColors[key % gaussianColors.length] : "#374e47";
        series.set(key, {color, visits: []});
      }
      series.get(key).visits.push(visit);
    }
    gaussianChart = {
      allLayers,
      first: 0,
      last: allLayers ? 23 : Math.max(state.condition.passes - 1, 0),
      label: get("gaussian-metric").selectedOptions[0].textContent,
      metric,
      reference: metric === "qq_rmse" ? state.summary.qq_reference : null,
      series: Array.from(series.values()),
      points: [],
      active: -1,
    };
    get("reference").textContent = state.summary.qq_reference.toFixed(4);
    get("observations").textContent = state.summary.count.toLocaleString();
    const canvas = get("gaussian-chart");
    canvas.setAttribute("aria-label", `${gaussianChart.label} ${allLayers ? "across layers" : `at layer ${layer} across loop visits`}`);
    showGaussianPoint(-1);
  }

  function drawGaussianity() {
    if (!gaussianChart) return;
    const {allLayers, first, last, label, metric, reference, series, active} = gaussianChart;
    const canvas = get("gaussian-chart");
    const width = canvas.clientWidth;
    const height = canvas.clientHeight;
    if (!width) return;
    const dpr = window.devicePixelRatio || 1;
    canvas.width = Math.round(width * dpr);
    canvas.height = Math.round(height * dpr);
    const ctx = canvas.getContext("2d");
    ctx.scale(dpr, dpr);
    const left = 52, right = width - 14, top = 25, bottom = height - 42;
    const values = series.flatMap(({visits}) => visits.map((visit) => visit[metric]));
    const max = Math.max(...values, reference ?? 0, 0.001) * 1.08;
    const xValue = (visit) => allLayers ? visit.layer : visit.loop ?? 0;
    const plottedLast = values.length === 1 ? 0 : last;
    const x = (index) => plottedLast === first ? (left + right) / 2 : left + (index - first) / (plottedLast - first) * (right - left);
    const y = (value) => bottom - value / max * (bottom - top);

    ctx.font = "11px system-ui";
    ctx.fillStyle = "#4b625b";
    ctx.fillText(label, left, 13);
    ctx.textAlign = "right";
    ctx.strokeStyle = "#d6dfd7";
    for (let i = 0; i <= 4; i++) {
      const value = i / 4 * max;
      const row = y(value);
      ctx.beginPath();
      ctx.moveTo(left, row);
      ctx.lineTo(right, row);
      ctx.stroke();
      ctx.fillText(value.toFixed(max < 1 ? 3 : 2), left - 7, row + 4);
    }
    ctx.textAlign = "center";
    const step = Math.max(1, Math.ceil((plottedLast + 1) * 23 / (right - left)));
    for (let i = first; i <= plottedLast; i += step) {
      const tick = values.length === 1 && series[0].visits[0].loop === null ? "Single visit" : String(i);
      ctx.fillText(tick, x(i), bottom + 17);
    }
    if (values.length > 1) ctx.fillText(allLayers ? "Physical layer" : "Loop index", (left + right) / 2, height - 6);

    if (reference !== null) {
      ctx.strokeStyle = "#667873";
      ctx.setLineDash([5, 4]);
      ctx.beginPath();
      ctx.moveTo(left, y(reference));
      ctx.lineTo(right, y(reference));
      ctx.stroke();
      ctx.setLineDash([]);
    }

    gaussianChart.points = series.flatMap(({color, visits}) => {
      const points = visits.map((visit) => ({visit, x: x(xValue(visit)), y: y(visit[metric])}));
      ctx.strokeStyle = ctx.fillStyle = color;
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      for (const point of points) ctx.lineTo(point.x, point.y);
      ctx.stroke();
      for (const point of points) {
        ctx.beginPath();
        ctx.arc(point.x, point.y, 2, 0, 2 * Math.PI);
        ctx.fill();
      }
      return points;
    });
    const point = gaussianChart.points[active];
    if (point) {
      ctx.strokeStyle = "#172723";
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.arc(point.x, point.y, 5, 0, 2 * Math.PI);
      ctx.stroke();
    }
  }

  function showGaussianPoint(index) {
    if (!gaussianChart) return;
    gaussianChart.active = index;
    drawGaussianity();
    const {points, metric, label, reference} = gaussianChart;
    const point = points[index];
    if (point) {
      const {visit} = point;
      const location = visit.loop === null ? "single visit" : `loop ${visit.loop}`;
      get("gaussian-point").textContent = `Layer ${visit.layer} · ${location} · ${label} ${visit[metric].toFixed(4)}`;
    } else {
      const note = reference === null ? "" : `Dashed line shows the Gaussian reference (${reference.toFixed(4)}). `;
      get("gaussian-point").textContent = note + "Hover or use the arrow keys to inspect points.";
    }
  }

  function hoverGaussianity(event) {
    if (!gaussianChart) return;
    const bounds = event.currentTarget.getBoundingClientRect();
    let selected = -1;
    let best = 20 ** 2;
    for (const [index, point] of gaussianChart.points.entries()) {
      const distance = (point.x - (event.clientX - bounds.left)) ** 2 + (point.y - (event.clientY - bounds.top)) ** 2;
      if (distance < best) {
        selected = index;
        best = distance;
      }
    }
    if (selected !== gaussianChart.active) showGaussianPoint(selected);
  }

  function renderDistances() {
    const distances = state.summary.mean_distances[get("layer").value];
    const currentSource = Number(get("from").value);
    const currentTarget = Number(get("to").value);
    const head = node("thead");
    const labels = node("tr");
    labels.append(node("th", "Source ↓ / Target →"),
      ...distances.map((_, index) => node("th", index)), node("th", "Mean"));
    head.append(labels);
    const body = node("tbody");
    for (const [source, values] of distances.entries()) {
      const row = node("tr");
      const label = node("th", source);
      label.scope = "row";
      row.append(label);
      for (const [target, distance] of values.entries()) {
        const cell = node("td");
        if (distance === null) {
          cell.textContent = "—";
        } else {
          const button = node("button", distance.toFixed(3));
          button.type = "button";
          button.setAttribute("aria-label", `Loop ${source} to ${target}: ${distance.toFixed(3)}`);
          button.setAttribute("aria-pressed", source === currentSource && target === currentTarget);
          button.addEventListener("click", () => {
            get("from").value = source;
            fillLoops();
            get("to").value = target;
            updatePair();
            get("distances").querySelector('button[aria-pressed="true"]')?.focus();
          });
          cell.append(button);
        }
        row.append(cell);
      }
      const valid = values.filter((distance) => distance !== null);
      row.append(node("td", (valid.reduce((sum, distance) => sum + distance, 0) / valid.length).toFixed(3)));
      body.append(row);
    }
    get("distances").replaceChildren(head, body);
  }

  function selectExample() {
    const dataset = get("dataset").value;
    options("example", state.examples.flatMap((example, index) => example.source === dataset
      ? [[index, `Row ${example.row} · ${example.tokens.length} tokens`]] : []));
    renderSequence();
  }

  function renderSequence() {
    state.sample = Number(get("example").value);
    state.position = 0;
    const example = state.examples[state.sample];
    const captured = new Map(example.positions.map((token, index) => [token, index]));
    const sequence = document.createDocumentFragment();
    for (const [index, token] of example.tokens.entries()) {
      if (!captured.has(index)) {
        sequence.append(document.createTextNode(token));
        continue;
      }
      const position = captured.get(index);
      const button = node("button", token.trim() ? token : token.replace(/ /g, "␠").replace(/\n/g, "↵") || "◌");
      button.type = "button";
      button.dataset.position = position;
      button.setAttribute("aria-label", `Token ${index + 1}: ${token.trim() || "whitespace or token fragment"}`);
      for (const event of ["pointerenter", "focus", "click"]) {
        button.addEventListener(event, () => {
          if (state.position === position) return;
          state.position = position;
          showMatches();
        });
      }
      sequence.append(button);
    }
    get("sequence").replaceChildren(sequence);
    get("sequence").scrollTop = 0;
    showMatches();
  }

  function renderMatch(entry, distance) {
    const sample = state.examples[Math.floor(entry / state.summary.positions)];
    const token = sample.positions[entry % state.summary.positions];
    const li = node("li");
    const source = state.catalog.sources.find(({name}) => name === sample.source);
    const link = node("a", sample.source);
    link.href = `https://huggingface.co/datasets/${source.repo}/tree/${source.revision}`;
    li.append(link, ` · row ${sample.row} · token ${token + 1} · distance ${distance.toFixed(4)}`);
    const context = node("div");
    context.className = "token-snippet";
    const start = Math.max(0, token - 8);
    const end = Math.min(sample.tokens.length, token + 9);
    context.append(
      (start ? "… " : "") + sample.tokens.slice(start, token).join(""),
      node("mark", sample.tokens[token] || "◌"),
      sample.tokens.slice(token + 1, end).join("") + (end < sample.tokens.length ? " …" : ""),
    );
    li.append(context);
    return li;
  }

  async function showMatches() {
    const version = ++matchVersion;
    const from = Number(get("from").value);
    const to = Number(get("to").value);
    const path = `${conditionPath()}/${get("layer").value}-${from}.bin`;
    const position = state.position;
    const query = state.sample * state.summary.positions + position;
    for (const button of get("sequence").querySelectorAll("button")) {
      button.setAttribute("aria-pressed", Number(button.dataset.position) === position);
    }
    get("matches").replaceChildren();
    get("match-status").textContent = "Finding target tokens…";
    if (shard.url !== path) {
      const promise = request(path).then((result) => result.arrayBuffer()).then((buffer) => new DataView(buffer));
      shard = {url: path, promise};
    }
    try {
      const view = await shard.promise;
      if (version !== matchVersion) return;
      const {count, neighbors} = state.summary;
      if (view.byteLength !== (state.condition.passes - 1) * count * neighbors * 8) {
        throw new Error("Wrong neighbor index size");
      }
      const targetOffset = to < from ? to : to - 1;
      const offset = (targetOffset * count + query) * neighbors * 8;
      const results = Array.from({length: neighbors}, (_, rank) => {
        const record = offset + rank * 8;
        return renderMatch(view.getUint32(record, true), view.getFloat32(record + 4, true));
      });
      get("matches").replaceChildren(...results);
      const tokenIndex = state.examples[state.sample].positions[position] + 1;
      get("match-status").textContent = `Closest tokens to source token ${tokenIndex} at target loop ${to}`;
    } catch (error) {
      if (version !== matchVersion) return;
      shard = {url: null, promise: null};
      get("match-status").textContent = "Neighbor examples are temporarily unavailable.";
      console.error(error);
    }
  }

  function updatePair() {
    renderDistances();
    showMatches();
  }

  async function selectCondition() {
    const version = ++conditionVersion;
    ++matchVersion;
    state.condition = state.catalog.conditions[Number(get("condition").value)];
    get("content").hidden = true;
    get("status").hidden = false;
    get("status").textContent = "Loading activation statistics…";
    try {
      const path = conditionPath();
      const [summary, examples] = await Promise.all([
        json(`${path}/summary.json`), state.condition.passes > 1 ? json(`${path}/examples.json`) : null,
      ]);
      if (version !== conditionVersion) return;
      Object.assign(state, {summary, examples});
      const layers = Array.from(new Set(summary.gaussianity.map(({layer}) => layer))).sort((a, b) => a - b);
      options("gaussian-layer", [["all", "All layers (0–23)"], ...layers.map((layer) => [layer, layer])]);
      const loops = Array.from({length: state.condition.passes}, (_, loop) => [loop, `Loop ${loop}`]);
      options("gaussian-loop", [["all", "All loop traces"], ...loops]);
      renderGaussianity();
      const hasNeighbors = state.condition.passes > 1;
      get("neighbor").hidden = !hasNeighbors;
      get("single").hidden = hasNeighbors;
      if (hasNeighbors) {
        options("layer", Object.keys(summary.mean_distances).map((layer) => [layer, layer]));
        fillLoops();
        options("dataset", state.catalog.sources.map(({name}) => [name, name]), get("dataset").value || "ARC-Challenge");
        renderDistances();
        selectExample();
      }
      get("content").hidden = false;
      drawGaussianity();
      get("status").hidden = true;
    } catch (error) {
      if (version !== conditionVersion) return;
      get("status").textContent = "Activation statistics are temporarily unavailable.";
      console.error(error);
    }
  }

  async function start() {
    try {
      state.catalog = await json(`${root}/index.json`);
      options("condition", state.catalog.conditions.map((condition, index) => {
        const name = condition.checkpoint === "fixed" ? "Fixed" : "Variable";
        const runtime = condition.backend === "ndif" ? "NDIF" : "Local IQ4_NL";
        return [index, `${runtime} · ${name} · ${condition.passes} ${condition.passes === 1 ? "pass" : "passes"}`];
      }));
      get("condition").disabled = false;
      await selectCondition();
    } catch (error) {
      get("status").textContent = "Activation statistics are temporarily unavailable.";
      console.error(error);
    }
  }

  get("condition").addEventListener("change", selectCondition);
  for (const control of ["gaussian-layer", "gaussian-loop", "gaussian-metric"]) {
    get(control).addEventListener("change", renderGaussianity);
  }
  const canvas = get("gaussian-chart");
  canvas.addEventListener("pointermove", hoverGaussianity);
  canvas.addEventListener("pointerleave", () => showGaussianPoint(-1));
  canvas.addEventListener("keydown", (event) => {
    const step = {ArrowLeft: -1, ArrowRight: 1}[event.key];
    if (!step || !gaussianChart.points.length) return;
    event.preventDefault();
    showGaussianPoint(Math.min(gaussianChart.points.length - 1, Math.max(0, gaussianChart.active + step)));
  });
  get("gaussian-chart").closest("details").addEventListener("toggle", drawGaussianity);
  window.addEventListener("resize", drawGaussianity);
  get("layer").addEventListener("change", updatePair);
  get("from").addEventListener("change", () => { fillLoops(); updatePair(); });
  get("to").addEventListener("change", updatePair);
  get("dataset").addEventListener("change", selectExample);
  get("example").addEventListener("change", renderSequence);
  start();
})();
