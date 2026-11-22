"use strict";

(() => {
  const get = (id) => document.getElementById(`geometry-${id}`);
  const root = "geometry/v1";
  const state = {catalog: null, condition: null, summary: null};
  let conditionVersion = 0;

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

  function conditionPath(condition = state.condition) {
    return `${root}/${condition.backend}/${condition.id}`;
  }

  const gaussianColors = [
    "#176b51", "#276aa2", "#a04a20", "#96408b", "#5f6718", "#ab2856", "#078183", "#6952b7",
    "#834c26", "#577540", "#323b83", "#ad5627", "#80364f", "#227591", "#6b5493", "#2f5031",
  ];
  let gaussianChart = null;

  function renderGaussianity() {
    if (!state.summary) return;
    const loop = get("gaussian-loop").value;
    const metric = get("gaussian-metric").value;
    const series = new Map();
    for (const visit of state.summary.gaussianity) {
      if (loop !== "all" && visit.loop !== null && visit.loop !== Number(loop)) continue;
      const key = visit.loop ?? (visit.layer < 6 ? "prelude" : "final");
      if (!series.has(key)) {
        const color = typeof key === "number" ? gaussianColors[key % gaussianColors.length] : "#374e47";
        series.set(key, {color, visits: []});
      }
      series.get(key).visits.push(visit);
    }
    gaussianChart = {
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
    canvas.setAttribute("aria-label", `${gaussianChart.label} across physical layers`);
    showGaussianPoint(-1);
  }

  function drawGaussianity() {
    if (!gaussianChart) return;
    const {label, metric, reference, series, active} = gaussianChart;
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
    const lastLayer = 23;
    const x = (layer) => left + layer / lastLayer * (right - left);
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
    const step = Math.max(1, Math.ceil((lastLayer + 1) * 23 / (right - left)));
    for (let i = 0; i <= lastLayer; i += step) ctx.fillText(i, x(i), bottom + 17);
    ctx.fillText("Physical layer", (left + right) / 2, height - 6);

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
      const points = visits.map((visit) => ({visit, x: x(visit.layer), y: y(visit[metric])}));
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
        cell.textContent = distance === null ? "—" : distance.toFixed(3);
        row.append(cell);
      }
      const valid = values.filter((distance) => distance !== null);
      row.append(node("td", (valid.reduce((sum, distance) => sum + distance, 0) / valid.length).toFixed(3)));
      body.append(row);
    }
    get("distances").replaceChildren(head, body);
  }

  async function selectCondition() {
    const version = ++conditionVersion;
    const content = get("content");
    const status = get("status");
    const condition = state.catalog.conditions[Number(get("condition").value)];
    if (condition === state.condition) {
      content.removeAttribute("aria-busy");
      return;
    }
    content.setAttribute("aria-busy", "true");
    status.hidden = Boolean(state.condition);
    status.textContent = "Loading activation statistics…";
    try {
      const path = conditionPath(condition);
      const summary = await json(`${path}/summary.json`);
      if (version !== conditionVersion) return;
      Object.assign(state, {condition, summary});
      content.hidden = false;
      const loops = Array.from({length: state.condition.passes}, (_, loop) => [loop, `Loop ${loop}`]);
      options("gaussian-loop", [["all", "All loop traces"], ...loops]);
      renderGaussianity();
      const hasNeighbors = state.condition.passes > 1;
      get("neighbor").hidden = !hasNeighbors;
      get("single").hidden = hasNeighbors;
      if (hasNeighbors) {
        options("layer", Object.keys(summary.mean_distances).map((layer) => [layer, layer]));
        renderDistances();
      }
      status.hidden = true;
    } catch (error) {
      if (version !== conditionVersion) return;
      if (state.condition) get("condition").value = state.catalog.conditions.indexOf(state.condition);
      status.textContent = "Activation statistics are temporarily unavailable.";
      status.hidden = false;
      console.error(error);
    } finally {
      if (version === conditionVersion) content.removeAttribute("aria-busy");
    }
  }

  async function start() {
    try {
      state.catalog = await json(`${root}/index.json`);
      const defaultCondition = state.catalog.conditions.findIndex(({backend, id}) => backend === "ndif" && id === "fixed-03");
      options("condition", state.catalog.conditions.map((condition, index) => {
        const name = condition.checkpoint === "fixed" ? "Fixed" : "Variable";
        const runtime = condition.backend === "ndif" ? "NDIF" : "Local IQ4_NL";
        return [index, `${runtime} · ${name} · ${condition.passes} ${condition.passes === 1 ? "pass" : "passes"}`];
      }), defaultCondition);
      get("condition").disabled = false;
      await selectCondition();
    } catch (error) {
      get("status").textContent = "Activation statistics are temporarily unavailable.";
      console.error(error);
    }
  }

  get("condition").addEventListener("change", selectCondition);
  for (const control of ["gaussian-loop", "gaussian-metric"]) {
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
  get("layer").addEventListener("change", renderDistances);
  start();
})();
