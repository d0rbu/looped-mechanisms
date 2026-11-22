"use strict";

(() => {
  const get = (id) => document.getElementById(`study-${id}`);
  const root = "loop-study/v1";
  const names = {random: "Random projection", pca: "PCA", tsne: "t-SNE", umap: "UMAP", lda: "LDA"};
  const accuracyColors = ["#176b51", "#a7501b", "#28659d"];
  const scoreRequests = new Map();
  let comparison = [];
  const distillationRuns = ["distillation-06.json", "distillation.json", "distillation-17.json"];
  const distillationColors = new Map([[6, "#28659d"], [10, "#176b51"], [17, "#a7501b"]]);
  let students = [], timeline = [];
  let checkpointVersion = 0;
  let catalog, report, projection;
  let conditionVersion = 0, projectionVersion = 0;
  let plotted = [];
  let hiddenLoops = new Set();

  function node(tag, content) {
    const element = document.createElement(tag);
    if (content !== undefined) element.textContent = content;
    return element;
  }

  async function json(path) {
    const response = await fetch(`${root}/${path}`, {cache: "no-cache"});
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    return response.json();
  }

  function options(id, entries) {
    const select = get(id);
    const previous = select.value;
    select.replaceChildren(...entries.map(([value, label]) => new Option(label, value)));
    if (entries.some(([value]) => String(value) === previous)) select.value = previous;
  }

  function label(run) {
    const backend = run.backend === "ndif" ? "NDIF" : "Local IQ4_NL";
    const checkpoint = run.checkpoint === "fixed" ? "Fixed" : "Variable";
    return `${backend} · ${checkpoint} · ${run.passes} passes`;
  }

  function runPath(run) {
    return `${run.backend}/${run.id}`;
  }

  function selectedRun() {
    return catalog.runs[Number(get("condition").value)];
  }

  function scoresFor(run) {
    const path = runPath(run);
    if (!scoreRequests.has(path)) {
      scoreRequests.set(path, json(`${path}/scores.json`).catch((error) => {
        scoreRequests.delete(path);
        throw error;
      }));
    }
    return scoreRequests.get(path);
  }

  const percent = (fraction) => `${(fraction * 100).toFixed(1)}%`;
  async function loadComparison() {
    const requests = catalog.runs.map(async (run, index) => ({
      path: runPath(run),
      reference: index < accuracyColors.length,
      name: label(run),
      data: await scoresFor(run),
      color: accuracyColors[index] ?? "#363a42",
    }));
    const variable16 = catalog.runs.find((run) => run.backend === "llama.cpp-iq4" && run.id === "variable-16");
    if (variable16) {
      requests.push((async () => ({
        name: `${label(variable16)} · expanded layer-6 MLP`,
        reference: true,
        data: await json(`${runPath(variable16)}/expanded-scores.json`),
        color: "#755097",
      }))());
    }
    const loaded = await Promise.allSettled(requests);
    comparison = loaded.flatMap((entry) => entry.status === "fulfilled" ? [entry.value] : []);
    if (!comparison.length) return;
    updateAccuracyLegend();
    get("accuracy-comparison").hidden = false;
    drawAccuracy();
  }

  function visibleComparison() {
    const path = runPath(selectedRun());
    return comparison.filter((entry) => entry.reference || entry.path === path);
  }

  function updateAccuracyLegend() {
    get("accuracy-legend").replaceChildren(...visibleComparison().map(({name, data, color}) => {
      const labelNode = node("div");
      const dot = node("span");
      dot.style.setProperty("--loop-color", color);
      dot.setAttribute("aria-hidden", "true");
      labelNode.append(dot, `${name} (${data.train_prompts.toLocaleString()} training prompts, ${percent(1 / data.head_dim)} chance)`);
      return labelNode;
    }));
  }

  function drawAccuracy() {
    if (!comparison.length) return;
    const canvas = get("accuracy-plot");
    const width = canvas.clientWidth, height = canvas.clientHeight;
    if (!width) return;
    const ratio = window.devicePixelRatio || 1;
    canvas.width = Math.round(width * ratio);
    canvas.height = Math.round(height * ratio);
    const ctx = canvas.getContext("2d");
    ctx.scale(ratio, ratio);
    const left = 47, right = width - 16, top = 15, bottom = height - 40;
    const first = 6, last = 17;
    const x = (layer) => left + (layer - first) / (last - first) * (right - left);
    const y = (score) => bottom - score * (bottom - top);
    const selected = Number(get("layer").value) || first;
    ctx.font = "11px system-ui";
    ctx.fillStyle = "#4b625b";
    ctx.strokeStyle = "#d6dfd7";
    ctx.textAlign = "right";
    for (const tick of [0, 0.25, 0.5, 0.75, 1]) {
      ctx.beginPath();
      ctx.moveTo(left, y(tick));
      ctx.lineTo(right, y(tick));
      ctx.stroke();
      ctx.fillText(`${tick * 100}%`, left - 7, y(tick) + 4);
    }
    ctx.textAlign = "center";
    for (let layer = first; layer <= last; layer++) {
      ctx.fillText(layer, x(layer), bottom + 16);
    }
    ctx.fillText("Recurrent layer", (left + right) / 2, height - 6);
    ctx.beginPath();
    ctx.moveTo(x(selected), top);
    ctx.lineTo(x(selected), bottom);
    ctx.setLineDash([3, 3]);
    ctx.strokeStyle = "#83958c";
    ctx.stroke();
    ctx.setLineDash([]);

    for (const {data, color} of visibleComparison()) {
      ctx.strokeStyle = ctx.fillStyle = color;
      ctx.lineWidth = 2;
      ctx.beginPath();
      for (const row of data.layers) ctx.lineTo(x(row.layer), y(row.accuracy));
      ctx.stroke();
      for (const row of data.layers) {
        ctx.beginPath();
        ctx.arc(x(row.layer), y(row.accuracy), row.layer === selected ? 4.5 : 3, 0, 2 * Math.PI);
        ctx.fill();
      }
    }
    const values = visibleComparison().flatMap(({name, data}) => {
      const row = data.layers.find(({layer}) => layer === selected);
      return row ? [`${name} ${percent(row.accuracy)}`] : [];
    });
    get("accuracy-layer").textContent = `Layer ${selected} · ${values.join(" | ")}`;
  }

  function predictionAxes(canvas, labels, ticks, range, bottomLabel, format) {
    const width = canvas.clientWidth, height = canvas.clientHeight;
    if (!width) return null;
    const ratio = window.devicePixelRatio || 1;
    canvas.width = Math.round(width * ratio);
    canvas.height = Math.round(height * ratio);
    const ctx = canvas.getContext("2d");
    ctx.scale(ratio, ratio);
    const left = 39, right = width - 13, top = 10, bottom = height - 37;
    const step = (right - left) / labels.length;
    const x = (layer) => left + (layer - labels[0] + 0.5) * step;
    const y = (value) => bottom - (value - range[0]) / (range[1] - range[0]) * (bottom - top);
    ctx.font = "11px system-ui";
    ctx.fillStyle = "#4b625b";
    ctx.strokeStyle = "#d6dfd7";
    ctx.textAlign = "right";
    for (const tick of ticks) {
      ctx.beginPath();
      ctx.moveTo(left, y(tick));
      ctx.lineTo(right, y(tick));
      ctx.stroke();
      ctx.fillText(format(tick), left - 7, y(tick) + 4);
    }
    ctx.textAlign = "center";
    const labelStep = Math.max(1, Math.ceil(16 / step));
    for (const [index, layer] of labels.entries()) {
      if (index % labelStep === 0) ctx.fillText(layer, x(layer), bottom + 15);
    }
    ctx.fillText(bottomLabel, (left + right) / 2, height - 5);
    return {ctx, x, y, step};
  }

  function drawLoopHistogram(row, labels) {
    const canvas = get("loop-histogram");
    const axis = predictionAxes(canvas, labels, [0, 0.25, 0.5, 0.75, 1], [0, 1], "Predicted loop",
      (value) => `${value * 100}%`);
    if (!axis) return;
    const {ctx, x, y, step} = axis;
    const width = step * 0.67;
    for (const [index, layer] of labels.entries()) {
      ctx.fillStyle = row.loop === layer ? "#176b51" : "#819aab";
      const top = y(row.counts[index] / row.examples);
      ctx.fillRect(x(layer) - width / 2, top, width, y(0) - top);
    }
    const distribution = labels.map((layer, i) => `loop ${layer} ${percent(row.counts[i] / row.examples)}`);
    canvas.setAttribute("aria-label", `Predictions for true loop ${row.loop}. ${distribution.join(", ")}.`);
  }

  function drawLoopTrend(rows, labels, selected) {
    const band = rows.map((row) => Math.sqrt(row.variance));
    const low = Math.min(labels[0], ...rows.map((row, i) => row.mean - band[i]));
    const high = Math.max(labels.at(-1), ...rows.map((row, i) => row.mean + band[i]));
    const step = Math.max(1, Math.ceil(labels.length / 6));
    const ticks = labels.filter((loop) => loop % step === 0 || loop === labels.at(-1));
    const axis = predictionAxes(get("loop-trend"), labels, ticks, [low - 0.2, high + 0.2],
      "True loop", String);
    if (!axis) return;
    const {ctx, x, y} = axis;
    ctx.fillStyle = "#dcebe3";
    ctx.beginPath();
    rows.forEach((row, i) => ctx.lineTo(x(row.loop), y(row.mean + band[i])));
    for (let i = rows.length - 1; i >= 0; i--) {
      ctx.lineTo(x(rows[i].loop), y(rows[i].mean - band[i]));
    }
    ctx.closePath();
    ctx.fill();

    ctx.strokeStyle = "#6c7d76";
    ctx.setLineDash([4, 4]);
    ctx.beginPath();
    ctx.moveTo(x(labels[0]), y(labels[0]));
    ctx.lineTo(x(labels.at(-1)), y(labels.at(-1)));
    ctx.stroke();
    ctx.setLineDash([]);

    ctx.strokeStyle = ctx.fillStyle = "#176b51";
    ctx.lineWidth = 2;
    ctx.beginPath();
    for (const row of rows) ctx.lineTo(x(row.loop), y(row.mean));
    ctx.stroke();
    for (const row of rows) {
      ctx.beginPath();
      ctx.arc(x(row.loop), y(row.mean), row.loop === selected ? 4.5 : 2.5, 0, 2 * Math.PI);
      ctx.fill();
    }
  }

  function drawLoopPredictions() {
    if (!report) return;
    const layer = Number(get("layer").value);
    const stats = report.layers.find((entry) => entry.layer === layer);
    const labels = stats.confusion.map((_, loop) => loop);
    const rows = stats.confusion.map((counts, loop) => {
      const examples = counts.reduce((sum, count) => sum + count, 0);
      const mean = counts.reduce((sum, count, predicted) => sum + count * predicted, 0) / examples;
      const variance = counts.reduce((sum, count, predicted) => sum + count * (predicted - mean) ** 2, 0) / examples;
      return {loop, counts, examples, mean, variance};
    });
    const selected = Number(get("true-loop").value);
    const row = rows[selected];
    get("loop-histogram-note").textContent = `Layer ${layer}, true loop ${selected} · ${row.examples.toLocaleString()} predictions · mean ${row.mean.toFixed(2)} · variance ${row.variance.toFixed(2)} loops².`;
    drawLoopHistogram(row, labels);
    drawLoopTrend(rows, labels, selected);
  }

  function renderScores() {
    const layer = Number(get("layer").value);
    const stats = report.layers.find((row) => row.layer === layer);
    const classifier = report.hidden_units ? `MLP with ${report.hidden_units} hidden units` : "linear classifier";
    get("accuracy").textContent = `Layer ${layer} · ${percent(stats.accuracy)} accuracy · ${report.head_dim}-output ${classifier} · ${percent(1 / report.head_dim)} chance. Trained on ${report.train_prompts} prompts and validated on ${report.validation_prompts} separate prompts, with four token positions per loop.`;
    const header = node("tr");
    header.append(node("th", "Actual ↓ / Predicted →"),
      ...stats.confusion.map((_, loop) => node("th", loop)));
    const head = node("thead");
    head.append(header);
    const body = node("tbody");
    body.append(...stats.confusion.map((counts, loop) => {
      const row = node("tr");
      row.append(node("th", loop), ...counts.map((count) => node("td", count)));
      return row;
    }));
    get("confusion").replaceChildren(head, body);
    drawAccuracy();
    drawLoopPredictions();
  }

  function renderLegend() {
    const buttons = Array.from({length: projection.loops}, (_, loop) => {
      const button = node("button");
      button.type = "button";
      button.setAttribute("aria-pressed", !hiddenLoops.has(loop));
      const dot = node("span");
      dot.style.setProperty("--loop-color", loopColor(loop, projection.loops));
      dot.setAttribute("aria-hidden", "true");
      button.append(dot, `Loop ${loop}`);
      button.addEventListener("click", () => {
        if (hiddenLoops.has(loop)) hiddenLoops.delete(loop);
        else hiddenLoops.add(loop);
        button.setAttribute("aria-pressed", !hiddenLoops.has(loop));
        drawProjection();
      });
      return button;
    });
    get("legend").replaceChildren(...buttons);
  }

  function drawProjection() {
    if (!projection) return;
    const method = get("method").value;
    const points = projection.methods[method];
    const canvas = get("plot");
    const width = canvas.clientWidth, height = canvas.clientHeight;
    if (!width) return;
    const px = window.devicePixelRatio || 1;
    canvas.width = Math.round(width * px);
    canvas.height = Math.round(height * px);
    const ctx = canvas.getContext("2d");
    ctx.scale(px, px);
    const left = 25, right = width - 19, top = 14, bottom = height - 30;
    const xs = points.map((point) => point[0]), ys = points.map((point) => point[1]);
    const x0 = Math.min(...xs), y0 = Math.min(...ys);
    const dx = Math.max(...xs) - x0 || 1, dy = Math.max(...ys) - y0 || 1;
    ctx.strokeStyle = "#d6dfd7";
    ctx.strokeRect(left, top, right - left, bottom - top);
    ctx.fillStyle = "#4b625b";
    ctx.font = "11px system-ui";
    ctx.textAlign = "center";
    ctx.fillText("Component 1", (left + right) / 2, height - 7);
    ctx.save();
    ctx.translate(12, (top + bottom) / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText("Component 2", 0, 0);
    ctx.restore();
    plotted = [];
    ctx.globalAlpha = 0.72;
    for (const [index, point] of points.entries()) {
      const loop = index % projection.loops;
      if (hiddenLoops.has(loop)) continue;
      const x = left + 9 + (point[0] - x0) / dx * (right - left - 18);
      const y = bottom - 9 - (point[1] - y0) / dy * (bottom - top - 18);
      const item = projection.items[Math.floor(index / projection.loops)];
      plotted.push({x, y, loop, item});
      ctx.fillStyle = loopColor(loop, projection.loops);
      ctx.beginPath();
      ctx.arc(x, y, 2.8, 0, 2 * Math.PI);
      ctx.fill();
    }
    canvas.setAttribute("aria-label", `${names[method]} of layer ${projection.layer} by loop index`);
    const annotation = method === "lda" ? " LDA uses loop labels during fitting on training prompts." : "";
    get("projection-note").textContent = `${projection.items.length * projection.loops} final-token activations from held-out prompts.${annotation}`;
  }

  async function selectLayer() {
    if (!report) return;
    renderScores();
    const version = ++projectionVersion;
    if (selectedRun().projections === false) return;
    try {
      const layer = get("layer").value;
      const data = await json(`${runPath(selectedRun())}/layer-${layer.padStart(2, "0")}.json`);
      if (version !== projectionVersion) return;
      projection = data;
      get("status").hidden = true;
      renderLegend();
      drawProjection();
    } catch (error) {
      if (version !== projectionVersion) return;
      get("status").textContent = "Projection is temporarily unavailable.";
      get("status").hidden = false;
      console.error(error);
    }
  }

  async function selectCondition() {
    const version = ++conditionVersion;
    ++projectionVersion;
    try {
      const run = selectedRun();
      const data = await scoresFor(run);
      if (version !== conditionVersion) return;
      report = data;
      const hasProjection = run.projections !== false;
      projection = null;
      get("projection-content").hidden = !hasProjection;
      get("projection-control").hidden = !hasProjection;
      updateAccuracyLegend();
      options("true-loop", Array.from({length: data.head_dim}, (_, loop) => [loop, `Loop ${loop}`]));

      hiddenLoops = new Set();
      options("layer", report.layers.map(({layer}) => [layer, `Layer ${layer}`]));
      get("status").hidden = true;
      get("content").hidden = false;
      await selectLayer();
    } catch (error) {
      if (version !== conditionVersion) return;
      get("status").textContent = "Classifier results are temporarily unavailable.";
      get("status").hidden = false;
      console.error(error);
    }
  }

  function drawDistillationMetric(metric, baseline) {
    const full = students.reduce((best, current) =>
      current.data.layers.length > best.data.layers.length ? current : best);
    const scores = full.data.layers;
    const values = students.flatMap(({data}) => data.layers.map((row) => row[metric]));
    const max = Math.max(...values, ...scores.map((row) => row[baseline]));
    const upper = metric === "rmse" ? Math.max(1, Math.ceil(max)) : 1;
    const lower = metric === "rmse" ? 0 : -1;
    const ticks = Array.from({length: 5}, (_, i) => lower + i * (upper - lower) / 4);
    const layers = Array.from({length: 24}, (_, layer) => layer);
    const axis = predictionAxes(get(`distill-${metric}`), layers, ticks, [lower, upper], "Teacher layer",
      (tick) => Number.isInteger(tick) ? String(tick) : tick.toFixed(2));
    if (!axis) return;
    const {ctx, x, y} = axis;
    ctx.lineWidth = 1.5;
    ctx.setLineDash([4, 4]);
    ctx.strokeStyle = "#6c7d76";
    ctx.beginPath();
    for (const row of scores) ctx.lineTo(x(row.layer), y(row[baseline]));
    ctx.stroke();
    ctx.setLineDash([]);

    for (const {data, color} of students) {
      ctx.strokeStyle = ctx.fillStyle = color;
      ctx.lineWidth = 2;
      ctx.beginPath();
      for (const row of data.layers) ctx.lineTo(x(row.layer), y(row[metric]));
      ctx.stroke();
      for (const row of data.layers) {
        ctx.beginPath();
        ctx.arc(x(row.layer), y(row[metric]), row.layer === data.train_layer ? 4 : 2, 0, 2 * Math.PI);
        ctx.fill();
      }
    }
  }

  function drawDistillation() {
    if (!students.length) return;
    drawDistillationMetric("rmse", "copy_rmse");
    drawDistillationMetric("cosine", "copy_cosine");
  }

  const exposure = (data) => data.history.reduce((sum, epoch) => sum + epoch.tokens, 0);

  async function showCheckpoint() {
    const control = get("distill-checkpoint");
    const selected = timeline[Number(control.value)];
    if (!selected) return;
    const version = ++checkpointVersion;
    const results = selected.data ?? (await Promise.allSettled(selected.files.map(({file}) =>
      json(`distillation-checkpoints/${file}`)))).flatMap((entry) => entry.status === "fulfilled" ? [entry.value] : []);
    if (version !== checkpointVersion || !results.length) return;
    students = results.map((data) => ({data, color: distillationColors.get(data.train_layer)}));
    const {data} = students[0];
    const labelText = selected.kind === "stream"
      ? `${selected.phase === "sft" ? "SFT" : "Pretraining"} · ${selected.tokens.toLocaleString()} cumulative training tokens`
      : `Pilot · ${selected.tokens.toLocaleString()} token presentations`;
    get("distill-checkpoint-label").textContent = labelText;
    control.setAttribute("aria-valuetext", labelText);
    const corpus = selected.kind === "stream"
      ? (selected.phase === "sft"
        ? "FineWeb-Edu pretraining was followed by distillation at SmolTalk assistant positions."
        : "FineWeb-Edu training examples were seen once.")
      : `The pilot checkpoint follows ${data.history.length} epochs over the frozen prompts.`;
    get("distill-summary").textContent = `The teacher is NDIF Inkling 2 Mini with three fixed passes. Each student has ${data.student_parameters.toLocaleString()} parameters. ${corpus} Validation uses ${data.validation_prompts} held-out prompts with prefixes up to ${data.max_prompt_tokens} tokens.`;
    const labels = [...students.map(({data: student, color}) => [`Trained on layer ${student.train_layer}`, color]),
      ["Copy-input baseline (dashed)", "#6c7d76"]];
    get("distill-legend").replaceChildren(...labels.map(([name, color]) => {
      const container = node("div");
      const dot = node("span");
      dot.style.setProperty("--loop-color", color);
      dot.setAttribute("aria-hidden", "true");
      container.append(dot, name);
      return container;
    }));
    get("distill-content").hidden = false;
    drawDistillation();
    const complete = students.filter(({data: student}) => student.layers.length === 24).length;
    const status = get("distill-status");
    status.hidden = complete === distillationRuns.length;
    status.textContent = `${complete}/${distillationRuns.length} students evaluated at all 24 layers for this checkpoint.`;
  }

  async function loadDistillation() {
    const [reports, index] = await Promise.all([
      Promise.allSettled(distillationRuns.map(json)),
      json("distillation-checkpoints/index.json").catch(() => ({checkpoints: []})),
    ]);
    const control = get("distill-checkpoint");
    const previous = timeline[Number(control.value)]?.key;
    const followedLatest = !previous || Number(control.value) === timeline.length - 1;
    const groups = new Map();
    for (const entry of index.checkpoints) {
      const key = `${entry.kind}/${entry.tokens}`;
      if (!groups.has(key)) groups.set(key, {key, kind: entry.kind, phase: entry.phase, tokens: entry.tokens, files: []});
      groups.get(key).files.push(entry);
    }
    const pilot = reports.flatMap((entry) => entry.status === "fulfilled" ? [entry.value] : []);
    if (pilot.length) {
      const tokens = exposure(pilot[0]);
      const key = `pilot/${tokens}`;
      groups.set(key, {key, kind: "pilot", tokens, data: pilot});
    }
    timeline = Array.from(groups.values()).sort((a, b) =>
      Number(a.kind === "stream") - Number(b.kind === "stream") || a.tokens - b.tokens);
    if (timeline.length) {
      control.max = timeline.length - 1;
      control.disabled = timeline.length < 2;
      const preserved = timeline.findIndex((point) => point.key === previous);
      control.value = followedLatest || preserved < 0 ? control.max : preserved;
      await showCheckpoint();
    }
    if (!index.completed) window.setTimeout(loadDistillation, 30_000);
  }

  async function start() {
    try {
      catalog = await json("index.json");
      options("condition", catalog.runs.map((run, index) => [index, label(run)]));
      options("method", catalog.methods.map((method) => [method, names[method]]));
      get("method").value = "pca";
      get("condition").disabled = false;
      await Promise.all([selectCondition(), loadComparison()]);
    } catch (error) {
      get("status").textContent = "Loop study data is temporarily unavailable.";
      console.error(error);
    }
  }

  get("condition").addEventListener("change", selectCondition);
  get("layer").addEventListener("change", selectLayer);
  get("method").addEventListener("change", drawProjection);
  get("plot").addEventListener("pointermove", (event) => {
    const {left, top} = event.currentTarget.getBoundingClientRect();
    let closest;
    let best = 8 ** 2;
    for (const point of plotted) {
      const distance = (point.x - (event.clientX - left)) ** 2 + (point.y - (event.clientY - top)) ** 2;
      if (distance < best) {
        closest = point;
        best = distance;
      }
    }
    get("point").textContent = closest ? `Loop ${closest.loop} · ${closest.item.source} · ${closest.item.id}`
      : "Hover over a point to see its dataset and loop.";
  });
  window.addEventListener("resize", drawProjection);
  window.addEventListener("resize", drawAccuracy);
  get("true-loop").addEventListener("change", drawLoopPredictions);
  window.addEventListener("resize", drawLoopPredictions);
  window.addEventListener("resize", drawDistillation);
  get("distill-checkpoint").addEventListener("input", showCheckpoint);
  start();
  loadDistillation();
})();
