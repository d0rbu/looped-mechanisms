"use strict";

(() => {
  const get = (id) => document.getElementById(`study-${id}`);
  const root = "loop-study/v1";
  const names = {random: "Random projection", pca: "PCA", tsne: "t-SNE", umap: "UMAP", lda: "LDA"};
  const colors = ["#126e4b", "#ad3e31", "#2b68ac", "#943f96", "#805309", "#26838d", "#d04985", "#454495",
    "#5e741c", "#b65c2b", "#655350", "#9063a9", "#0080ad", "#844146", "#337d2d", "#30383f"];
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

  function selectedPath() {
    const run = catalog.runs[Number(get("condition").value)];
    return `${run.backend}/${run.id}`;
  }

  const percent = (fraction) => `${(fraction * 100).toFixed(1)}%`;
  function renderScores() {
    const layer = Number(get("layer").value);
    const stats = report.layers.find((row) => row.layer === layer);
    get("accuracy").textContent = `Layer ${layer} · ${percent(stats.accuracy)} accuracy · ${report.head_dim}-output linear head · ${percent(1 / report.head_dim)} chance. Trained on ${report.train_prompts} prompts and validated on ${report.validation_prompts} separate prompts, with four token positions per loop.`;
    get("scores").replaceChildren(...report.layers.map((score) => {
      const row = node("tr");
      row.dataset.selected = score.layer === layer;
      row.append(...[score.layer, percent(score.accuracy), `${score.correct.toLocaleString()} / ${score.examples.toLocaleString()}`]
        .map((value) => node("td", value)));
      return row;
    }));
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
  }

  function renderLegend() {
    const buttons = Array.from({length: projection.loops}, (_, loop) => {
      const button = node("button");
      button.type = "button";
      button.setAttribute("aria-pressed", !hiddenLoops.has(loop));
      const dot = node("span");
      dot.style.setProperty("--loop-color", colors[loop]);
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
      ctx.fillStyle = colors[loop];
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
    try {
      const layer = get("layer").value;
      const data = await json(`${selectedPath()}/layer-${layer.padStart(2, "0")}.json`);
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
      const data = await json(`${selectedPath()}/scores.json`);
      if (version !== conditionVersion) return;
      report = data;
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

  async function loadDistillation() {
    const status = get("distill-status");
    try {
      const data = await json("distillation.json");
      get("distill-summary").textContent = `Trained on layer ${data.train_layer} with ${data.student_parameters.toLocaleString()} parameters versus ${data.teacher_parameters.toLocaleString()} in the NDIF teacher layer. Validation uses ${data.validation_prompts} held-out prompt prefixes of up to ${data.max_prompt_tokens} tokens. The same student is evaluated at every layer.`;
      get("distill-rows").replaceChildren(...data.layers.map((score) => {
        const row = node("tr");
        row.dataset.trainingLayer = score.layer === data.train_layer;
        row.append(...[score.layer, score.normalized_rmse.toFixed(4), score.copy_normalized_rmse.toFixed(4),
          score.cosine.toFixed(4), score.tokens.toLocaleString()].map((value) => node("td", value)));
        return row;
      }));
      status.hidden = data.layers.length === 24;
      status.textContent = `${data.layers.length}/24 evaluation layers available.`;
      get("distill-content").hidden = false;
      if (status.hidden) return;
    } catch {
      if (get("distill-content").hidden) status.textContent = "Waiting for held-out distillation results.";
    }
    window.setTimeout(loadDistillation, 30_000);
  }

  async function start() {
    try {
      catalog = await json("index.json");
      options("condition", catalog.runs.map((run, index) => [index, label(run)]));
      options("method", catalog.methods.map((method) => [method, names[method]]));
      get("method").value = "pca";
      get("condition").disabled = false;
      await selectCondition();
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
  start();
  loadDistillation();
})();
