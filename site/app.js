"use strict";

const $ = (id) => document.getElementById(id);
let data;
let chartRows = [];

function optionList(id, options) {
  const select = $(id);
  const previous = select.value;
  select.replaceChildren(...options.map(([value, label]) => new Option(label, value)));
  if (options.some(([value]) => value === previous)) select.value = previous;
}

function element(tag, text) {
  const node = document.createElement(tag);
  node.textContent = text;
  return node;
}

function checkpointName(condition) {
  return condition.checkpoint === "fixed" ? "Fixed" : "Variable";
}

function wilson(correct, count) {
  const p = correct / count;
  const z = 1.95996398454;
  const denom = 1 + z ** 2 / count;
  const midpoint = (p + z ** 2 / (2 * count)) / denom;
  const radius = z * Math.sqrt(p * (1 - p) / count + z ** 2 / (4 * count ** 2)) / denom;
  return [Math.max(0, midpoint - radius), Math.min(1, midpoint + radius)];
}

function drawChart() {
  if (!chartRows.length) return;
  const canvas = $("chart");
  const width = canvas.clientWidth;
  const height = canvas.clientHeight;
  if (!width) return;
  const dpi = window.devicePixelRatio || 1;
  canvas.width = width * dpi;
  canvas.height = height * dpi;
  const ctx = canvas.getContext("2d");
  ctx.scale(dpi, dpi);
  const left = 49, right = width - 15, top = 17, bottom = height - 42;
  const ndif = $("backend").value === "ndif";
  const x = (passes) => ndif ? (left + right) / 2 : left + (passes - 1) * (right - left) / 15;
  const y = (value) => bottom - (bottom - top) * value;
  const colors = {variable: "#16634e", fixed: "#a24f19"};
  ctx.font = "11px system-ui";
  ctx.textAlign = "right";
  for (const step of [0, .25, .5, .75, 1]) {
    ctx.strokeStyle = step === .25 ? "#acbcb2" : "#e0e8e1";
    ctx.setLineDash(step === .25 ? [4, 4] : []);
    ctx.beginPath();
    ctx.moveTo(left, y(step));
    ctx.lineTo(right, y(step));
    ctx.stroke();
    ctx.fillStyle = "#4b625b";
    ctx.fillText(`${step * 100}%`, left - 8, y(step) + 4);
  }
  ctx.setLineDash([]);
  ctx.textAlign = "center";
  const passes = ndif ? [3] : Array.from({length: 16}, (_, i) => i + 1);
  for (const pass of passes) ctx.fillText(pass, x(pass), bottom + 16);
  ctx.fillText("Loop passes", (left + right) / 2, height - 7);

  const curve = chartRows.filter((row) => row.checkpoint === "variable");
  ctx.strokeStyle = colors.variable;
  ctx.lineWidth = 1.6;
  for (let i = 1; i < curve.length; i++) {
    const a = curve[i - 1], b = curve[i];
    if (a.passes + 1 !== b.passes || a.ranks.join() !== b.ranks.join()) continue;
    ctx.beginPath();
    ctx.moveTo(x(a.passes), y(a.correct / a.evaluated));
    ctx.lineTo(x(b.passes), y(b.correct / b.evaluated));
    ctx.stroke();
  }

  for (const row of chartRows) {
    const dx = row.checkpoint === "fixed" ? 6 : (row.passes === 3 ? -5 : 0);
    const cx = x(row.passes) + dx, cy = y(row.correct / row.evaluated);
    const [low, high] = wilson(row.correct, row.evaluated);
    ctx.strokeStyle = ctx.fillStyle = colors[row.checkpoint];
    ctx.lineWidth = 1.25;
    ctx.beginPath();
    ctx.moveTo(cx, y(low));
    ctx.lineTo(cx, y(high));
    for (const bound of [low, high]) {
      ctx.moveTo(cx - 3, y(bound));
      ctx.lineTo(cx + 3, y(bound));
    }
    ctx.stroke();
    ctx.beginPath();
    if (row.checkpoint === "fixed") {
      ctx.moveTo(cx, cy - 5);
      ctx.lineTo(cx + 5, cy);
      ctx.lineTo(cx, cy + 5);
      ctx.lineTo(cx - 5, cy);
      ctx.closePath();
    } else {
      ctx.arc(cx, cy, 4, 0, 2 * Math.PI);
    }
    ctx.fill();
  }
}

function renderBenchmark(conditions) {
  const name = $("benchmark").value;
  chartRows = conditions
    .map((condition) => ({...condition, ...condition.sources[name]}))
    .filter((row) => row.evaluated > 0)
    .sort((a, b) => a.passes - b.passes || a.checkpoint.localeCompare(b.checkpoint));
  $("benchmark-empty").hidden = chartRows.length > 0;
  $("chart-area").hidden = chartRows.length === 0;
  $("benchmark-empty").textContent = "Scored items will appear here as they finish.";
  if (chartRows.length) {
    const counts = chartRows.map((row) => row.evaluated);
    const low = Math.min(...counts), high = Math.max(...counts);
    $("scored-count").textContent = `${low === high ? low : `${low}–${high}`} of ${data.target_per_source} items scored per shown condition`;
  }
  $("scores").replaceChildren(...chartRows.map((row) => {
    const tr = document.createElement("tr");
    const pct = (100 * row.correct / row.evaluated).toFixed(1) + "%";
    tr.append(...[checkpointName(row), row.passes, pct, `${row.correct} / ${row.evaluated}`]
      .map((value) => element("td", value)));
    return tr;
  }));
  drawChart();
}

function render() {
  if (!data) return;
  const backend = $("backend").value;
  const conditions = data.backends[backend];
  const total = conditions.length * data.sources.length * data.target_per_source;
  const count = conditions.reduce((sum, arm) => sum + Object.values(arm.sources)
    .reduce((subtotal, source) => subtotal + source.count, 0), 0);
  $("progress").textContent = `${count.toLocaleString()} / ${total.toLocaleString()} verified forwards`;
  $("runtime-note").textContent = backend === "ndif"
    ? "Hosted deployments run at three passes."
    : "Fixed three-pass baseline and variable passes 1–16. Quantization: IQ4_NL.";
  renderBenchmark(conditions);
  $("updated").dateTime = data.generated_at;
  $("updated").textContent = `Updated ${new Date(data.generated_at).toLocaleString()}`;
}

async function refresh() {
  try {
    const response = await fetch("results.json", {cache: "no-store"});
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const initial = !data;
    data = await response.json();
    if (initial) {
      const benchmarks = data.sources.filter((source) => source.category === "Benchmark");
      optionList("benchmark", benchmarks.map((source) => [source.name, source.name]));
    }
    render();
  } catch {
    $("progress").textContent = data ? "Latest refresh unavailable" : "Waiting for the study data file";
  }
}

for (const name of ["backend", "benchmark"]) {
  $(name).addEventListener("change", render);
}
window.addEventListener("resize", drawChart);
refresh();
window.setInterval(refresh, 30000);
