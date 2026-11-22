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

  function renderGaussianity() {
    get("reference").textContent = state.summary.qq_reference.toFixed(4);
    get("observations").textContent = state.summary.count.toLocaleString();
    get("gaussian").replaceChildren(...state.summary.gaussianity.map((visit) => {
      const row = node("tr");
      row.append(...[visit.layer, visit.loop ?? "—", visit.qq_rmse.toFixed(4),
        visit.median_abs_excess_kurtosis.toFixed(4)].map((value) => node("td", value)));
      return row;
    }));
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
  get("layer").addEventListener("change", updatePair);
  get("from").addEventListener("change", () => { fillLoops(); updatePair(); });
  get("to").addEventListener("change", updatePair);
  get("dataset").addEventListener("change", selectExample);
  get("example").addEventListener("change", renderSequence);
  start();
})();
