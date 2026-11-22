# Looped Mechanisms

Layer- and pass-resolved activation capture and benchmark evaluation for the two
looped Thinking Machines Lab checkpoints, Inkling 2 Mini and
TML-d24-varloop-research. Both checkpoints use 24 physical transformer layers.
Layers 6–17 recur, so a forward with *p* passes visits 12 + 12*p* layers.

[Live dashboard](https://san-standards-ethnic-boutique.trycloudflare.com/) ·
[Study protocol](docs/pipelines/inkling-study.md) ·
[Development guide](docs/README.md)

## Reproduce

```sh
uv sync --locked
uv run --locked python -m looped_mechanisms.inputs
uv run --locked python -m looped_mechanisms.report
python -m http.server 45717 --bind 127.0.0.1 --directory site
```

Follow the [study protocol](docs/pipelines/inkling-study.md) to configure NDIF
and build the local runner. The dashboard shows benchmark accuracy, activation Gaussianity, cross-loop distances, loop classifiers, activation projections, and layer distillation. Benchmark plots read aggregate statistics
from `site/results.json`. Raw vectors and item scores are stored under
`artifacts/`, with input lineage under `data/cohort/`.

The repository includes project-local copies of the four
[NDIF skills](https://github.com/ndif-team/skills), pinned to upstream commit
[`fb71ca6`](https://github.com/ndif-team/skills/commit/fb71ca667ec91e37a58e677e8010a54e33ccd9d4).
They are discoverable in `.agents/skills/` when an agent starts in the project.
See the [upstream license](.agents/skills/NDIF-LICENSE).

MIT license for project code. See [LICENSE](LICENSE).
