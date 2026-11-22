# File Reference

[Documentation index](../README.md) · [Architecture](architecture.md) · [Configuration](configuration.md)

## Top Level

| File | Purpose |
|---|---|
| `README.md` | Project summary, quickstart, and doc links |
| `AGENTS.md` | Agent entry point and repo conventions |
| `CLAUDE.md` | Claude-specific pointer to agent conventions |
| `pyproject.toml` | Package metadata and tool configuration |
| `uv.lock` | Locked dependency graph |
| `.pre-commit-config.yaml` | Shared gate for Python tooling, workflow semantics, local Markdown targets, justfile syntax, and file hygiene |
| `justfile` | Setup, checks, formatting, types, dead code, tests, coverage, workflow, and docs commands |
| `.python-version` | Python version for local tooling |
| `.gitignore` | Local artifacts excluded from git |
| `.editorconfig` | Editor defaults for encoding, indentation, whitespace, and line endings |
| `LICENSE` | MIT license |

## Tests

| File | Purpose |
|---|---|
| `tests/test_correctness_tools.py` | Phantom types, validated parsing, stable normalization, runtime array contracts, and property/regression tests |
| `tests/test_markdown_links.py` | File/link semantics and Git-discovery regression tests for the Markdown checker |

## Maintenance Commands

| File | Purpose |
|---|---|
| `scripts/__init__.py` | Importable namespace for repository tools |
| `scripts/check_markdown_links.py` | Offline check for local Markdown file and image targets across tracked and new documents |

## Docs

| Path | Purpose |
|---|---|
| `docs/README.md` | Documentation index |
| `docs/onboarding/getting-started.md` | Prerequisites, locked setup, and first checks |
| `docs/onboarding/workflows.md` | Change types, code, dependencies, and prepare a handoff |
| `docs/onboarding/glossary.md` | Shared correctness and tooling vocabulary |
| `docs/development/correctness.md` | Refinement types, runtime contracts, numeric validation, and static analysis |
| `docs/development/testing.md` | Test strategy, selection, strictness, and coverage interpretation |
| `docs/pipelines/experiment-lifecycle.md` | From a research question to reproducible outputs |
| `docs/reference/architecture.md` | Current layout and introducing the first source package |
| `docs/reference/configuration.md` | Tool settings, hook ownership, recipes, and CI |
| `docs/reference/file-reference.md` | This file map; update it when file roles change |

## CI

| File | Purpose |
|---|---|
| `.github/workflows/ci.yml` | Runs the shared pre-commit gate on main pushes, pull requests, and manual dispatch |
| `.github/dependabot.yml` | Weekly grouped uv and GitHub Actions dependency update proposals |
| `.github/rulesets/default-branch.json` | Portable PR and required-CI policy, applied separately to each GitHub repository |

## Inkling study

| File or directory | Purpose |
| --- | --- |
| `docs/pipelines/inkling-study.md` | Study design, runner setup, and activation schema |
| `looped_mechanisms/design.py` | Checkpoint conditions and visit order |
| `looped_mechanisms/inputs.py` | Pinned inputs and cohort preparation |
| `looped_mechanisms/ndif_run.py` | Hosted three-pass trace and benchmark runner |
| `looped_mechanisms/local_run.py` | Local IQ4_NL sweep orchestration |
| `native/` | llama.cpp graph observation bridge and build command |
| `looped_mechanisms/results.py` | Raw activation validation and result persistence |
| `looped_mechanisms/report.py` | Public aggregate generator |
| `looped_mechanisms/geometry.py` | Gaussianity summaries and directed cross-loop nearest-neighbor indexes |
| `looped_mechanisms/probe_study.py` | Prompt-grouped loop classifier training and five activation projections |
| `looped_mechanisms/layer_pairs.py` | NDIF decoder inputs and outputs for the distillation study |
| `looped_mechanisms/distill.py` | Teacher-layer-sized student training and transfer evaluation |
| `looped_mechanisms/stream_corpus.py` | Pinned tokenized corpus shards for larger distillation runs |
| `looped_mechanisms/stream_sft.py` | Chat corpus shards with assistant-token supervision |
| `looped_mechanisms/stream_capture.py` | Resume-safe NDIF capture for explicit token sequences |
| `looped_mechanisms/stream_step.py` | Student and optimizer frontier updates for one teacher shard |
| `looped_mechanisms/stream_campaign.py` | Bounded staging and evaluation milestones for the 10M-token run |
| `looped_mechanisms/checkpoint_catalog.py` | Publish evaluated student checkpoints to the dashboard slider |
| `site/plot-colors.js` | Ordered loop color scale shared by activation charts |
| `site/` | Benchmark, activation geometry, loop classification, projection, and layer transfer dashboard |
| `.agents/skills/` | Pinned upstream NDIF skills |
