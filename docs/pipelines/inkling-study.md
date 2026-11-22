# Inkling 2 Mini study protocol

[Project README](../../README.md) · [NDIF model guide](https://nnsight.net/blog/2026/11/20/opening-the-box-of-inkling-2-mini-using-nnsight-ndif/) · [Release](https://thinkingmachines.ai/news/inkling-2-mini/)

## Conditions

The public NDIF inventory is queried before a collection. The two hosted
`image-text-to-text` deployments have three passes configured. The local
llama.cpp sweep includes Inkling 2 Mini at its fixed three passes, and
TML-d24-varloop-research at every integer pass count from 1 to 16. Local weights
are the ggml-org IQ4_NL conversions. All 17 local conditions use the same
runtime and quantization format. The dashboard shows hosted and local results
under separate runtime selectors. Passes 6–16 exceed the variable model's
training range of 1–5.

The study asks how activity changes at repeated physical layers and whether
more passes improve accuracy on identical benchmark questions. Benchmark
results can be nonmonotonic, including at higher pass counts where the model
runs outside its training range. Active-block RMS and cosine similarity
describe geometric changes and do not identify their causal role.

## Frozen inputs and evaluation

| Source | Use | Split |
| --- | --- | --- |
| [ToolACE](https://huggingface.co/datasets/Team-ACE/ToolACE) | Recorded agent dialog with tool observations | Train |
| [Dolci Tool Use](https://huggingface.co/datasets/allenai/Dolci-Think-SFT-Olmo-Hybrid-Tool-Use-SA) | Recorded multistep retrieval conversations | Train |
| [Everyday Conversations](https://huggingface.co/datasets/HuggingFaceTB/smoltalk) | Conversation | Test |
| [SystemChats](https://huggingface.co/datasets/HuggingFaceTB/smoltalk) | System-instruction conversation | Test |
| [ARC-Challenge](https://huggingface.co/datasets/allenai/ai2_arc) | Four-choice science QA | Test |
| [HellaSwag](https://huggingface.co/datasets/Rowan/hellaswag) | Four-choice continuation | Validation |

The input freezer pins the source file revisions and hashes, shuffles row
indices with seed 20261121 separately for each source, and ranks 64 eligible
unique items per source. Eligible items fit a 2400-byte rendered text
budget and contain no reserved Inkling markers. ARC items must have four
answer options. The byte budget favors shorter traces. Raw source text is
replayed using text role markers. ToolACE contributes the system instructions
before the function catalog and the complete dialog, including recorded tool
calls and responses. Dolci environment messages use the tool role.
No tools are invoked during scoring or tracing.

The benchmark protocol takes the highest final-token logit among the literal
ASCII labels A, B, C, D under a zero-shot prompt with thinking effort set to
zero. It is a four-choice, text-only comparison of loop counts with no
generated reasoning or external judge. Scores reflect the protocol and frozen
sample described here. The site shows per-condition denominators and 95%
Wilson intervals. Interim points built from different item sets are left
unconnected.

## Activation artifacts

One forward saves the 2304 hidden coordinates of the active-block output at
the 25%, 50%, 75% and final token indices at each visit, in FP16. Positions are
zero-based, computed against the final input-token index and stored alongside
the token IDs. The activation array is `[visit, position, hidden]`. Visits are
in prelude, pass-major core, final-block order. The first and last six layers
execute once. At 16 passes there are 204 visits.

The active block is the accumulation in the current six-layer window. The
model also carries completed blocks and cross-pass messages in its forward
state. The published activation metrics use the last token only. For RMS,
each vector's RMS is computed before averaging over items. For pass cosine,
each vector is compared to the same physical layer one pass earlier within the
same forward. Prelude, final-block, and pass-one cosine are undefined.

Each NPZ has a JSON completion record containing provenance, input digest,
observed visit count, file size, token count, summary statistics, and the
choice logits and answer where applicable. Only shape-checked finite tensors
and complete visit sets enter the report. Raw vectors are retained locally.
The public web root serves the static app and cohort-level aggregate JSON.

## Activation geometry

The dashboard reports layer-by-layer Gaussianity for each recorded visit.
The line plot shows Gaussianity across all physical layers, with optional
filtering by loop trace. The explorer opens on the NDIF fixed three-pass run
and keeps the current results in view while another condition loads. Each visit pools the four saved positions from all 384 prompts. The Q–Q
error is the mean root mean squared difference between ordered, standardized
samples and theoretical Gaussian quantiles, across 32 seeded random
projections. The dashed Q–Q reference is the same score averaged over 128 equally
sized simulated IID standard-Gaussian samples. It is descriptive because
tokens in the same prompt are dependent. Median absolute excess kurtosis
measures marginal tail departure over nonconstant hidden coordinates.

For physical layers 6–17, each directed source/target loop pair has one
nearest-neighbor search per captured source token. Candidates come from all
captured tokens at the same physical layer in the target loop, excluding
the entire source prompt. Distances are Euclidean divided by the square
root of the hidden width. The displayed pair average uses the nearest
candidate for each source token. The row mean averages over all other loops.
Select a distance matrix cell to choose a source and target loop. The
explorer shows three ranked matches from the frozen cohort, each with up to
32 tokens of context on either side. Hover and keyboard focus are available
on the four recorded source tokens. The display uses zero-based loop indices.

Build the public summaries and neighbor indexes from local artifacts and
the pinned tokenizer JSON shared by both checkpoints. The indexes contain
target token indices and distances, with separate tokenized examples and
dataset attribution. Raw activation tensors remain in the artifact directory.

```sh
OPENBLAS_NUM_THREADS=8 uv run --locked python -m looped_mechanisms.geometry --tokenizer "$tokenizer_json"
```

For CUDA acceleration, add `--with cupy-cuda12x` before `python` and
`--cuda` after `--tokenizer "$tokenizer_json"`. The builder publishes
completed conditions incrementally under `site/geometry/v1`.

## Run NDIF

Use a separate client environment so the regular project checks can run with
no model dependencies. Get a key from [NDIF](https://login.ndif.us/) and export
`NDIF_API_KEY` or store it in a local Git-ignored `.env` loaded by the launcher.

```sh
uv venv --python 3.12 .venv-ndif
uv pip install --python .venv-ndif/bin/python --index https://download.pytorch.org/whl/cpu 'torch==2.10.0+cpu' 'torchvision==0.25.0+cpu'
uv pip install --python .venv-ndif/bin/python 'nnsight==0.8.3' 'transformers==5.24.0' pillow
uv run --env-file .env --no-project --python .venv-ndif/bin/python -- python -m looped_mechanisms.ndif_run
```

## Run the full sweep

Build llama.cpp at commit
[`225a5345`](https://github.com/ggml-org/llama.cpp/commit/225a5345bbb0e0c70859af5441ba922beb666fcf)
with shared libraries and CUDA for the local GPU. Apply the graph capacity
patch before building. Inkling shares weights across passes, but each visit
requires its own graph nodes. Link the activation reader against that build.

```sh
git -C "$llama_source" apply "$PWD/native/llama-inkling-graph-capacity.patch"
cmake -S "$llama_source" -B "$llama_source/build" -DGGML_CUDA=ON -DBUILD_SHARED_LIBS=ON
cmake --build "$llama_source/build" --target llama
native/build.sh "$llama_source" "$llama_source/build/bin"
```

Download the IQ4_NL model files from
[Inkling 2 Mini GGUF](https://huggingface.co/ggml-org/Inkling-2-Mini-GGUF/tree/e8a2cecf0e78914f07cc27b9be711a9b2c7561d9) and
[TML-d24-varloop-research GGUF](https://huggingface.co/ggml-org/TML-d24-varloop-research-GGUF/tree/7af7f5e17b57395455bd8a37d8cfefa98824ca6e)
into a directory of your choice.

```sh
uv run --locked python -m looped_mechanisms.local_run --models "$models" --verify-weights
uv run --locked python -m looped_mechanisms.report --interval 25
```

Either runner takes `--per-source`, `--source`, and `--condition` for a
bounded initial run. Rerun against the original artifact directory to extend or
resume it. Use one writer per backend and condition. The complete study plans
768 hosted forwards and 6,528 local forwards. The pinned local weight files
total about 35.3 GB, and uncompressed activation arrays for both runtimes
total about 14 GB.
