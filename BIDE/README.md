# BIDE implementation

This directory contains the public implementation of the final MAP memory
experiment. It is the code component used by the main LoCoMo answer-judge
pipeline described in `docs/MAIN_EXPERIMENT_IMPLEMENTATION.md`.

## Main pipeline

```text
Question
  → Query Frame / query_text
  → Clause, Occurrence, Structural, Multivector, SPLADE, Typed-L2 retrieval
  → Raw-ID projection
  → RRF Candidate128
  → Qwen3 explicit-pair reranking
  → Top10 / Top20 evidence packet
  → original text + speaker + time + image caption
  → extract-first Reader
  → Judge
```

The six retrieval families are projected back to conversation-local Raw IDs
before fusion. RRF uses:

```text
score(r) = Σ_f 1 / (20 + rank_f(r))
```

Top10 and Top20 are prefixes of the same reranked candidate list. The Reader
receives the original evidence fields and produces the answer used by the
judge.

## Directory layout

- `src/best_memory/`: memory construction, query construction, retrieval,
  reranking, answering, runtime clients, and evaluation code.
- `prompts/`: memory and Query Frame prompt contracts.
- `schemas/`: machine-readable Query Frame and memory assertion schemas.
- `configs/`: example model and service profiles; credentials are external.
- `scripts/`: run, audit, ablation, and summary utilities.
- `tests/`: contract and smoke tests.
- `docs/`: implementation contract, runbooks, and experiment records.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e .
```

Full execution requires the original dataset, local embedding/reranker models,
and a configured generation and judge service. Model paths and credentials are
not included in this public bundle.

## Checks

```bash
python -m compileall -q src scripts
PYTHONPATH=src python -m pytest -q tests
```

Use a small fixture before launching a full run. Runtime outputs should be
written to an external run directory and accompanied by configuration and
source-hash manifests.

## Public release boundary

This release contains source code, prompts, schemas, tests, and documentation.
Large generated runs, model weights, API keys, request/response caches, and
private logs are excluded. The files in this directory are the explicit source
for the current main pipeline described above.
