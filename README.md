# MAP Platform

MAP Platform is a reproducible evaluation workspace for long-context memory systems. It combines a structured memory method, unified benchmark evaluation, diagnostic analysis, and a browser-based result viewer.

This directory is the curated public release. It is designed for a fresh GitHub repository and excludes private credentials, model weights, API caches, and large transient experiment directories.

## Included components

```text
github_release/
├── BIDE/              # BIDE implementation
├── unified_evaluation/          # shared evaluation and analysis code
│   ├── raw_eval/                # current raw-data evaluation entry points
│   ├── MEMPATH/           # MEMPATH diagnostic metrics
│   ├── analysis/                # result and failure analysis
│   └── configs/                 # non-secret example configurations
├── map_eval_frontend_legacy/    # legacy MAP result viewer
├── MANIFEST.json
└── README.md
```

The final main experiment is documented in `BIDE/docs/MAIN_EXPERIMENT_IMPLEMENTATION.md`.

## BIDE pipeline

```text
Question Q
  → Query Frame / query_text
  → Clause, Occurrence, Structural, Multivector, SPLADE, Typed-L2 retrieval
  → Raw-ID projection
  → RRF Candidate128
  → Qwen3 explicit-pair reranking
  → Top10 / Top20 evidence packet
  → original text + speaker + time + image caption
  → extract-first Reader
  → Judge
  → final evaluation
```

The RRF score is `score(r) = Σ_f 1 / (20 + rank_f(r))`. Top10 and Top20 are prefixes of the same reranked list. The Query Frame stores entities, roles, relations, events, values, time, and state/modality fields.

The implementation separates paper-level abstractions from executable objects: Query Frame, Candidate128, Top-K evidence packet, and Reader input.

## Unified evaluation

The current system-comparison entry point is:

```bash
python unified_evaluation/raw_eval/run_memory_eval.py \
  --config unified_evaluation/configs/memory_eval.yaml
```

The runner supports LoCoMo and LongMemEval adapters, system preflight checks, incremental JSONL output, source tracing, official-style judging, and summary tables. Parallel variants are in the same directory.

`MEMPATH/` contains the MEMPATH diagnostic pipeline: encoding, organization, retrieval, injection, and utilization. It is used for diagnosis and comparative analysis and is separate from the current raw-data comparison.

## Analysis and reporting

`unified_evaluation/analysis/` contains utilities for aggregation, failure analysis, lexical metrics, report tables, and audit summaries. Published results should include the dataset split, system version, model and judge configuration, question count, abstention policy, file checksums, and known limitations.

## Frontend viewer

`map_eval_frontend_legacy/` contains the legacy viewer for system profiles, mechanism flows, funnel summaries, and sample traces. Its checked-in `public-data/` is a LoCoMo-only baseline bundle aligned with the paper's comparison and MemPath lifecycle tables. It intentionally excludes BIDE's own results and private traces; unavailable diagnostic fields are shown as `—`.

## Installation

Python 3.10+ is recommended:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e BIDE
```

Some retrieval families require local embedding or reranker services. Model paths and service credentials are external inputs and are not included here.

## Smoke checks

```bash
python -m compileall -q \
  BIDE/src BIDE/scripts \
  unified_evaluation/raw_eval unified_evaluation/MEMPATH \
  unified_evaluation/analysis

cd BIDE
PYTHONPATH=src python -m pytest -q tests
```

Use a small dataset slice or the checked-in fixture first. Full experiments require the original dataset, local model services, and a judge endpoint.

## Paper result status

The public code bundle includes the final BIDE pipeline contract and the verified
LoCoMo headline counts documented in `BIDE/docs/MAIN_EXPERIMENT_IMPLEMENTATION.md`.
It does not redistribute BIDE private records, request/response logs, frozen
per-question outputs, ablation tables, MEMPATH alignment annotations, or paper
figures. The legacy viewer contains benchmark demonstration data only and is not
a BIDE result archive. Publish those materials separately only after checking
dataset redistribution and privacy permissions.

## Data and reproducibility

Raw datasets, model weights, credentials, and large generated runs are external inputs. Runtime outputs should be written to a separate run directory. A publishable run should include configuration, source hashes, model identity, judge identity, and output checksums. Never commit `.env` files, API keys, request caches, or unreviewed model responses.

Results should be interpreted together with the dataset split, Reader configuration, judge protocol, and provenance manifest. The current main table must use a matching dataset, Reader configuration, judge protocol, and provenance manifest.

## License and citation

Add the project license and citation record before publishing this directory as an independent repository. Keep third-party licenses and upstream attribution with each adapter or frontend component.


Use `configs/settings.example.yaml` as the single local settings template. Copy it to `settings.local.yaml`, edit dataset/model/service fields, and keep the local file ignored by Git. No provider-specific endpoint or credential is required by the public release.
