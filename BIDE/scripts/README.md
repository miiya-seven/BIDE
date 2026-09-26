# Main experiment utilities

Only utilities used by the public main implementation are kept here:

- `doctor.py`: validate local runtime prerequisites;
- `check_contracts.py`: validate stage input contracts;
- `build_manifests.py`: build provenance manifests;
- `audit_runtime.py`: inspect runtime paths and configuration;
- `judge_locomo_official.py`: run the configured judge;
- `run_downstream_gpt54_full.sh`: launch the documented downstream pipeline;
- `serve_embedding_http.py` and `serve_reranker_http.py`: optional local model services.

One-off ablations, debugging scripts, and generated run files are excluded from
this public component.
