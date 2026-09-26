# Evaluation adapter runtime

The runner loads systems through `memory_systems.registry.build_memory_adapter`.
An external memory method must implement the `MemoryAdapter` contract in
`memory_systems/base.py`:

- `validate_setup()` reports whether dependencies and credentials are available;
- `build_memory(sample)` ingests one conversation/sample;
- `get_memory_entries()` returns normalized `MemoryEntry` objects;
- `retrieve(question, sample, top_k)` returns `RetrievalResult` with source IDs;
- `build_prompt(question, sample, retrieval)` returns a `PromptRecord`;
- `answer(prompt, sample)` returns the generated answer and usage metadata.

Register the adapter in `memory_systems/registry.py`, add a profile in the
runner's `SYSTEM_PROFILES`, and pass its name with `--systems`. Keep external
credentials in environment variables and preserve original source IDs so that
retrieval recall and provenance metrics remain meaningful.

## Generic external adapter

For a system that cannot be imported into Python, expose two JSON HTTP routes:

```text
POST /build    {sample_id, history} -> {entries: [...]}
POST /retrieve {sample_id, query, top_k} -> {retrieved: [...]}
```

Or provide `EXTERNAL_MEMORY_ARTIFACT`, pointing to a JSON/JSONL file or a
 directory containing `<sample_id>.jsonl`. Each entry should contain `content`
and, whenever possible, `source_ids` pointing to original conversation turns.

Then run:

```bash
export EXTERNAL_MEMORY_URL=http://127.0.0.1:9000
python raw_eval/run_memory_eval.py --dataset locomo \
  --data_path /path/to/locomo.json --systems external \
  --output_dir outputs/external
```

This path covers hosted memory APIs, separate services, precomputed stores, and
Python implementations wrapped by a small HTTP shim. The evaluator remains
responsible for prompt injection, answer generation, source tracing, judging,
and summary metrics.
