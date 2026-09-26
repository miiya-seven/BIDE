# Unified evaluation and analysis

This component provides the shared LoCoMo/LongMemEval evaluation runner,
external-method adapter runtime, MEMPATH diagnostics, and result analysis.

- `raw_eval/`: current runners and judge entry points.
- `map_platform/`: normalized adapter interfaces and built-in adapters.
- `MEMPATH/`: diagnostic metrics.
- `analysis/`: result aggregation and audits.
- `configs/`: non-secret examples.

To add an external memory method, implement the adapter contract documented in
`map_platform/README.md`, register it, and run:

```bash
python raw_eval/run_memory_eval.py \
  --dataset locomo \
  --data_path /path/to/locomo.json \
  --systems your_method \
  --output_dir outputs/your_method
```


Use `configs/settings.example.yaml` as the single local settings template. Copy it to `settings.local.yaml`, edit dataset/model/service fields, and keep the local file ignored by Git. No provider-specific endpoint or credential is required by the public release.
