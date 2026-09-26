# BIDE evaluation viewer

This is the static legacy viewer data and source used to display system
profiles, mechanisms, funnels, and sample traces. The checked-in public bundle
is a LoCoMo-only baseline comparison aligned with the paper tables (n=1540),
including the MemPath lifecycle coverage values where reported.
BIDE's own headline results, private records, and experiment traces are not
included. Diagnostic fields that are not reported in the paper are represented
as unavailable rather than estimated. Regenerate the bundle with
`scripts/build_data.py` only from a reviewed, public evaluation export.
