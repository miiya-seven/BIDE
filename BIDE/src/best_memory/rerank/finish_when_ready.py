"""Evaluate the frozen output after the already-running inference process finishes."""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

H = Path(__file__).resolve().parent
a = argparse.ArgumentParser()
a.add_argument('--pid', type=int, required=True)
args = a.parse_args()
while not (H/'RANKINGS.json').exists():
    proc = Path(f'/proc/{args.pid}/cmdline')
    try:
        running = '274_full_rrf6_qwen8b/run.py' in proc.read_bytes().decode(errors='replace')
    except FileNotFoundError:
        running = False
    if not running:
        (H/'FINALIZATION.json').write_text(json.dumps({'status':'inference_stopped_without_rankings','pid':args.pid}))
        sys.exit(1)
    time.sleep(10)
with (H/'evaluation.log').open('w') as log:
    result = subprocess.run([sys.executable, str(H/'evaluate.py')], stdout=log, stderr=subprocess.STDOUT)
(H/'FINALIZATION.json').write_text(json.dumps({'status':'complete' if result.returncode==0 else 'evaluation_failed',
                                             'returncode':result.returncode}))
sys.exit(result.returncode)
