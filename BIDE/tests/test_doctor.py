from pathlib import Path
import subprocess, sys
ROOT=Path(__file__).resolve().parents[1]
def test_doctor_reports_without_crashing():
    p=subprocess.run([sys.executable,str(ROOT/'scripts/doctor.py')],cwd=ROOT,text=True,capture_output=True)
    assert 'dataset' in p.stdout
