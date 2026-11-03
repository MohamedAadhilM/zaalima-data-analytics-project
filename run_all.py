"""Run the whole project end to end: Week 1 -> Week 4 (every day), the unit tests, and write a run log.

    python run_all.py            # full run
    python run_all.py --fresh    # delete generated data/results first, then rebuild everything
"""
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
STEPS = [
    "week1_data_dag_baseline.py",
    "week2_day1_donor_pool.py", "week2_day2_synthetic_control.py", "week2_day3_mspe_validation.py",
    "week2_day4_double_ml.py", "week2_day5_midproject_review.py",
    "week3_day1_ate_estimators.py", "week3_day2_cate_segments.py", "week3_day3_scm_inference.py",
    "week3_day4_sensitivity.py", "week3_day5_cross_validation.py",
    "week4_day1_counterfactual_diagnostics.py", "week4_day2_whatif_engine.py",
    "week4_day3_dashboard_smoke_test.py", "week4_day4_executive_report.py", "week4_day5_final_review.py",
]


def main() -> int:
    if "--fresh" in sys.argv:
        for d in ("data/raw", "data/processed", "results", "artifacts", "reports/figures"):
            shutil.rmtree(ROOT / d, ignore_errors=True)
        for f in (ROOT / "reports").glob("*.md"):
            f.unlink()
    env = {**os.environ, "MPLBACKEND": "Agg", "PYTHONWARNINGS": "ignore"}
    log, failed = [], []
    for step in STEPS:
        t = time.time()
        r = subprocess.run([sys.executable, str(ROOT / "weeks" / step)], cwd=ROOT / "weeks", env=env,
                           capture_output=True, text=True)
        ok = r.returncode == 0
        log.append(f"{'OK  ' if ok else 'FAIL'} {step:48s} {time.time() - t:6.1f}s")
        print(log[-1], flush=True)
        if not ok:
            failed.append(step); print(r.stderr[-1500:])
            break
    t = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", str(ROOT / "tests")],
                       cwd=ROOT, env=env, capture_output=True, text=True)
    log.append("OK   pytest: " + (t.stdout.strip().splitlines() or [""])[-1] if t.returncode == 0
               else "FAIL pytest\n" + t.stdout[-1500:])
    print(log[-1])
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "run_log.txt").write_text("\n".join(log) + "\n")
    return 1 if failed or t.returncode else 0


if __name__ == "__main__":
    sys.exit(main())
