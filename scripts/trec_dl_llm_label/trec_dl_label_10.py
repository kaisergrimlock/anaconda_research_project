# trec_dl_llm_label/trec_dl_label_10.py
import sys
import time
import subprocess
from pathlib import Path
from datetime import datetime

# -------- Settings --------
RUNS = 10               # how many times to run
SLEEP_BETWEEN = 2       # seconds between runs (helps with rate limits)
STOP_ON_ERROR = False   # set True to abort if a run fails

# Paths
BASE_DIR  = Path(__file__).resolve().parent                  # .../trec_dl_llm_label
ROOT_DIR  = BASE_DIR.parent                                  # project root
SCRIPT    = BASE_DIR / "trec_dl_label.py"                    # the script to run
LOG_DIR   = ROOT_DIR / "outputs" / "trec_dl" / "run_logs"    # where we keep per-run logs
PROMPT_SRC = BASE_DIR / "prompt.txt"                         # your prompt next to this file
PROMPT_DST = ROOT_DIR / "prompt.txt"                         # script expects PROMPT_FILE="prompt.txt"

LOG_DIR.mkdir(parents=True, exist_ok=True)

def ts():
    return datetime.now().strftime("%Y%m%d_%H%M%S")

# Ensure the called script can find prompt.txt when cwd=ROOT_DIR
if PROMPT_SRC.exists():
    PROMPT_DST.write_text(PROMPT_SRC.read_text(encoding="utf-8"), encoding="utf-8")

for i in range(1, RUNS + 1):
    stamp = ts()
    print(f"[{stamp}] Starting run {i}/{RUNS} -> {SCRIPT.name}")

    log_path = LOG_DIR / f"{stamp}_trec_dl_label_run_{i}.log"
    with log_path.open("w", encoding="utf-8") as lf:
        # Run using the current Python interpreter; cwd=ROOT_DIR so all relative paths in trec_dl_label.py work
        proc = subprocess.run([sys.executable, str(SCRIPT)],
                              cwd=ROOT_DIR,
                              stdout=lf,
                              stderr=subprocess.STDOUT)

    exit_code = proc.returncode
    print(f"[{ts()}] Run {i} finished with code {exit_code}; log: {log_path}")

    if exit_code != 0 and STOP_ON_ERROR:
        print("Stopping due to error.")
        break

    if i < RUNS and SLEEP_BETWEEN > 0:
        time.sleep(SLEEP_BETWEEN)

print("All done.")
