import wandb
import re
import ast
import time
from tqdm import tqdm

# --- CONFIG ---
ENTITY = "vdaita"    # e.g. "vjcodes"
PROJECT = "sar_transformer"
LOG_FILENAME = "output.log"
SLEEP = 1.0  # seconds between runs

api = wandb.Api()

# Robust log file fetcher (old parser)
def fetch_log_text(run):
    files = {f.name: f for f in run.files()}
    candidates = [
        "output.log",
        "logs/output.log",
        "files/output.log",
        "root/output.log",
        "media/output.log",
    ]
    for name in candidates:
        if name in files:
            print(f"  📂 Found {name}")
            f = files[name]
            local_path = f.download(replace=True).name
            with open(local_path, "r", encoding="utf-8", errors="ignore") as fh:
                return fh.read()
    raise FileNotFoundError(f"No output.log found for {run.id}")

# Parse step logs from output.log
def parse_step_metrics(log_text):
    # remove ANSI escape codes
    log_text = re.sub(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])", "", log_text)
    # find "Step X: {...}"
    matches = re.findall(r"Step\s+(\d+):\s+(\{.*?\})", log_text)
    parsed = []
    for step_str, dict_text in matches:
        step = int(step_str)
        try:
            data = ast.literal_eval(dict_text)
        except Exception:
            continue
        parsed.append((step, data))
    return parsed


def recover_run(run):
    print(f"\n🧩 Recovering run: {run.name} ({run.id})")
    try:
        log_text = fetch_log_text(run)
    except FileNotFoundError as e:
        print(f"  ⚠️ {e}")
        return

    step_metrics = parse_step_metrics(log_text)
    if not step_metrics:
        print("  ⚠️ No metrics found in log.")
        return

    # New run with recovered_<original_name>
    new_run_name = f"recovered_{run.name}"
    recovered_run = wandb.init(
        entity=ENTITY,
        project=PROJECT,
        name=new_run_name,
        reinit=True,
    )

    count = 0
    for step, data in tqdm(step_metrics, desc=new_run_name):
        # Only keep eval metrics
        data_eval = {k: v for k, v in data.items() if k.startswith("eval/")}
        if not data_eval:
            continue
        wandb.log(data_eval, step=step)
        count += 1

    recovered_run.finish()
    print(f"✅ Logged {count} eval metric points to {new_run_name}")
    time.sleep(SLEEP)


if __name__ == "__main__":
    for run in api.runs(f"{ENTITY}/{PROJECT}"):
        recover_run(run)

print("🎉 All runs processed!")
