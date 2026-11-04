"""Runs a list of experiment commands with bounded concurrency, logging every outcome.

Each job is `name|command`; output goes to runs/logs/<name>.log and one JSON line per finished
job (exit code, wall time) is appended to runs/logs/queue_status.jsonl, so failed runs remain
on record instead of disappearing.

Usage:  python experiments/run_queue.py <jobs.txt> <max_parallel>
"""
from __future__ import annotations
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOGS = ROOT / "runs" / "logs"


def main():
    jobs = [ln.split("|", 1) for ln in Path(sys.argv[1]).read_text().splitlines()
            if ln.strip() and not ln.startswith("#")]
    width = int(sys.argv[2])
    LOGS.mkdir(parents=True, exist_ok=True)
    pending, running = list(jobs), []
    while pending or running:
        while pending and len(running) < width:
            name, cmd = pending.pop(0)
            log = open(LOGS / f"{name.strip()}.log", "w")
            p = subprocess.Popen(cmd.strip(), shell=True, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            running.append((name.strip(), cmd.strip(), p, time.time(), log))
            print(f"start {name.strip()}", flush=True)
        for item in list(running):
            name, cmd, p, t0, log = item
            if p.poll() is not None:
                log.close()
                rec = {"name": name, "cmd": cmd, "exit": p.returncode, "wall_s": round(time.time() - t0, 1)}
                with open(LOGS / "queue_status.jsonl", "a") as f:
                    f.write(json.dumps(rec) + "\n")
                print(json.dumps(rec), flush=True)
                running.remove(item)
        time.sleep(5)


if __name__ == "__main__":
    main()
