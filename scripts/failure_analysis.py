# -*- coding: utf-8 -*-
"""Mine every run's checkpoint logs + debug.log to classify why it failed.
Read-only: prints a per-run failure report to stdout."""
import json, re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# (folder glob, paper-label prefix, label offset)
GROUPS = [
    ("allRuns/gpt-5.4", "L", -1),        # run_01 -> L00
    ("allRuns/gpt-5.4-mini", "M", 0),    # run_01 -> M01
    ("tristanRuns", "T", 0),             # run_01 -> T01  (DeepSeek)
    ("codexRuns", "C", 0),               # run_01 -> C01
]

ERR_PATTERNS = [
    "TimeoutError", "MemoryError", "KeyError", "RuntimeError",
    "FileNotFoundError", "ValueError", "AssertionError", "TypeError",
    "IndexError", "Killed", "MemoryError", "OutOfMemory", "OSError",
]
SPECIAL = ["ml-100k.zip", "test_size=0.0", "validation_size=0.0", "test_size=0",
           "cannot reshape", "No such file"]

def run_dirs(folder):
    p = ROOT / folder
    return sorted([d for d in p.iterdir() if d.is_dir() and d.name.startswith("run_")],
                  key=lambda d: d.name)

def debug_end_status(dbg):
    """Classify how the debug.log ends: clean (next-run/finished marker) vs abrupt."""
    if not dbg.exists():
        return "no debug.log", ""
    lines = [l for l in dbg.read_text(errors="ignore").splitlines() if l.strip()]
    last = lines[-1] if lines else ""
    tail = "\n".join(lines[-4:])
    if re.search(r"Starting run \d+/|Finished all \d+ runs", tail):
        return "CLEAN (batch advanced to next run)", last[:90]
    if re.search(r"Selecting best buggy node|Writing code to agent file|interpreter: Done", last):
        return "ABRUPT (ends mid-iteration, no error/marker)", last[:90]
    return "OTHER", last[:90]

def scan_run(d):
    ck = d / "checkpoint"
    n_nodes = len([x for x in ck.iterdir() if x.is_dir()]) if ck.exists() else 0
    errs = {}
    timeouts = 0
    for logf in ck.glob("*/out.log") if ck.exists() else []:
        txt = logf.read_text(errors="ignore")
        if "TimeoutError" in txt or "timed out" in txt:
            timeouts += 1
        for pat in ERR_PATTERNS:
            if pat in txt:
                errs[pat] = errs.get(pat, 0) + 1
        for pat in SPECIAL:
            if pat in txt:
                errs[pat] = errs.get(pat, 0) + 1
    status, last = debug_end_status(d / "debug.log")
    return n_nodes, timeouts, errs, status, last

print(f"{'RUN':<6}{'nodes':<6}{'TO':<4}{'end-status':<44}error signatures")
print("-" * 120)
for folder, prefix, off in GROUPS:
    for i, d in enumerate(run_dirs(folder)):
        num = int(d.name.split("_")[1]) + off
        label = f"{prefix}{num:02d}"
        n_nodes, timeouts, errs, status, last = scan_run(d)
        sig = ", ".join(f"{k}×{v}" for k, v in sorted(errs.items(), key=lambda x: -x[1]))
        print(f"{label:<6}{n_nodes:<6}{timeouts:<4}{status:<44}{sig}")
    print()
