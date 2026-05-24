"""
Run this in your paper folder to see the exact LaTeX error.
It reads the last 50 lines of main.log
"""
from pathlib import Path

log = Path(r"C:\Users\User\Documents\Projects\earth2research\paper\main.log")
if log.exists():
    lines = log.read_text(encoding="utf-8", errors="ignore").splitlines()
    # Find error lines
    errors = [(i, l) for i, l in enumerate(lines) if l.startswith("!") or "Error" in l or "error" in l]
    print("=== ERROR LINES ===")
    for i, l in errors[:20]:
        print(f"Line {i}: {l}")
    print("\n=== LAST 60 LINES OF LOG ===")
    for l in lines[-60:]:
        print(l)
else:
    print("main.log not found")
