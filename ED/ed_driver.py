"""
Example driver that loops over every CSV found in ED/configs/
and saves each run to ED/output/.
"""

from pathlib import Path
#from ed_solver import run_ed
from ed_solver1d import run_ed

cfg_dir = Path("ED/configs")          # put many config CSVs here
for cfg in sorted(cfg_dir.glob("*.csv")):
    print(f"\n=== Running {cfg.name} ===")
    outfile = run_ed(cfg)             # uses default out_dir
    print(f"→ saved to {outfile}")
