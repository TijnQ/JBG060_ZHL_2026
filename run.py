"""Root execution file for Northern Bahr el Ghazal Flood Modeling.

Delegates execution to LightGBM_v1/run.py.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add repository root to sys.path
root_dir = Path(__file__).resolve().parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from LightGBM_v1.run import main

if __name__ == "__main__":
    main()
