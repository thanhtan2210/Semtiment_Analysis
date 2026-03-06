import sys
from pathlib import Path

# Ensure Semtiment_Analysis directory is on sys.path when tests are run from repo root
PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))
