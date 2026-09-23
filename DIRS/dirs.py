from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent

RESULTS_DIR = PROJECT_DIR / "results"
MODEL_PATH = PROJECT_DIR / "model" / "model.pth"
