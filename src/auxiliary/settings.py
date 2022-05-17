import os

from src.auxiliary.utils import get_device

# --- Random seed (for reproducibility) ---

RANDOM_SEED = 0

# --- Device (cpu or cuda:n) ---

DEVICE_TYPE = "cpu"
DEVICE = get_device(DEVICE_TYPE)

# --- PATHS ---

PROJECT_NAME = "faithful-attention-eval"
BASE_PATH = os.path.join("/media", "matteo", "Extreme SSD")
PATH_TO_DATASET = os.path.join(BASE_PATH, "dataset", "ccc")
PATH_TO_PRETRAINED = os.path.join(BASE_PATH, "models", PROJECT_NAME)
PATH_TO_RESULTS: str = os.path.join(BASE_PATH, "results", PROJECT_NAME)
PATH_TO_PLOTS = "plots"
DEFAULT_METRICS_FILE = "metrics.csv"
