import os

from auxiliary.utils import get_device

# --- Random seed (for reproducibility) ---

RANDOM_SEED = 0

# --- Device (cpu or cuda:n) ---

DEVICE_TYPE = "cuda:0"
DEVICE = get_device(DEVICE_TYPE)

# --- PATHS ---

BASE_PATH = os.path.join("/media", "matteo", "Extreme SSD")
PATH_TO_DATASET = os.path.join(BASE_PATH, "dataset", "ccc")
PATH_TO_PRETRAINED = os.path.join(BASE_PATH, "models", "faithful-attention-eval")
PATH_TO_RESULTS = os.path.join(BASE_PATH, "results", "faithful-attention-eval", "tests")
PATH_TO_PLOTS = "plots"
