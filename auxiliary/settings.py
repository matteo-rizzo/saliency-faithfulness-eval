from auxiliary.utils import get_device

# --- Random seed (for reproducibility) ---

RANDOM_SEED = 0

# --- Device (cpu or cuda:n) ---

DEVICE_TYPE = "cuda:0"
DEVICE = get_device(DEVICE_TYPE)

# --- PATHS ---

PATH_TO_DATASET = "/media/matteo/Extreme SSD/dataset/ccc"
PATH_TO_PRETRAINED = "trained_models"
