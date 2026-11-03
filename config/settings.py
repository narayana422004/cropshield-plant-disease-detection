from pathlib import Path


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_DIR = PROJECT_ROOT / "models"
METADATA_DIR = PROJECT_ROOT / "metadata"


# ============================================================
# MODEL FILES
# ============================================================

LEAF_VALIDATOR_MODEL = (
    MODEL_DIR / "cropshield_leaf_validator.keras"
)

DISEASE_MODEL = (
    MODEL_DIR / "cropshield_mobilenetv2_finetuned.keras"
)


# ============================================================
# METADATA FILES
# ============================================================

CLASS_INDICES_FILE = (
    METADATA_DIR / "class_indices.json"
)

LEAF_VALIDATOR_METADATA_FILE = (
    METADATA_DIR / "leaf_validator_metadata.json"
)


# ============================================================
# IMAGE SETTINGS
# ============================================================

LEAF_IMAGE_SIZE = (128, 128)

DISEASE_IMAGE_SIZE = (224, 224)


# ============================================================
# PREDICTION SETTINGS
# ============================================================

LEAF_THRESHOLD = 0.50

DISEASE_CONFIDENCE_THRESHOLD = 0.60

TOP_K_PREDICTIONS = 3


# ============================================================
# APPLICATION
# ============================================================

APP_TITLE = "CropShield"

APP_SUBTITLE = (
    "AI-Powered Plant Disease Detection"
)