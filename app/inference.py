import json
from pathlib import Path

import numpy as np
import tensorflow as tf
import keras
from keras import layers

from config.settings import (
    LEAF_VALIDATOR_MODEL,
    DISEASE_MODEL,
    CLASS_INDICES_FILE,
    METADATA_DIR,
    LEAF_IMAGE_SIZE,
    DISEASE_IMAGE_SIZE,
    LEAF_THRESHOLD,
    DISEASE_CONFIDENCE_THRESHOLD,
    TOP_K_PREDICTIONS,
)

TREATMENT_TIPS_FILE = METADATA_DIR / "treatment_tips.json"


# --------------------------------------------------
# Compatibility patch for models saved with a newer
# Keras version that added `quantization_config` to
# Dense layers. custom_objects lookups by plain "Dense"
# don't reliably override the built-in class during
# deserialization, so we patch Dense.__init__ directly.
# --------------------------------------------------

_original_dense_init = layers.Dense.__init__


def _patched_dense_init(self, *args, quantization_config=None, **kwargs):
    _original_dense_init(self, *args, **kwargs)


layers.Dense.__init__ = _patched_dense_init


# --------------------------------------------------
# Main inference wrapper used by app.py
# --------------------------------------------------

class CropShieldInference:
    def __init__(self):
        print(f"Loading leaf validator from: {LEAF_VALIDATOR_MODEL}")
        self.leaf_validator_model = keras.models.load_model(
            LEAF_VALIDATOR_MODEL,
            safe_mode=False,
        )
        print("Leaf validator loaded successfully.")

        print(f"Loading disease model from: {DISEASE_MODEL}")
        self.disease_model = keras.models.load_model(
            DISEASE_MODEL,
            safe_mode=False,
        )
        print("Disease model loaded successfully.")

        with open(CLASS_INDICES_FILE, "r") as f:
            raw = json.load(f)
        # keys are strings like "0", "1", ... -> sort numerically
        self.class_names = [raw[str(i)] for i in range(len(raw))]

        with open(TREATMENT_TIPS_FILE, "r") as f:
            self.treatment_tips = json.load(f)

    # ----------------------------------------
    # Helpers
    # ----------------------------------------

    def _load_image_array(self, uploaded_file):
        """Decode a Streamlit UploadedFile into a uint8/float array."""
        uploaded_file.seek(0)
        image = tf.io.decode_image(
            uploaded_file.read(),
            channels=3,
            expand_animations=False,
        )
        return image

    def _preprocess(self, image, size):
        image = tf.image.resize(image, size, method="bilinear")
        image = tf.cast(image, tf.float32)
        image = tf.expand_dims(image, axis=0)
        return image

    def _get_treatment_info(self, class_name):
        return self.treatment_tips.get(
            class_name,
            {
                "severity": "unknown",
                "treatment": ["No treatment information available for this class yet."],
                "prevention": [],
            },
        )

    # ----------------------------------------
    # Public API used by app.py
    # ----------------------------------------

    def predict(self, uploaded_file):
        image = self._load_image_array(uploaded_file)

        # ---- Step 1: leaf validation ----
        leaf_input = self._preprocess(image, LEAF_IMAGE_SIZE)
        leaf_pred = self.leaf_validator_model.predict(leaf_input, verbose=0)
        leaf_probability = float(np.squeeze(leaf_pred))

        if leaf_probability < LEAF_THRESHOLD:
            return {
                "status": "not_leaf",
                "message": (
                    "The uploaded image does not appear to contain a "
                    "plant leaf. Please upload a clearer photo of a leaf."
                ),
                "leaf_probability": leaf_probability,
            }

        # ---- Step 2: disease classification ----
        disease_input = self._preprocess(image, DISEASE_IMAGE_SIZE)
        disease_pred = self.disease_model.predict(disease_input, verbose=0)
        disease_pred = np.squeeze(disease_pred)

        top_index = int(np.argmax(disease_pred))
        confidence = float(disease_pred[top_index])
        predicted_class = self.class_names[top_index]

        # ---- Top-K predictions ----
        top_k_indices = np.argsort(disease_pred)[::-1][:TOP_K_PREDICTIONS]
        top_3 = [
            {
                "class_name": self.class_names[int(i)],
                "confidence": float(disease_pred[int(i)]),
            }
            for i in top_k_indices
        ]

        if confidence < DISEASE_CONFIDENCE_THRESHOLD:
            return {
                "status": "low_confidence",
                "message": (
                    "A leaf was detected, but the model isn't confident "
                    "enough in the disease prediction."
                ),
                "leaf_probability": leaf_probability,
                "predicted_class": predicted_class,
                "confidence": confidence,
                "top_3": top_3,
                "treatment_info": self._get_treatment_info(predicted_class),
            }

        return {
            "status": "success",
            "leaf_probability": leaf_probability,
            "predicted_class": predicted_class,
            "confidence": confidence,
            "top_3": top_3,
            "treatment_info": self._get_treatment_info(predicted_class),
        }