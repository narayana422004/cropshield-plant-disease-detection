# 🌱 CropShield — AI-Powered Plant Disease Detection

CropShield is a Streamlit web app that identifies plant diseases from a leaf
photo. It first verifies the uploaded image actually contains a leaf, then
classifies the disease across 38 classes spanning 14 crop species, and
returns treatment and prevention guidance.

**[Live Demo](#)** *(cropshield-plant-disease-detection-vshb6zdg9vsjxyqknwd2hw.streamlit.app)*

## Features

- 🌿 **Leaf validation** — rejects non-leaf images before running disease
  classification, using a MobileNetV3Small binary classifier.
- 🔬 **Disease classification** — MobileNetV2 fine-tuned on the
  [PlantVillage dataset](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset),
  38 classes, 97.0% test accuracy / 99.7% top-3 accuracy.
- 📊 **Top-3 predictions** with confidence scores.
- 💊 **Treatment & prevention tips** for every class, with severity ratings.
- 📜 **Prediction history**, persisted locally across sessions.
- 📄 **Downloadable PDF report** for any diagnosis.

## Tech Stack

- **Model training**: TensorFlow / Keras, transfer learning (MobileNetV2,
  MobileNetV3Small), trained on Google Colab.
- **App**: Streamlit
- **PDF generation**: ReportLab

## Project Structure

```
Cropshield/
├── app/
│   ├── app.py            # Streamlit entry point
│   ├── inference.py       # Model loading + prediction logic
│   ├── report.py          # PDF report generation
│   └── utils.py            # Formatting helpers
├── config/
│   └── settings.py        # Paths, thresholds, image sizes
├── models/                # Trained .keras model files
├── metadata/               # Class labels + treatment tips
├── notebooks/
│   └── CropShield_Industry_Training_Final.ipynb
└── requirements.txt
```

## Running Locally

```bash
git clone https://github.com/AAliAhmedd/cropshield-plant-disease-detection.git
cd cropshield
python -m venv .venv
.venv\Scripts\activate      # Windows
pip install -r requirements.txt
streamlit run app/app.py
```

## Model Details

| Model | Architecture | Test Accuracy | Notes |
|---|---|---|---|
| Leaf Validator | MobileNetV3Small (binary) | 100%* | *Trained against CIFAR-10 negatives — see limitations below |
| Disease Classifier | MobileNetV2 (fine-tuned) | 97.04% | Stratified 80/10/10 split, 5,431 test images |

## Known Limitations

- The leaf validator's near-perfect internal accuracy is partly an artifact
  of its training setup: "not-leaf" negative examples were drawn from
  CIFAR-10 (upscaled 32×32 images), which are trivially distinguishable
  from clean PlantVillage leaf photos. Real-world non-leaf inputs (hands,
  household objects at normal photo quality) haven't been validated
  against it. Retraining with realistic negative examples is a planned
  improvement.
- The disease classifier is trained and evaluated entirely on PlantVillage,
  a lab-quality dataset with plain backgrounds. Accuracy on cluttered,
  real-world field photos has not yet been independently measured.
  # 👨‍💻 Author

**Ali Ahmed**

Machine Learning Intern  
BS Artificial Intelligence Student

---

⭐ If you found this project useful, consider giving the repository a **Star**!

## License

MIT