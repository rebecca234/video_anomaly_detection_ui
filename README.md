# video_anomaly_detection_ui
Implementing the Video anomaly detection in a user interface
# video_anomaly_detection_ui
Implementing the Video anomaly detection in a user interface

## Overview

This project is an end-to-end pipeline for detecting anomalies (e.g., criminal events) in surveillance video. It uses a Spatio-Temporal Convolutional Network (STCN) to learn normal motion–appearance patterns and flag deviations as anomalies. The system includes data preprocessing, model training, evaluation, inference, and an interactive Streamlit UI for live demos.

---

## Repository Structure

```
CrimeScope/
├── data/
│   ├── annotations/                      # Temporal_Anomaly_Annotation_for_Testing_Videos.txt
│   ├── frames/                           # Extracted frame folders
│   ├── preprocessed_frames/              # Resized & normalized frames
│   ├── sequences/                        # Pickled 16-frame sequences
│   └── labeled_sequences/                # Pickled (sequences, labels)
├── models/
│   └── model3.pt                         # Trained STCN checkpoint
├── src/
│   ├── main.py                           # Preprocessing, model, training, evaluation
│   └── ui_app.py                         # Streamlit demo application
├── README.md                             # This file
└── requirements.txt                      # Python dependencies
```

---

## Requirements

* Python 3.8+
* PyTorch 1.12+
* CUDA Toolkit (if using GPU)
* OpenCV, NumPy, pandas, scikit-learn, tqdm, seaborn
* Streamlit

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Data Preparation

1. **Download** the UCF-Crimes dataset and unzip into `data/`. (url to datset: `https://www.kaggle.com/datasets/minhajuddinmeraj/anomalydetectiondatasetucf?resource=download`)
2. **Annotations**: place `Temporal_Anomaly_Annotation_for_Testing_Videos.txt` in `data/annotations/`.

---

## Preprocessing Pipeline

Run `src/main.py` to:

1. **Extract frames** at 30 fps into `data/frames/`.
2. **Resize & normalize** frames to 64×64 → `data/preprocessed_frames/`.
3. **Sequence generation**: 16‑frame windows, stride 8 → pickled in `data/sequences/`.
4. **Labeling**: using temporal annotations → saved in `data/labeled_sequences/`.


```bash
python src/main.py --prepare-data
```

---

## Training the STCN

Train with balanced oversampling, weighted BCE, gradient clipping, LR scheduling, and early stopping on val F1:

```bash
python src/main.py --train --epochs 20 --batch-size 8 --lr 1e-3
```

Checkpoints saved to `models/`.

---

## Evaluation

Compute metrics and visualizations:

```bash
python src/main.py --evaluate --model-path models/model3.pt
```

Generates:

* Confusion matrices
* Precision/Recall/F1
* ROC-AUC curves

---

## Inference & Post-Processing

Use `src/inference.py` to run on new video:

1. Load model and weights
2. Extract & preprocess frames
3. Generate sequences and predict probabilities
4. Apply temporal smoothing and ensemble
5. Output clip‑level and video‑level anomaly scores

---

## Streamlit UI Demo

Launch the interactive app:

```bash
streamlit run src/ui_app.py
```

### Here is are sample screenshots of the user interface.

![alt text](image.png)

![alt text](image-1.png)

Features:

* Video upload & playback
* Live threshold sliders
* Progress bar for processing
* Instant clip replay for anomalies

---

## Future Enhancements

* Integrate video transformers for long‑range context
* Self‑supervised pretraining on normal data
* Pixel‑level anomaly localization
* Online learning for concept drift
* Edge deployment via model quantization

---

## License & Citation

Please cite:

> *Rebecca Namala*, “A Spatio Temporal Vision for Real-Time Anomaly Detection”, 2025.

---
