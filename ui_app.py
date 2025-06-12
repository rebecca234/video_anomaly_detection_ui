import streamlit as st
import tempfile
import os
import cv2
import torch
import numpy as np
from collections import deque
from typing import List
 
# === Constants ===
MODEL_PATH = "D:\study\SEM-4\CapstoneProject\model3.pt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
 
# === Streamlit App ===
st.set_page_config(page_title="Anomaly Detection Demo", layout="wide")
 
@st.cache_resource
def load_model(path: str = MODEL_PATH):
    """Load and cache the STCN model once."""
    from main import STCN  # adjust if your STCN class lives elsewhere
    model = STCN(input_channels=3, sequence_length=16, height=64, width=64)
    state_dict = torch.load(path, map_location=DEVICE)
    model.load_state_dict(state_dict, strict=False)
    model.to(DEVICE).eval()
    return model
 
model = load_model()
 
st.title("🎥 Video Anomaly Detection Demo")
st.markdown(
    """
    Upload a video clip and press **Detect Anomaly**.  
    The model processes every 30‑frame chunk and reports the fraction flagged as anomalous.
    """
)
 
# Slider to tune your decision threshold live
threshold = 0.50
 
uploaded = st.file_uploader("Upload a video file", type=["mp4","avi","mov"])
if not uploaded:
    st.info("Please upload a video to start.")
    st.stop()
 
# Write upload to a temp file once
suffix = os.path.splitext(uploaded.name)[1]
with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_vid:
    tmp_vid.write(uploaded.read())
    tmp_path = tmp_vid.name
 
st.video(tmp_path)
 
# In‑memory extraction + preprocessing
def extract_and_preprocess(video_path: str, size=(64,64), fps_interval=1):
    """
    Returns a list of normalized frames [H,W,3] at 30fps/fps_interval.
    """
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS) or 30)
    interval = max(1, fps // fps_interval)
    frames = []
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % interval == 0:
            frame = cv2.resize(frame, size)
            frames.append(frame.astype(np.float32) / 255.0)
        idx += 1
    cap.release()
    return frames

def make_sequences(frames, seq_len=16, stride=8):
    seqs = []
    for i in range(0, len(frames) - seq_len + 1, stride):
        chunk = np.stack(frames[i : i + seq_len], axis=0)
        seqs.append(chunk)
    return seqs

def infer_sequences(model, sequences: List[np.ndarray], device, seq_thresh: float):
    # … build tensor …
    """
    Batch all sequences through the model, return List[0/1].
    """
    # Convert to torch Tensor [N,3,16,64,64]
    arr = np.stack(sequences, axis=0)  # [N,T,H,W,3]
    tensor = torch.from_numpy(arr).permute(0,4,1,2,3).to(device)
    with torch.no_grad():
        probs = model(tensor).squeeze().cpu().numpy()
    seq_preds = (probs > seq_thresh)
    return seq_preds, probs


if st.button("🔍 Detect Anomaly"):
    # 0) sliders first
    # seq_thresh   = st.slider("Chunk cutoff",   0.0, 1.0, 0.25, 0.01)
    seq_thresh   = 0.25
    # video_thresh = st.slider("Video cutoff",   0.0, 1.0, 0.05, 0.01)

    with st.spinner("Processing video..."):
        # 1) frames → overlapping sequences
        frames    = extract_and_preprocess(tmp_path, size=(64,64), fps_interval=30)
        sequences = make_sequences(frames, seq_len=16, stride=8)
        n = len(sequences)
        if n == 0:
            st.error("Video too short for 16‑frame chunks.")
            st.stop()

        # 2) run inference with progress bar
        progress     = st.progress(0)
        preds        = []   # binary predictions
        all_probs    = []   # raw probabilities
        batch_size   = 32

        for start in range(0, n, batch_size):
            end    = min(start + batch_size, n)
            batch  = sequences[start:end]
            b_preds, b_probs = infer_sequences(model, batch, DEVICE, seq_thresh)
            preds.extend(b_preds)
            all_probs.extend(b_probs)
            progress.progress(end / n)

        # 3) compute video‑level score
        anomaly_prob = sum(preds) / n
        # st.write(f"**Anomaly probability:** {anomaly_prob:.1%}")

        # flag if ANY chunk is anomalous or if fraction > video_thresh
        if any(preds) or (anomaly_prob > threshold):
            st.error("🚨 Anomaly Detected!")
        else:
            st.success("✅ No Anomaly Detected")

        # 4) optionally show the first anomalous snippet
        if any(preds):
            idx = preds.index(1)
            clip = frames[idx*16 : idx*16 + 16]
            clip_path = os.path.join(tempfile.gettempdir(), "anomaly_clip.mp4")
            writer = cv2.VideoWriter(
                clip_path, cv2.VideoWriter_fourcc(*"mp4v"), 30,
                (clip[0].shape[1], clip[0].shape[0])
            )
            for frm in clip:
                writer.write((frm * 255).astype(np.uint8)[..., ::-1])
            writer.release()
            st.video(clip_path)