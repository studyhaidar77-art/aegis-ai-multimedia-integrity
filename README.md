# 🛡️ AegisAI — Multimedia Integrity Analyzer (Video + Photo)

AegisAI is a Streamlit-based **multimedia integrity and deepfake risk analyzer** that helps evaluate **suspect videos and photos** for potential manipulation (deepfake / face-swap / heavy edits) using a combination of:

- **Quality signals** (blur + noise → baseline risk)
- **Multi-face extraction** (video frames + photos)
- **Model-based deepfake scoring**
- **Optional identity verification** (Reference vs Suspect)
- **Evidence strength scoring**
- **Downloadable forensic reports (CSV)**

> ⚠️ **Disclaimer:** This tool provides an **AI-assisted assessment**, not legal proof. Results may be incorrect. Always verify with human experts and additional forensic methods.

---

## ✅ Live Demo
👉 Hugging Face Space: https://huggingface.co/spaces/alihaidar-ai/aegis-ai-multimedia-integrity-full
👉 Streamlit: https://aegis-ai-multimedia-integrity-4vypudbpyourhrkul5xu7h.streamlit.app/

---

## ✨ Features

### 🎞️ Suspect VIDEO Analysis
- Upload suspect video (`mp4`, `mov`, `avi`)
- Extract frames with slider control (1 frame every N frames)
- Compute per-frame evidence:
  - Blur score
  - Noise score
  - Baseline risk (0–100)
- Detect faces and show **face crops (224×224)**
- Optional multi-person check (if identity module is enabled)
- **Download CSV exports**
  - ✅ Video frames evidence CSV
  - ✅ Deepfake per-face scores CSV (video)

### 🖼️ Suspect PHOTO Analysis
- Upload 1+ suspect photos (`jpg`, `jpeg`, `png`)
- Detect faces and show **face crops (224×224)**
- Optional multi-person check (if identity module is enabled)
- **Download CSV export**
  - ✅ Deepfake per-face scores CSV (photos)

### 🪪 Reference Identity Verification (Optional)
- Upload reference **photos (1–5)** and/or **videos (1–3)**
- Builds a **reference identity embedding**
- Filters inconsistent reference selfies (outlier detection)
- Compares suspect faces vs reference (cosine similarity)
- Identity verdict:
  - `MATCH` / `UNCERTAIN` / `NOT MATCH`

### 📊 Final Forensic Risk + Smart Verdict
- Combines multiple signals into a final risk score:
  - Deepfake model probability
  - Baseline quality risk
  - Optional identity similarity
- Produces:
  - ✅ Final Forensic Risk (0–100)
  - ✅ Evidence Strength score (0–100)
  - ✅ Smart Verdict summary message

### 📥 Downloadable Report (CSV)
One-click downloadable report includes:
- `final_risk`
- `evidence_strength`
- `identity_similarity_best`
- `fake_score_best`
- `avg_blur`, `avg_baseline_risk`
- flags: identity enabled, reference provided, video uploaded
- face counts (video + photos)

---

## 🧠 Tech Stack
- **Python**
- **Streamlit**
- **OpenCV**
- **NumPy, Pandas**
- **Pillow**
- Deepfake model: **Hugging Face Transformers pipeline** (loaded via `utils_deepfake_model.py`)
- Optional identity matching: **InsightFace** (embeddings) via `utils_identity.py`

---

## 🗂️ Project Structure

```bash
.
├── app.py
├── streamlit_app.py              # HF entry file (imports app.py)
├── utils_video.py
├── utils_identity.py
├── utils_deepfake_model.py
├── requirements.txt
└── README.mdthis project useful, consider giving it a star!

Create environment
python -m venv venv
# Windows:
# venv\Scripts\activate
# Linux/Mac:
# source venv/bin/activate
2) Install dependencies
pip install -r requirements.txt
3) Run the app
streamlit run app.py
✅ How To Use

(Optional) Upload reference identity media:

Reference photos (1–5 clear selfies)

Reference videos (1–3 short real videos)

Upload suspect evidence:

Suspect video OR suspect photos OR both

Review outputs:

Face crops

Baseline risk

Deepfake probability

Identity similarity (if reference provided)

Final risk + evidence strength

Export CSV reports:

Report summary CSV

Video frames CSV (video only)

Deepfake per-face CSV (video/photo)

📌 Notes & Limitations

Performance depends on media quality (lighting, blur, resolution, compression).

If faces are too small or occluded, detection may fail.

Deepfake models can produce false positives/negatives.

Identity verification accuracy depends heavily on clean reference selfies/videos.

This is an assistive tool, not a final forensic authority.

🔮 Future Improvements

Per-frame deepfake timeline chart

Better face tracking across frames

Auto-select best frames for analysis

Threshold calibration + confidence intervals

PDF report export

👤 Author

Ali Haidar
BS Computer Science (2020–2024) | AI/ML Projects Portfolio

GitHub: (add your profile/repo link here)
LinkedIn: (optional)

⭐ Support

If you find this useful, please ⭐ the repository.
