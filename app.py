import streamlit as st
import tempfile
import cv2
import pandas as pd
from PIL import Image
import numpy as np
import os
os.environ["INSIGHTFACE_HOME"] = "./.insightface"

from utils_video import (
    extract_frames,
    blur_score,
    noise_score,
    forgery_risk_from_signals,
    detect_faces,
    deepfake_likeliness_from_faces
)

from utils_identity import load_det_rec_models, get_face_embedding
from utils_deepfake_model import load_detector, predict_faces


# =======================
# Streamlit Config
# =======================
st.set_page_config(page_title="AegisAI — Multimedia Integrity Analyzer", layout="wide")
st.title("🛡️ AegisAI — Multimedia Integrity Analyzer (Video + Photo)")
st.info(
    "✅ APP VERSION: FINAL-REF(PHOTO+VIDEO)+SUS(PHOTO/VIDEO/OR-BOTH)+OUTLIERS+TRUECOS+EVIDENCE+MULTI-PERSON+ROI-FALLBACK+SMARTVERDICT-FIX+EVIDENCE-TUNED+PHOTO-EMB-FALLBACK+HF-SAFELOAD"
)


# =======================
# Cached Loaders
# =======================
@st.cache_resource(show_spinner="Loading face recognition models...")
def get_models_cached():
    # prefer_gpu=True will auto-use GPU if available
    return load_det_rec_models(prefer_gpu=True)


det_model, rec_model = get_models_cached()


@st.cache_resource
def get_detector_safe():
    """
    SAFE detector loader:
    - prevents app crash if HF model missing image processor config
    - returns (detector_or_None, error_string_or_None)
    """
    try:
        det = load_detector()
        return det, None
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"


# =======================
# Helpers
# =======================
def _resize_if_big(bgr, max_side=1600):
    h, w = bgr.shape[:2]
    if max(h, w) <= max_side:
        return bgr
    scale = max_side / max(h, w)
    return cv2.resize(bgr, (int(w * scale), int(h * scale)))


def _enhance_for_detection(bgr):
    try:
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l2 = clahe.apply(l)
        lab2 = cv2.merge([l2, a, b])
        return cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)
    except Exception:
        return bgr


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a).reshape(-1).astype(np.float32)
    b = np.asarray(b).reshape(-1).astype(np.float32)
    a = a / (np.linalg.norm(a) + 1e-9)
    b = b / (np.linalg.norm(b) + 1e-9)
    return float(np.dot(a, b))


def safe_len(x) -> int:
    if x is None:
        return 0
    try:
        return int(len(x))
    except Exception:
        return 0


def normalize_face_crop_to_224(rgb):
    if rgb is None or getattr(rgb, "size", 0) == 0:
        return None
    try:
        return cv2.resize(rgb, (224, 224), interpolation=cv2.INTER_AREA)
    except Exception:
        return None


def compute_ref_embedding(ref_rgbs, det_model, rec_model, show_debug=False):
    def center_crop(rgb, crop_ratio=0.75):
        h, w = rgb.shape[:2]
        nh, nw = int(h * crop_ratio), int(w * crop_ratio)
        y1 = (h - nh) // 2
        x1 = (w - nw) // 2
        return rgb[y1:y1 + nh, x1:x1 + nw]

    success_embs = []
    used = 0
    failed = 0

    for idx, rgb in enumerate(ref_rgbs, start=1):
        variants = [
            ("orig", rgb),
            ("rot90", np.rot90(rgb, 1)),
            ("rot180", np.rot90(rgb, 2)),
            ("rot270", np.rot90(rgb, 3)),
            ("crop85", center_crop(rgb, 0.85)),
            ("crop70", center_crop(rgb, 0.70)),
        ]

        got_one = False
        last_err = None

        for _, v in variants:
            bgr = cv2.cvtColor(v, cv2.COLOR_RGB2BGR)
            bgr = _resize_if_big(bgr, 1600)
            bgr = _enhance_for_detection(bgr)
            try:
                emb = get_face_embedding(bgr, det_model, rec_model)
                emb = np.asarray(emb).reshape(-1).astype(np.float32)
                emb = emb / (np.linalg.norm(emb) + 1e-9)
                success_embs.append(emb)
                used += 1
                got_one = True
                break
            except Exception as e:
                last_err = e

        if not got_one:
            failed += 1
            if show_debug:
                st.warning(f"Reference item {idx} skipped. Last error: {type(last_err).__name__}: {last_err}")

    if len(success_embs) == 0:
        return None, used, failed

    emb = np.mean(np.stack(success_embs), axis=0)
    emb = emb / (np.linalg.norm(emb) + 1e-9)
    return emb.reshape(-1), used, failed


def enroll_from_reference_videos(ref_videos, det_model, rec_model, frame_every=5, max_frames_per_video=20):
    if not ref_videos:
        return None, 0, 0, 0

    all_ref_frames = []
    total_frames_seen = 0
    failed_videos = 0

    for v in ref_videos:
        try:
            vb = v.getvalue()
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            vpath = tmp.name
            tmp.write(vb)
            tmp.close()

            cap = cv2.VideoCapture(vpath)
            count = 0
            frames = []
            while cap.isOpened() and len(frames) < max_frames_per_video:
                ok, frame_bgr = cap.read()
                if not ok:
                    break
                if count % frame_every == 0:
                    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                    frames.append(frame_rgb)
                count += 1
            cap.release()

            total_frames_seen += len(frames)
            all_ref_frames.extend(frames)
        except Exception:
            failed_videos += 1

    if not all_ref_frames:
        return None, 0, failed_videos + 1, total_frames_seen

    emb, used, failed = compute_ref_embedding(all_ref_frames, det_model, rec_model, show_debug=False)
    return emb, used, failed + failed_videos, total_frames_seen


def filter_reference_outliers(emb_list, thr=0.35):
    n = len(emb_list)
    if n <= 2:
        return emb_list, [], None

    idxs = [i for i, _ in emb_list]
    embs = [e for _, e in emb_list]

    sims = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for j in range(n):
            sims[i, j] = cosine_sim(embs[i], embs[j])

    sim_df = pd.DataFrame(
        sims,
        index=[f"ref_{i}" for i in idxs],
        columns=[f"ref_{i}" for i in idxs]
    )

    kept = []
    dropped = []
    for i in range(n):
        others = [sims[i, j] for j in range(n) if j != i]
        med = float(np.median(others)) if others else 1.0
        if med >= thr:
            kept.append(emb_list[i])
        else:
            dropped.append((emb_list[i][0], emb_list[i][1], med))

    return kept, dropped, sim_df


def simple_cluster_embeddings(embs, thr=0.42):
    clusters = []
    for i, e in enumerate(embs):
        placed = False
        for c in clusters:
            centroid = np.mean([embs[j] for j in c], axis=0)
            centroid = centroid / (np.linalg.norm(centroid) + 1e-9)
            if cosine_sim(e, centroid) >= thr:
                c.append(i)
                placed = True
                break
        if not placed:
            clusters.append([i])
    return clusters


def pick_cluster(clusters, embs, ref_emb=None):
    if not clusters:
        return None
    if ref_emb is None:
        sizes = [len(c) for c in clusters]
        return int(np.argmax(sizes))

    best_ci, best_score = None, -999
    for ci, c in enumerate(clusters):
        centroid = np.mean([embs[j] for j in c], axis=0)
        centroid = centroid / (np.linalg.norm(centroid) + 1e-9)
        s = cosine_sim(centroid, ref_emb)
        if s > best_score:
            best_score, best_ci = s, ci
    return best_ci


def similarity_stats_from_embs(embs, ref_emb):
    sims = [cosine_sim(e, ref_emb) for e in embs if e is not None]
    if not sims:
        return 0.0, 0.0, 0, []
    return float(np.mean(sims)), float(np.max(sims)), int(len(sims)), sims


def build_crops_and_rois_from_image(rgb, pad=0.45):
    boxes = detect_faces(rgb)
    if boxes is None or len(boxes) == 0:
        return [], []

    h, w = rgb.shape[:2]
    crops224 = []
    rois = []

    for (x, y, bw, bh) in boxes:
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(w, x + bw), min(h, y + bh)

        crop = rgb[y1:y2, x1:x2].copy()
        crop224 = normalize_face_crop_to_224(crop)
        if crop224 is None:
            continue

        px, py = int(bw * pad), int(bh * pad)
        rx1, ry1 = max(0, x - px), max(0, y - py)
        rx2, ry2 = min(w, x + bw + px), min(h, y + bh + py)
        roi = rgb[ry1:ry2, rx1:rx2].copy()

        crops224.append(crop224)
        rois.append(roi)

    return crops224, rois


def embed_rois(rois_rgb, det_model, rec_model, fallback_frames_rgb=None):
    embs = []
    failed = 0

    for i, roi in enumerate(rois_rgb):
        e = None

        try:
            bgr = cv2.cvtColor(roi, cv2.COLOR_RGB2BGR)
            bgr = _resize_if_big(bgr, 1600)
            bgr = _enhance_for_detection(bgr)
            e = get_face_embedding(bgr, det_model, rec_model)
        except Exception:
            e = None

        if (e is None) and (fallback_frames_rgb is not None) and (i < len(fallback_frames_rgb)):
            try:
                fr = fallback_frames_rgb[i]
                bgr2 = cv2.cvtColor(fr, cv2.COLOR_RGB2BGR)
                bgr2 = _resize_if_big(bgr2, 1600)
                bgr2 = _enhance_for_detection(bgr2)
                e = get_face_embedding(bgr2, det_model, rec_model)
            except Exception:
                e = None

        if e is not None:
            e = np.asarray(e).reshape(-1).astype(np.float32)
            e = e / (np.linalg.norm(e) + 1e-9)
            embs.append(e)
        else:
            failed += 1

    return embs, failed


def embed_full_images(rgb_list, det_model, rec_model):
    out = []
    for rgb in rgb_list:
        try:
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            bgr = _resize_if_big(bgr, 1600)
            bgr = _enhance_for_detection(bgr)
            e = get_face_embedding(bgr, det_model, rec_model)
            e = np.asarray(e).reshape(-1).astype(np.float32)
            e = e / (np.linalg.norm(e) + 1e-9)
            out.append(e)
        except Exception:
            pass
    return out


# =======================
# UI Inputs
# =======================
st.subheader("🪪 Reference Identity (Optional)")
ref_files = st.file_uploader(
    "Upload reference PHOTOS (optional) — 1–5 clear selfies (jpg/png).",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True
)
ref_videos = st.file_uploader(
    "Upload reference VIDEOS (optional) — 1–3 short real videos (mp4/mov/avi).",
    type=["mp4", "mov", "avi"],
    accept_multiple_files=True
)

st.subheader("🎯 Suspect Evidence (Upload Video OR Photos OR Both)")
uploaded_video = st.file_uploader("Upload SUSPECT video (optional)", type=["mp4", "mov", "avi"])
suspect_photos = st.file_uploader(
    "Upload SUSPECT photo(s) (optional)",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True
)

if (uploaded_video is None) and (not suspect_photos):
    st.info("Please upload at least **one** suspect evidence: **video OR photo(s)**.")
    st.stop()


# =======================
# Reference Enrollment
# =======================
st.subheader("🧾 Reference Enrollment (Optional)")
ref_emb = None
used_refs = 0
photo_emb = None
video_emb = None

if ref_files:
    per_photo_embs = []
    max_photos = min(5, len(ref_files))

    for idx, f in enumerate(ref_files[:max_photos], start=1):
        rgb = np.array(Image.open(f).convert("RGB"))
        emb, _, _ = compute_ref_embedding([rgb], det_model, rec_model, show_debug=False)
        if emb is not None:
            per_photo_embs.append((idx, emb))
        else:
            st.warning(f"Reference photo #{idx}: no face detected, skipped.")

    if len(per_photo_embs) >= 2:
        kept, dropped, sim_df = filter_reference_outliers(per_photo_embs, thr=0.35)
        st.write("Reference consistency matrix:")
        if sim_df is not None:
            st.dataframe(sim_df, use_container_width=True)

        if dropped:
            st.warning("Excluded reference outliers:")
            for (bad_idx, _, med) in dropped:
                st.write(f"❌ Photo #{bad_idx} excluded (median similarity={med:.3f})")

        if len(kept) > 0:
            used_refs += len(kept)
            photo_emb = np.mean(np.stack([e for _, e in kept]), axis=0)
            photo_emb = photo_emb / (np.linalg.norm(photo_emb) + 1e-9)
            photo_emb = photo_emb.reshape(-1)
            st.success(f"✅ Using {len(kept)}/{len(per_photo_embs)} reference photo(s).")
        else:
            st.error("All reference photos are inconsistent. Upload correct selfies only.")
    elif len(per_photo_embs) == 1:
        used_refs += 1
        photo_emb = per_photo_embs[0][1]
        st.info("Only 1 valid reference photo. Outlier check skipped.")

if ref_videos:
    video_emb, used_v, failed_v, total_frames_seen = enroll_from_reference_videos(
        ref_videos, det_model, rec_model, frame_every=5, max_frames_per_video=20
    )
    if video_emb is not None:
        used_refs += 1
    st.info(f"Reference video enrollment: extracted {total_frames_seen} frame(s). Used {used_v}, skipped {failed_v}.")

emb_list = []
if photo_emb is not None:
    emb_list.append(photo_emb)
if video_emb is not None:
    emb_list.append(video_emb)

if emb_list:
    ref_emb = np.mean(np.stack(emb_list), axis=0)
    ref_emb = ref_emb / (np.linalg.norm(ref_emb) + 1e-9)
    ref_emb = ref_emb.reshape(-1)
    st.success("✅ Reference embedding created.")
else:
    st.info("No reference provided → identity matching will be skipped.")


# =======================
# Suspect VIDEO pipeline (if provided)
# =======================
frames = []
avg_risk = 0.0
avg_blur = 0.0
filtered_faces_video = []
video_roi_embs = []
video_clusters = None

if uploaded_video is not None:
    st.subheader("🎞️ Suspect VIDEO Analysis")

    suspect_bytes = uploaded_video.getvalue()
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    suspect_path = tfile.name
    tfile.write(suspect_bytes)
    tfile.close()

    st.success("Suspect video uploaded ✅")
    st.video(suspect_bytes)

    cap = cv2.VideoCapture(suspect_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    duration = total / fps if fps else 0
    st.write(f"**FPS:** {fps:.2f} | **Frames:** {total} | **Duration:** {duration:.1f}s")

    frame_every = st.slider("Extract 1 frame every N frames (video)", 1, 60, 10)
    frames = extract_frames(suspect_path, frame_every, max_frames=12)

    if not frames:
        st.warning("No frames extracted from video. Try smaller N.")
    else:
        rows = []
        for i, fr in enumerate(frames, start=1):
            b = blur_score(fr)
            n = noise_score(fr)
            r = forgery_risk_from_signals(b, n)
            rows.append({"frame": i, "blur_score": round(b, 2), "noise_score": round(n, 3), "risk": round(r, 1)})

        df = pd.DataFrame(rows)
        avg_blur = float(df["blur_score"].mean())
        avg_risk = float(df["risk"].mean())

        c1, c2, c3 = st.columns(3)
        c1.metric("Avg Blur Score (higher=sharper)", f"{avg_blur:.1f}")
        c2.metric("Baseline Risk", f"{avg_risk:.0f}/100")
        c3.metric("Frames used", f"{len(frames)}")
        st.progress(int(max(0, min(100, avg_risk))))
        st.dataframe(df, use_container_width=True)

        all_faces_video = []
        all_rois_video = []
        roi_to_frame = []
        for fr in frames:
            crops224, rois = build_crops_and_rois_from_image(fr, pad=0.45)
            all_faces_video.extend(crops224)
            all_rois_video.extend(rois)
            roi_to_frame.extend([fr] * len(rois))

        st.subheader("🧩 Face Crops (224×224) — Suspect VIDEO")
        if all_faces_video:
            st.write(f"Total face crops: **{len(all_faces_video)}**")
            cols = st.columns(6)
            for i, face_img in enumerate(all_faces_video[:18]):
                cols[i % 6].image(face_img, caption=f"VFace {i+1}", use_container_width=True)
        else:
            st.info("No faces found in suspect video.")

        video_roi_embs, _ = embed_rois(all_rois_video, det_model, rec_model, fallback_frames_rgb=roi_to_frame)

        st.subheader("🧷 Suspect VIDEO Multi-Person Check")
        cluster_thr = st.slider("Clustering threshold (video)", 0.30, 0.60, 0.42, 0.01)
        if len(video_roi_embs) >= 3:
            video_clusters = simple_cluster_embeddings(video_roi_embs, thr=cluster_thr)
            if len(video_clusters) >= 2:
                st.error("🚨 TICKET: Suspect VIDEO contains MULTIPLE IDENTITIES.")
                st.write("Cluster sizes:", [len(c) for c in video_clusters])
            else:
                st.success("✅ Suspect VIDEO looks like a single identity.")
        else:
            st.info("Not enough face embeddings for multi-person check (need ~3+).")

        filtered_faces_video = all_faces_video
        if all_faces_video and video_clusters and len(video_clusters) >= 2:
            sel = pick_cluster(video_clusters, video_roi_embs, ref_emb=ref_emb)
            keep = set(video_clusters[sel])
            filtered_faces_video = [all_faces_video[i] for i in range(len(all_faces_video)) if i in keep]
            st.warning(f"✅ Using ONLY identity cluster #{sel+1} (others skipped). Kept {len(filtered_faces_video)} faces.")
        elif all_faces_video:
            filtered_faces_video = all_faces_video


# =======================
# Suspect PHOTOS pipeline (if provided)
# =======================
filtered_faces_photo = []
photo_roi_embs = []
photo_clusters = None
suspect_rgb_list = []

if suspect_photos:
    st.subheader("🖼️ Suspect PHOTO Analysis")

    cols = st.columns(4)
    for i, f in enumerate(suspect_photos, start=1):
        rgb = np.array(Image.open(f).convert("RGB"))
        suspect_rgb_list.append(rgb)
        cols[(i-1) % 4].image(rgb, caption=f"Suspect Photo {i}", use_container_width=True)

    all_faces_photo = []
    all_rois_photo = []
    roi_to_photo = []
    for rgb in suspect_rgb_list:
        crops224, rois = build_crops_and_rois_from_image(rgb, pad=0.45)
        all_faces_photo.extend(crops224)
        all_rois_photo.extend(rois)
        roi_to_photo.extend([rgb] * len(rois))

    st.subheader("🧩 Face Crops (224×224) — Suspect PHOTOS")
    if all_faces_photo:
        st.write(f"Total face crops: **{len(all_faces_photo)}**")
        cols = st.columns(6)
        for i, face_img in enumerate(all_faces_photo[:18]):
            cols[i % 6].image(face_img, caption=f"PFace {i+1}", use_container_width=True)
    else:
        st.info("No faces found in suspect photos.")

    photo_roi_embs, _ = embed_rois(all_rois_photo, det_model, rec_model, fallback_frames_rgb=roi_to_photo)

    if len(photo_roi_embs) < 2 and len(suspect_rgb_list) >= 2:
        extra = embed_full_images(suspect_rgb_list, det_model, rec_model)
        photo_roi_embs = photo_roi_embs + extra

    st.subheader("🧷 Suspect PHOTOS Multi-Person Check")
    cluster_thr_p = st.slider("Clustering threshold (photos)", 0.30, 0.60, 0.42, 0.01)
    if len(photo_roi_embs) >= 2:
        photo_clusters = simple_cluster_embeddings(photo_roi_embs, thr=cluster_thr_p)
        if len(photo_clusters) >= 2:
            st.error("🚨 TICKET: Suspect PHOTOS contain MULTIPLE IDENTITIES.")
            st.write("Cluster sizes:", [len(c) for c in photo_clusters])
        else:
            st.success("✅ Suspect PHOTOS look like a single identity.")
    else:
        st.info("Not enough face embeddings for multi-person check (need ~2+).")

    filtered_faces_photo = all_faces_photo


# =======================
# Identity Result
# =======================
st.subheader("🪪 Identity Result (Reference vs Suspect)")

sim_use = 0.0
avg_sim_video, avg_sim_photo = 0.0, 0.0

if ref_emb is None:
    st.info("No reference uploaded → identity match is skipped.")
else:
    if video_roi_embs:
        avg_sim_video, _, n_video, _ = similarity_stats_from_embs(video_roi_embs, ref_emb)
    else:
        n_video = 0

    if photo_roi_embs:
        avg_sim_photo, _, n_photo, _ = similarity_stats_from_embs(photo_roi_embs, ref_emb)
    else:
        n_photo = 0

    candidates = []
    if n_video > 0:
        candidates.append(avg_sim_video)
    if n_photo > 0:
        candidates.append(avg_sim_photo)
    sim_use = float(max(candidates)) if candidates else 0.0

    if sim_use == 0.0:
        st.warning("Could not compute identity similarity (no usable face embeddings).")
    else:
        if sim_use >= 0.45:
            id_verdict = "MATCH (same person)"
        elif sim_use >= 0.35:
            id_verdict = "UNCERTAIN"
        else:
            id_verdict = "NOT MATCH"

        c1, c2, c3 = st.columns(3)
        c1.metric("Avg similarity (best source)", f"{sim_use:.3f}")
        c2.metric("Avg similarity (video)", f"{avg_sim_video:.3f}")
        c3.metric("Avg similarity (photos)", f"{avg_sim_photo:.3f}")
        st.success(f"Identity Result: **{id_verdict}**")


# =======================
# Deepfake Detection (SAFE)
# =======================
st.subheader("🧠 Model-based Deepfake Detection")

detector, det_err = get_detector_safe()

if detector is None:
    st.error("Deepfake model could not be loaded, so deepfake scoring is DISABLED.")
    st.caption("Fix in your environment (recommended):")
    st.code(
        'pip install -U "transformers==4.41.2" "huggingface_hub==0.23.4" timm pillow\n'
        "Then delete HF cache folder for this model and rerun.",
        language="bash"
    )
    st.caption(f"Loader error: {det_err}")
    calibrated_fake_video = 0.0
    calibrated_fake_photo = 0.0
else:
    calibrated_fake_video = 0.0
    calibrated_fake_photo = 0.0

    if uploaded_video is not None and filtered_faces_video:
        per_face_v, avg_fake_v, verdict_v = predict_faces(detector, filtered_faces_video)
        avg_fake_v = float(avg_fake_v)
        calibrated_fake_video = 0.6 * avg_fake_v + 0.4 * float(avg_risk)
        calibrated_fake_video = max(0.0, min(100.0, calibrated_fake_video))

        st.subheader("🎞️ Deepfake Result — Suspect VIDEO")
        st.metric("Avg FAKE probability (raw)", f"{avg_fake_v:.0f}%")
        st.metric("Calibrated FAKE probability", f"{calibrated_fake_video:.0f}%")
        st.success(f"Verdict: **{verdict_v}**")
        st.progress(int(calibrated_fake_video))

    if suspect_photos and filtered_faces_photo:
        per_face_p, avg_fake_p, verdict_p = predict_faces(detector, filtered_faces_photo)
        avg_fake_p = float(avg_fake_p)
        calibrated_fake_photo = max(0.0, min(100.0, 0.85 * avg_fake_p))

        st.subheader("🖼️ Deepfake Result — Suspect PHOTOS")
        st.metric("Avg FAKE probability (raw)", f"{avg_fake_p:.0f}%")
        st.metric("Calibrated FAKE probability", f"{calibrated_fake_photo:.0f}%")
        st.success(f"Verdict: **{verdict_p}**")
        st.progress(int(calibrated_fake_photo))

    if (uploaded_video is None or not filtered_faces_video) and (not suspect_photos or not filtered_faces_photo):
        st.info("No usable faces found → deepfake model cannot run.")


# =======================
# Final Risk
# =======================
st.subheader("✅ Final Forensic Risk")

final_risk = 0.0
deepfake_score = 0.0

if uploaded_video is not None and filtered_faces_video and detector is not None:
    deepfake_score, _ = deepfake_likeliness_from_faces(filtered_faces_video)
    final_risk = 0.20 * float(avg_risk) + 0.20 * float(deepfake_score) + 0.60 * float(calibrated_fake_video)
    final_risk = max(0.0, min(100.0, final_risk))
    st.metric("Final Risk (VIDEO primary)", f"{final_risk:.0f}/100")
    st.progress(int(final_risk))
elif suspect_photos and filtered_faces_photo and detector is not None:
    deepfake_score, _ = deepfake_likeliness_from_faces(filtered_faces_photo)
    final_risk = 0.30 * float(deepfake_score) + 0.70 * float(calibrated_fake_photo)
    final_risk = max(0.0, min(100.0, final_risk))
    st.metric("Final Risk (PHOTO only)", f"{final_risk:.0f}/100")
    st.progress(int(final_risk))
else:
    st.info("Not enough model data to compute final risk (deepfake model disabled or no usable faces).")


# =======================
# Evidence Strength
# =======================
st.subheader("📊 Evidence Strength")

evidence = 0

ref_sources = 0
if ref_files and len(ref_files) > 0:
    ref_sources += 1
if ref_videos and len(ref_videos) > 0:
    ref_sources += 1

if ref_sources >= 2:
    evidence += 35
elif ref_sources == 1:
    evidence += 25

if ref_emb is not None:
    if sim_use >= 0.70:
        evidence += 25
    elif sim_use >= 0.60:
        evidence += 20
    elif sim_use >= 0.45:
        evidence += 12

if uploaded_video is not None and safe_len(filtered_faces_video) >= 8:
    evidence += 30
elif uploaded_video is not None and safe_len(filtered_faces_video) >= 4:
    evidence += 20

if suspect_photos and safe_len(filtered_faces_photo) >= 4:
    evidence += 20
elif suspect_photos and safe_len(filtered_faces_photo) >= 2:
    evidence += 15
elif suspect_photos and safe_len(filtered_faces_photo) >= 1:
    evidence += 8

if uploaded_video is not None:
    if avg_blur > 40:
        evidence += 15
    elif avg_blur > 30:
        evidence += 8

evidence = min(100, evidence)

if evidence >= 80:
    st.success(f"Evidence Strength: HIGH ({evidence}/100)")
elif evidence >= 50:
    st.warning(f"Evidence Strength: MODERATE ({evidence}/100)")
else:
    st.error(f"Evidence Strength: LOW ({evidence}/100)")


# =======================
# Smart Verdict (Summary)
# =======================
st.subheader("🧠 Smart Verdict (Summary)")

best_id_candidates = []
if ref_emb is not None:
    if video_roi_embs:
        sv = [cosine_sim(e, ref_emb) for e in video_roi_embs if e is not None]
        if sv:
            best_id_candidates.append(float(np.mean(sv)))
    if photo_roi_embs:
        sp = [cosine_sim(e, ref_emb) for e in photo_roi_embs if e is not None]
        if sp:
            best_id_candidates.append(float(np.mean(sp)))

sim_best = max(best_id_candidates) if best_id_candidates else 0.0

fake_candidates = []
try:
    if 'calibrated_fake_video' in locals() and calibrated_fake_video > 0:
        fake_candidates.append(float(calibrated_fake_video))
    if 'calibrated_fake_photo' in locals() and calibrated_fake_photo > 0:
        fake_candidates.append(float(calibrated_fake_photo))
except Exception:
    pass

fake_score = max(fake_candidates) if fake_candidates else 0.0

if detector is None:
    st.info("Result: **Deepfake model disabled** — only identity + quality signals available.")
else:
    if ref_emb is None:
        if fake_score >= 80:
            st.error("Result: **High suspicion of deepfake/manipulation** (no reference identity used).")
        elif fake_score >= 60:
            st.warning("Result: **Uncertain** — suspicious artifacts detected. Provide clearer evidence.")
        else:
            st.success("Result: **Low suspicion** (likely real).")
    else:
        if sim_best >= 0.45 and fake_score >= 80:
            st.warning("Result: **Same person BUT likely manipulated** (possible deepfake / face-swap / heavy editing).")
        elif sim_best >= 0.45 and 60 <= fake_score < 80:
            st.info("Result: **Same person but UNCERTAIN** — suspicious artifacts; needs more evidence.")
        elif sim_best >= 0.45 and fake_score < 60:
            st.success("Result: **Same person and likely real**.")
        elif sim_best < 0.35 and fake_score >= 80:
            st.error("Result: **Different person AND likely impersonation/deepfake**.")
        else:
            st.info("Result: **Uncertain** — improve reference quality and upload clearer suspect media.")

st.caption("ℹ️ Temp media is stored in system temp. (We don't delete it automatically on Windows.)")
