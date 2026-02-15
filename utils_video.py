import cv2
import numpy as np

# Load face detector ONCE (important for performance)
FACE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


# ---------------------------------------------------
# FRAME EXTRACTION
# ---------------------------------------------------
def extract_frames(video_path, frame_every=15, max_frames=12):
    cap = cv2.VideoCapture(video_path)

    frames = []
    idx = 0
    ok, frame = cap.read()

    while ok and len(frames) < max_frames:
        if idx % frame_every == 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)

        idx += 1
        ok, frame = cap.read()

    cap.release()
    return frames


# ---------------------------------------------------
# IMAGE QUALITY SIGNALS
# ---------------------------------------------------
def blur_score(frame_rgb):
    """
    Blur detection using Laplacian variance.
    Lower value = blurrier image.
    """
    gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def noise_score(frame_rgb):
    """
    Estimate compression / noise artifacts.
    """
    gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    high_freq = cv2.absdiff(gray, blur)
    return float(np.mean(high_freq))


def forgery_risk_from_signals(blur, noise):
    """
    Heuristic forgery risk (0-100).
    This is NOT deepfake detection yet —
    just manipulation clues.
    """

    blur_norm = min(max((200.0 - blur) / 200.0, 0.0), 1.0)
    noise_norm = min(max((noise - 2.0) / 8.0, 0.0), 1.0)

    risk = 100 * (0.65 * blur_norm + 0.35 * noise_norm)

    return float(min(max(risk, 0.0), 100.0))


# ---------------------------------------------------
# FACE DETECTION (VERY IMPORTANT FOR DEEPFAKE)
# ---------------------------------------------------
def detect_faces(frame_rgb):
    """
    Detect faces using Haar Cascade.
    Returns list of (x, y, w, h)
    """

    gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)

    faces = FACE_CASCADE.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(60, 60)
    )

    return faces


# ---------------------------------------------------
# FACE CROPPING (FOR MODEL INPUT)
# ---------------------------------------------------
def crop_faces(frame_rgb):
    """
    Returns cropped face images.
    This is what deepfake models use.
    """

    faces = detect_faces(frame_rgb)

    crops = []

    for (x, y, w, h) in faces:
        face = frame_rgb[y:y+h, x:x+w]
        face = cv2.resize(face, (224, 224))  # standard model size
        crops.append(face)

    return crops
def _center_crop(img, size=112):
    h, w = img.shape[:2]
    y1 = max((h - size) // 2, 0)
    x1 = max((w - size) // 2, 0)
    return img[y1:y1+size, x1:x1+size]

def deepfake_likeliness_from_faces(face_crops):
    """
    CPU-only prototype score (0-100) based on:
    - texture inconsistency (high-frequency energy)
    - edge/aliasing artifacts
    Not a true deepfake classifier, but a useful baseline signal.
    """
    if not face_crops:
        return 0.0, {}

    tex_scores = []
    edge_scores = []

    for face in face_crops:
        # face is RGB 224x224
        gray = cv2.cvtColor(face, cv2.COLOR_RGB2GRAY)

        # focus on central area (avoid background)
        c = _center_crop(gray, 112)

        # texture (high-frequency)
        blur = cv2.GaussianBlur(c, (3, 3), 0)
        hf = cv2.absdiff(c, blur)
        tex = float(np.mean(hf))  # higher => more artifacts
        tex_scores.append(tex)

        # edges / aliasing
        edges = cv2.Canny(c, 60, 120)
        edge = float(np.mean(edges)) / 255.0  # 0-1
        edge_scores.append(edge)

    avg_tex = float(np.mean(tex_scores))
    avg_edge = float(np.mean(edge_scores))

    # Heuristic normalization (tuned for typical webcam/phone faces)
    # Calibrated for typical phone videos (your values ~0.5–2.0)
    tex_norm = min(max((avg_tex - 0.30) / 1.70, 0.0), 1.0)
    
    # Edge density often low; treat >0.01 as starting risk

    edge_norm = min(max((avg_edge - 0.01) / 0.10, 0.0), 1.0)

    score = 100.0 * (0.65 * tex_norm + 0.35 * edge_norm)
    score = float(min(max(score, 0.0), 100.0))

    details = {
        "avg_texture": round(avg_tex, 3),
        "avg_edge_density": round(avg_edge, 3),
    }
    return score, details
