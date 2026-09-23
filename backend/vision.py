"""
Core vision/OCR/liveness/matching logic for the demo.

IMPORTANT — read this before treating any number here as gospel:
This is a PROTOTYPE built to demonstrate the pipeline end-to-end, not a
certified identity-verification product. Specifically:

- Document "authenticity" is limited to OCR + a few structural heuristics.
  There is no connection to any Bhutanese government registry, so this can
  never actually confirm a CID/passport/licence number is real or unrevoked.
- Liveness uses the 5-point landmarks that come back from OpenCV's YuNet
  face detector (eyes, nose tip, mouth corners). That's enough to detect a
  deliberate head turn or a smile, but NOT enough to do proper blink/EAR
  detection or robust presentation-attack detection (no certified PAD model
  is used here). The active challenge-response design (server picks a random
  order) is the main real defense; the "spoof_flag" heuristic is a soft,
  best-effort signal on top of it, not a certified liveness check.
- Face-match thresholds below are placeholders calibrated only against the
  publicly documented SFace reference point, not a validated dataset. Tune
  them against real data (ideally including aged photos) before trusting them.
"""
import base64
import random
import re
import sys
import tempfile
import urllib.request
from pathlib import Path

import cv2
import numpy as np
import pytesseract

MODELS_DIR = Path(__file__).parent / "models"

# The face-recognition weights (~37MB) are fetched on first run instead of
# being bundled in the repo/zip. Source: OpenCV Zoo, Apache-2.0 license
# (https://github.com/opencv/opencv_zoo/tree/main/models/face_recognition_sface).
SFACE_URL = (
    "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/"
    "models/face_recognition_sface/face_recognition_sface_2021dec.onnx"
)
SFACE_PATH = MODELS_DIR / "face_recognition_sface.onnx"


def _ensure_sface_downloaded():
    if SFACE_PATH.exists() and SFACE_PATH.stat().st_size > 1_000_000:
        return
    print(f"[vision] Downloading face recognition model (~37MB) to {SFACE_PATH} ...", file=sys.stderr)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(SFACE_URL, SFACE_PATH)
    print("[vision] Download complete.", file=sys.stderr)


# ---- Model loading (once, at import time) -----------------------------------

_detector = None
_recognizer = None


def get_models():
    global _detector, _recognizer
    if _detector is None:
        _detector = cv2.FaceDetectorYN_create(
            str(MODELS_DIR / "face_detection_yunet.onnx"), "", (320, 320),
            score_threshold=0.7,
        )
    if _recognizer is None:
        _ensure_sface_downloaded()
        _recognizer = cv2.FaceRecognizerSF_create(str(SFACE_PATH), "")
    return _detector, _recognizer


def _detect_best_face(bgr_image):
    """Returns the highest-confidence face row from YuNet, or None."""
    detector, _ = get_models()
    h, w = bgr_image.shape[:2]
    detector.setInputSize((w, h))
    _, faces = detector.detect(bgr_image)
    if faces is None or len(faces) == 0:
        return None
    # faces: Nx15 [x,y,w,h, 5x(lx,ly), score]
    best = max(faces, key=lambda f: f[-1])
    return best


def _face_metrics(face_row):
    """Derive yaw-proxy and smile-proxy signals from YuNet's 5-pt landmarks."""
    x, y, w, h = face_row[0:4]
    r_eye = face_row[4:6]
    l_eye = face_row[6:8]
    nose = face_row[8:10]
    r_mouth = face_row[10:12]
    l_mouth = face_row[12:14]
    score = face_row[14]

    eye_dist = float(np.linalg.norm(r_eye - l_eye)) or 1.0
    mouth_width = float(np.linalg.norm(r_mouth - l_mouth))
    bbox_center_x = x + w / 2.0
    # negative -> nose sits left of bbox center (subject's face turned so their
    # right side is more visible), positive -> turned the other way. This is a
    # coarse proxy, not a calibrated pose-estimation result.
    nose_offset = float((nose[0] - bbox_center_x) / (w / 2.0 or 1.0))
    smile_ratio = mouth_width / eye_dist

    return {
        "score": float(score),
        "nose_offset": nose_offset,
        "smile_ratio": smile_ratio,
        "bbox": [float(x), float(y), float(w), float(h)],
    }


def _blur_score(gray):
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _encode_jpg_b64(bgr_image):
    ok, buf = cv2.imencode(".jpg", bgr_image, [cv2.IMWRITE_JPEG_QUALITY, 88])
    if not ok:
        return None
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()


# ---- Document OCR + field-guessing heuristics -------------------------------

CID_RE = re.compile(r"\b\d{11}\b")
PASSPORT_RE = re.compile(r"\b[A-Z]{1,2}\d{6,7}\b")
DATE_RE = re.compile(r"\b(\d{1,2}[\/\-.]\d{1,2}[\/\-.]\d{2,4}|\d{4}[\/\-.]\d{1,2}[\/\-.]\d{1,2})\b")
MRZ_LINE_RE = re.compile(r"[A-Z0-9<]{20,}")

_MRZ_WEIGHTS = [7, 3, 1]
_MRZ_VALUES = {c: i for i, c in enumerate("0123456789")}
_MRZ_VALUES.update({c: i + 10 for i, c in enumerate("ABCDEFGHIJKLMNOPQRSTUVWXYZ")})
_MRZ_VALUES["<"] = 0


def _mrz_check_digit(data: str) -> int:
    total = 0
    for i, ch in enumerate(data):
        total += _MRZ_VALUES.get(ch, 0) * _MRZ_WEIGHTS[i % 3]
    return total % 10


def guess_document_type(raw_text: str) -> str:
    t = raw_text.upper()
    if "PASSPORT" in t or MRZ_LINE_RE.search(t):
        return "passport"
    if "DRIVING" in t or "LICENCE" in t or "LICENSE" in t:
        return "driving_license"
    if "CITIZENSHIP" in t or "KINGDOM OF BHUTAN" in t or CID_RE.search(t):
        return "cid"
    return "unknown"


def guess_name(raw_text: str):
    banned = ("KINGDOM", "BHUTAN", "DEPARTMENT", "CARD", "LICENCE", "LICENSE",
              "PASSPORT", "CIVIL", "REGISTRATION", "CENSUS", "IDENTITY",
              "GOVERNMENT", "MINISTRY")
    best = None
    for line in raw_text.splitlines():
        clean = line.strip()
        letters = re.sub(r"[^A-Za-z ]", "", clean)
        if len(letters) < 4:
            continue
        if any(b in clean.upper() for b in banned):
            continue
        upper_ratio = sum(1 for c in letters if c.isupper()) / max(len(letters.replace(" ", "")), 1)
        if upper_ratio > 0.8 and (best is None or len(clean) > len(best)):
            best = clean
    return best


def try_validate_mrz(raw_text: str):
    """Very small subset of ICAO 9303 TD3 check-digit validation, best-effort."""
    candidates = [l.strip().replace(" ", "") for l in raw_text.splitlines()]
    candidates = [l for l in candidates if MRZ_LINE_RE.fullmatch(l or "")]
    if len(candidates) < 2:
        return None
    line2 = candidates[1] if len(candidates) > 1 else candidates[0]
    if len(line2) < 10:
        return None
    doc_number = line2[0:9]
    check_digit = line2[9:10]
    if not check_digit.isdigit():
        return None
    computed = _mrz_check_digit(doc_number)
    return {"mrz_line_found": True, "checksum_valid": computed == int(check_digit)}


def analyze_document_image(image_bytes: bytes) -> dict:
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return {"error": "Could not decode image. Please upload a JPEG/PNG."}

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = img.shape[:2]
    blur = _blur_score(gray)
    quality_flags = []
    if blur < 60:
        quality_flags.append("Image looks blurry — ask the user to retake it")
    if min(h, w) < 400:
        quality_flags.append("Resolution is low — extraction may be unreliable")

    raw_text = pytesseract.image_to_string(gray)
    doc_type = guess_document_type(raw_text)
    fields = {
        "document_type_guess": doc_type,
        "cid_number_guess": next(iter(CID_RE.findall(raw_text)), None),
        "passport_number_guess": next(iter(PASSPORT_RE.findall(raw_text)), None),
        "date_guess": next(iter(DATE_RE.findall(raw_text)), None),
        "name_guess": guess_name(raw_text),
    }
    mrz = try_validate_mrz(raw_text) if doc_type == "passport" else None

    face_row = _detect_best_face(img)
    face_thumbnail = None
    embedding = None
    if face_row is not None:
        _, recognizer = get_models()
        aligned = recognizer.alignCrop(img, face_row)
        embedding = recognizer.feature(aligned)
        face_thumbnail = _encode_jpg_b64(aligned)
    else:
        quality_flags.append("No face detected on the document — ask for a clearer photo of the photo page/front")

    return {
        "quality": {"blur_score": blur, "width": w, "height": h, "flags": quality_flags},
        "raw_text_preview": raw_text.strip()[:500],
        "fields": fields,
        "mrz": mrz,
        "face_found": face_row is not None,
        "face_thumbnail": face_thumbnail,
        "_embedding": embedding,  # server-side only, stripped before sending to client
    }


# ---- Liveness challenge-response --------------------------------------------

CHALLENGE_POOL = ["turn_left", "turn_right", "smile"]
CHALLENGE_LABELS = {
    "neutral": "Look straight at the camera",
    "turn_left": "Slowly turn your head to your LEFT",
    "turn_right": "Slowly turn your head to your RIGHT",
    "smile": "Smile",
}


def new_challenge_sequence():
    rest = random.sample(CHALLENGE_POOL, k=2)
    return ["neutral"] + rest


def _sample_frames(video_path, target_count=90):
    """Sample up to ~target_count frames spread evenly across the clip.

    target_count has to scale with clip length, not just with "how many frames
    feel like enough": the challenges are scored on the most extreme frame in
    each segment, so sampling too sparsely in time can step straight over the
    peak of a head turn and fail a user who did exactly as asked.

    Read sequentially with a stride rather than seeking to each sample. Browser
    MediaRecorder webm carries sparse keyframes, so a seek per sample forces a
    re-decode from the preceding keyframe — affordable at 30 samples of a short
    clip, not at 90 of a longer one.

    Deliberately ignores CAP_PROP_FRAME_COUNT and derives the stride from the
    frames actually decoded. The challenge flow pauses the recorder between
    challenges, which leaves timestamp gaps in the webm; decoders faced with
    those tend to report a count derived from the clip's wall-clock duration
    (paused time included) rather than its real frame count, which would
    overestimate the stride and quietly halve the sample density. Thinning as we
    go instead keeps spacing perfectly even, needs no count up front, and bounds
    memory at ~2x target_count frames (holding a whole clip raw is ~1MB/frame).
    """
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    stride = 1
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % stride == 0:
            frames.append(frame)
            if len(frames) > 2 * target_count:
                frames = frames[::2]
                stride *= 2
        idx += 1
    cap.release()
    return frames


def analyze_liveness_video(video_bytes: bytes, expected_sequence, reference_embedding):
    with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as tmp:
        tmp.write(video_bytes)
        tmp_path = tmp.name

    frames = _sample_frames(tmp_path)
    Path(tmp_path).unlink(missing_ok=True)

    if not frames:
        return {"error": "Could not read any frames from the uploaded video."}

    n_segments = len(expected_sequence)
    seg_size = max(len(frames) // n_segments, 1)
    segments = [frames[i * seg_size:(i + 1) * seg_size] for i in range(n_segments)]
    segments[-1] = frames[(n_segments - 1) * seg_size:]  # last segment gets the remainder

    per_frame_metrics = []  # (segment_index, metrics dict or None, raw frame)
    for seg_i, seg_frames in enumerate(segments):
        for frame in seg_frames:
            face_row = _detect_best_face(frame)
            metrics = _face_metrics(face_row) if face_row is not None else None
            per_frame_metrics.append((seg_i, metrics, frame))

    detected = [m for _, m, _ in per_frame_metrics if m is not None]
    face_detection_rate = len(detected) / max(len(per_frame_metrics), 1)

    baseline_seg = [m for seg_i, m, _ in per_frame_metrics if seg_i == 0 and m is not None]
    baseline_smile = np.mean([m["smile_ratio"] for m in baseline_seg]) if baseline_seg else None
    baseline_offset = np.mean([m["nose_offset"] for m in baseline_seg]) if baseline_seg else None

    challenge_results = []
    for seg_i, challenge in enumerate(expected_sequence):
        seg_metrics = [m for si, m, _ in per_frame_metrics if si == seg_i and m is not None]
        if not seg_metrics:
            challenge_results.append({"challenge": challenge, "passed": False, "reason": "No face detected during this segment"})
            continue
        offsets = [m["nose_offset"] for m in seg_metrics]
        smiles = [m["smile_ratio"] for m in seg_metrics]
        if challenge == "neutral":
            challenge_results.append({"challenge": challenge, "passed": True, "reason": "Baseline captured", "value": {"nose_offset": float(np.mean(offsets)), "smile_ratio": float(np.mean(smiles))}})
        elif challenge == "turn_left":
            extreme = min(offsets)
            passed = baseline_offset is not None and (extreme - baseline_offset) < -0.18
            challenge_results.append({"challenge": challenge, "passed": bool(passed), "value": {"min_offset": float(extreme), "baseline": float(baseline_offset) if baseline_offset is not None else None}})
        elif challenge == "turn_right":
            extreme = max(offsets)
            passed = baseline_offset is not None and (extreme - baseline_offset) > 0.18
            challenge_results.append({"challenge": challenge, "passed": bool(passed), "value": {"max_offset": float(extreme), "baseline": float(baseline_offset) if baseline_offset is not None else None}})
        elif challenge == "smile":
            extreme = max(smiles)
            passed = baseline_smile is not None and (extreme / baseline_smile) > 1.12
            challenge_results.append({"challenge": challenge, "passed": bool(passed), "value": {"max_smile_ratio": float(extreme), "baseline": float(baseline_smile) if baseline_smile is not None else None}})

    non_neutral = [r for r in challenge_results if r["challenge"] != "neutral"]
    liveness_passed = bool(non_neutral) and all(r["passed"] for r in non_neutral) and face_detection_rate > 0.5

    # Best (near-frontal, highest-confidence) frame for matching
    best_seg0 = [(m, f) for si, m, f in per_frame_metrics if si == 0 and m is not None]
    if best_seg0:
        best_metrics, best_frame = max(best_seg0, key=lambda mf: mf[0]["score"])
    else:
        scored = [(m, f) for _, m, f in per_frame_metrics if m is not None]
        best_metrics, best_frame = max(scored, key=lambda mf: mf[0]["score"]) if scored else (None, None)

    match_result = {"match_score": None, "band": "no_face"}
    selfie_thumbnail = None
    if best_frame is not None and reference_embedding is not None:
        _, recognizer = get_models()
        face_row = _detect_best_face(best_frame)
        aligned = recognizer.alignCrop(best_frame, face_row)
        selfie_embedding = recognizer.feature(aligned)
        selfie_thumbnail = _encode_jpg_b64(aligned)
        score = float(recognizer.match(reference_embedding, selfie_embedding, cv2.FaceRecognizerSF_FR_COSINE))
        if score >= 0.45:
            band = "match"
        elif score >= 0.30:
            band = "review"
        else:
            band = "no_match"
        match_result = {"match_score": score, "band": band}

    # Soft, best-effort spoof heuristic: near-zero motion across the whole clip
    # despite challenges supposedly happening is suspicious (e.g. a static photo
    # held up to the camera). This is NOT a certified presentation-attack model.
    if len(detected) >= 2:
        offsets_all = [m["nose_offset"] for m in detected]
        motion = float(np.std(offsets_all))
        spoof_flag = motion < 0.02
    else:
        motion = None
        spoof_flag = True  # can't verify — treat conservatively

    return {
        "challenge_results": challenge_results,
        "liveness_passed": liveness_passed,
        "face_detection_rate": face_detection_rate,
        "motion_std": motion,
        "spoof_flag": spoof_flag,
        "match": match_result,
        "selfie_thumbnail": selfie_thumbnail,
    }


# ---- Decision engine ---------------------------------------------------------

def decide(document_result: dict, liveness_result: dict) -> dict:
    reasons = []
    if not document_result.get("face_found"):
        return {"decision": "REJECTED", "reasons": ["No face detected on the submitted document"]}
    if liveness_result.get("error"):
        return {"decision": "REJECTED", "reasons": [liveness_result["error"]]}

    match = liveness_result.get("match", {})
    band = match.get("band")

    if document_result.get("quality", {}).get("flags"):
        reasons.extend(document_result["quality"]["flags"])

    if not liveness_result.get("liveness_passed"):
        reasons.append("Liveness challenges were not clearly completed")
    if liveness_result.get("spoof_flag"):
        reasons.append("Little to no facial motion detected across the clip (possible static image/replay) — heuristic only, not certified PAD")

    if band == "no_match":
        return {"decision": "REJECTED", "reasons": reasons + [f"Face match score too low ({match.get('match_score'):.3f})"], "match_score": match.get("match_score")}

    if band == "match" and liveness_result.get("liveness_passed") and not liveness_result.get("spoof_flag") and not document_result.get("quality", {}).get("flags"):
        return {"decision": "APPROVED", "reasons": ["All checks passed"], "match_score": match.get("match_score")}

    # Anything else (borderline match, failed challenge, quality flags, spoof
    # flag) is routed to a human — this is a deliberate design choice, not a
    # fallback: see the design notes on aged-photo matching.
    if not reasons:
        reasons.append("Borderline face-match score — routed for manual review")
    return {"decision": "MANUAL_REVIEW", "reasons": reasons, "match_score": match.get("match_score")}
