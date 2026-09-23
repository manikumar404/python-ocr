# Bhutan Document + Liveness Verification — Prototype

A working demo of the document-OCR → active-liveness → face-match pipeline discussed in the design
notes. Runs entirely on free/open-source components (OpenCV's YuNet + SFace models, Tesseract OCR,
FastAPI) — no cloud vision APIs, no paid vendor, no external calls at runtime once the two small
model files are downloaded.

**Read this whole README before demoing it to anyone** — the "what this is not" section at the
bottom is exactly what you should say out loud to stakeholders so the demo doesn't get read as more
finished than it is.

## Quick start (Docker)

The container already contains Tesseract, OpenCV's shared libraries, and **both** model files, so
this is the shortest path to a working demo and needs nothing installed but Docker:

```bash
docker compose up --build          # first build takes a few minutes
# then open http://localhost:8000 in Chrome or Firefox (needs a webcam)
```

Or without Compose:

```bash
docker build -t bhutan-kyc-demo .
docker run --rm -p 127.0.0.1:8000:8000 bhutan-kyc-demo
```

Two notes specific to the container:

- The ~37MB face-recognition model is downloaded **at build time**, not on first request, so a
  running container makes no external calls at all. The build fails loudly if that download fails,
  rather than producing an image that breaks on the first face match.
- The port is deliberately published on `127.0.0.1` only. Browser webcam access requires
  `localhost` or HTTPS, so exposing it on your LAN would just yield a page whose camera step fails.
  To demo from another machine, put a TLS-terminating reverse proxy in front of it.

Verification state lives in memory, so restarting the container drops any in-progress sessions —
fine for a demo, and called out again under "what this is not" below.

## Quick start (local Python)

```bash
# 1. System dependency: Tesseract's OCR engine (pytesseract is just a wrapper around this binary)
#    macOS:   brew install tesseract
#    Ubuntu:  sudo apt install tesseract-ocr
#    Windows: https://github.com/UB-Mannheim/tesseract/wiki

# 2. Python dependencies
cd backend
pip install -r requirements.txt

# 3. Run
uvicorn app:app --reload --port 8000

# 4. Open http://localhost:8000 in Chrome or Firefox (needs a webcam)
```

The small face-detection model (`face_detection_yunet.onnx`, ~230KB) is bundled in `backend/models/`.
The face-recognition model (`face_recognition_sface.onnx`, ~37MB) is downloaded automatically the
first time you run the app (so it isn't sitting in every copy of this zip) — you'll see a one-line
log message the first time `vision.py` loads it, and it's cached in `backend/models/` after that, so
it only happens once. Both are from the [OpenCV Zoo](https://github.com/opencv/opencv_zoo) under its
Apache-2.0 license, so they're fine to use commercially, unlike some of the other open
face-recognition weights flagged as a licensing risk in the design notes. If your machine has no
internet access at runtime, download it once yourself from
`https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx`
and place it at `backend/models/face_recognition_sface.onnx`.

Webcam access requires either `localhost` (what you get by default above) or HTTPS — it will not
work if you try to open the page over plain `http://` from another machine on your network.

## What to actually show a stakeholder

A five-minute walkthrough that demonstrates the real mechanism, not a canned video:

1. Upload any ID-like photo (a real Bhutanese CID/passport/licence if you have a sample with consent
   to use it, or any government-ID-shaped photo for the demo) and show the extracted fields, the
   cropped face, and the quality/authenticity flags appearing live.
2. Click into the liveness step and actually do it on camera — the point to make out loud is that the
   two-action sequence is **randomized per session**, so a pre-recorded video of you can't be replayed
   for a different session and expect to pass.
3. Show the final decision screen: the match score, the pass/fail per challenge, and — importantly —
   demonstrate what a **manual-review** outcome looks like (e.g. by holding the ID up out of frame
   during one challenge, or looking away), not just an approval. Stakeholders should see the review
   path exists, since that's the honest answer to "what happens with a 12-year-old ID photo."

## Architecture map (what's in this repo vs. the full design)

| Pipeline stage | This prototype | Full design notes |
|---|---|---|
| Document OCR | Tesseract + regex field-guessing, generic across doc types | Same idea, with per-generation CID template regions and dedicated MRZ parsing |
| Document authenticity | Blur/resolution check + MRZ checksum only | Adds ELA tamper detection, template homography, screen-recapture detection |
| Liveness | Active challenge (head-turn/smile) scored from OpenCV's 5-point face landmarks | Adds MediaPipe Face Mesh for real blink/EAR detection and a certified/production-grade passive anti-spoofing model |
| Face match | OpenCV SFace (cosine similarity) | InsightFace/ArcFace or a commercial vendor, chosen after checking model licensing |
| Decision engine | Hard-coded thresholds → approve/reject/manual-review | Same three-band shape, thresholds tuned against a real validation set |
| Bhutan NDI integration | Not attempted here | Recommended as the primary path for anyone already enrolled in NDI — see the design notes |

## What this is NOT (say this part out loud)

- **Not connected to any Bhutanese government system.** It cannot tell you whether a CID number,
  passport number, or licence number is genuine or has been revoked — only that the image looks
  structurally plausible. That check requires the Bhutan NDI integration path described in the design
  notes, or direct data-sharing agreements with DCRC/Immigration/RSTA.
- **No blink detection.** OpenCV's YuNet gives 5 landmark points (eyes, nose, mouth corners) — enough
  for head-turn and smile detection, not enough for eye-aspect-ratio blink detection. MediaPipe's Face
  Mesh would add that, but its model bundle wasn't reachable from this build environment's network, so
  it's left as a documented next step rather than silently faked.
- **The anti-spoofing "spoof_flag" is a soft heuristic** (near-zero motion across the clip), not a
  certified presentation-attack-detection model. A moderately sophisticated attack (e.g. a video
  replay with some motion) would not reliably be caught by this build.
- **Face-match thresholds are placeholders**, anchored only to the SFace project's own published
  reference point (~0.363 cosine similarity), not tuned against a real validation set — and matching
  a photo that's 10-15 years old will genuinely perform worse than the same code matching a fresh
  photo, for every vendor, not just this one.
- **In-memory storage only.** Restarting the server loses every verification. There's no encryption
  at rest, no auth on the API, and no audit-log persistence — all required before this touches real
  user data.

## Project layout

```
backend/
  app.py            FastAPI routes + in-memory verification store
  vision.py         OCR, face detection/matching, liveness scoring, decision engine
  models/           YuNet + SFace ONNX weights (bundled, Apache-2.0)
  requirements.txt
frontend/
  index.html        Single-page demo UI (vanilla JS, no build step, no external calls)
```
