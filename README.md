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

## Deploying (Render)

Render is a container host, so this image runs as-is — nothing about the app changes to deploy it.

Create the service **manually**, not from a Blueprint. Render's free tier covers web services, but
Blueprints are not listed among the free-tier features and the Blueprint flow asks for a payment
method. A hand-created web service on the free instance type does not.

1. Push this repo to GitHub (it already lives at `manikumar404/python-ocr`).
2. [dashboard.render.com](https://dashboard.render.com) → **New** → **Web Service** → pick the repo.
3. Set these in the form:
   - **Language**: `Docker` (Render detects the Dockerfile; do not pick Python — the app needs apt
     packages that a native Python runtime cannot install)
   - **Branch**: `main`
   - **Region**: Singapore (closest to Bhutan)
   - **Instance type**: `Free`
   - **Advanced → Health Check Path**: `/api/health`
4. **Create Web Service**. First build takes several minutes (apt packages, OpenCV wheel, the 37MB
   model download). When it goes live you get an `https://<name>.onrender.com` URL.

`render.yaml` is kept in the repo as an accurate record of that configuration, and works directly if
you ever have billing enabled — but the dashboard form above is the card-free route.

The HTTPS that Render terminates for you is what makes this deployable at all: browser webcam
access requires `localhost` or TLS, which is exactly why `docker-compose.yml` binds to `127.0.0.1`
locally. On Render the camera step works from any machine.

Two things to know before demoing it:

- **Verification state is still an in-memory dict.** A redeploy, a crash, or Render restarting the
  instance drops every in-progress session. Fine for a demo, and the same caveat as running locally.
- **Sizing.** One in-flight verification peaks at roughly 280MB RSS with both ONNX models resident,
  so the free instance type (512MB) holds a demo session comfortably. Free instances spin down after
  15 minutes idle and take about a minute to wake, so load the URL once before demoing. Two people
  running the flow simultaneously would crowd 512MB; that is the point to consider a paid plan.

### Why not Vercel / Netlify / other serverless hosts

They cannot run this, and the failure is not a configuration gap:

- `pytesseract` is a wrapper around the `tesseract` **binary**, an apt package. Serverless Python
  runtimes have no system package manager.
- `opencv-contrib-python` links against `libGL`/`libglib`, which those runtimes do not ship.
- The 808MB image is far past the ~250MB unzipped function limit these platforms enforce.
- The in-memory session store assumes one long-lived process. Serverless invocations are ephemeral
  and independent, so `start → document → challenge → liveness → result` would break as soon as two
  requests landed on different instances.

Any container host works instead — Fly.io, Railway, and Google Cloud Run all take this Dockerfile
unchanged; only the `PORT` convention and the sizing knob differ.

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

## Supported documents

`backend/documents.py` identifies the document from what's printed on it (its title, field labels,
number formats, MRZ), then re-reads the ID number from a targeted crop. Photos can be sideways or
upside down: the portrait photo is always printed upright on the card, so the rotation where the
face detector finds an upright face is used.

| Document | Extracted | How it's checked |
|---|---|---|
| Work permit (current and older design) | Permit no., issue / expiry dates | Number = dzongkhag prefix + processing date (YYMMDD) + serial, and its date must be near the printed issue date |
| Student / trader / dependent permit, immigration card | `SP`/`TP`/`DP`/`MC` number, issue / expiry dates | Type-specific format |
| CID card, front | Citizenship ID No. | `1` + dzongkhag (01–20) + 8 digits |
| CID card, back | Card no. (`B`/`C` + 10 digits), household no., issue / expiry dates | Format; household no. starts with the dzongkhag code |
| Special residence permit | SRP No. (front), issue / expiry dates (back) | `3` + dzongkhag + 8 digits |
| Driving licence | Licence no., CID, issue / expiry dates | Format |
| Voter photo ID | Citizen ID No. | As CID |
| Bhutan passport | Passport no., CID, birth / issue / expiry dates | MRZ check digits (the CID is the MRZ personal number) |
| Foreign passport | Passport no., nationality, birth / issue / expiry dates | MRZ check digits |
| Hong Kong ID, German ID, Aadhaar | ID no. | HKID check digit, MRZ, Verhoeff |

Dates are returned as `YYYY-MM-DD`. Permit dates are voted across several OCR readings and must be in
order and at most 3 years apart. CID cards, SRP cards and passports in the samples all expire exactly
5 or 10 years after issue, minus a day (the licence: exactly 10 years), so an issue/expiry pair read
that far apart confirms both. A passport's MRZ has no issue date, so it's taken from the printed page:
the date one validity term before the check-digit-confirmed expiry. A document whose expiry date has
passed is flagged as expired, which routes it to manual review.

Anything else (bank cards, loyalty cards, forms) is reported as *not a supported document*.

Every number comes back with a status: **verified** (several OCR readings agreed, or a check digit
confirmed it), **check manually** (read but not confirmed), or **not found**. Anything short of
verified adds a flag, which routes the verification to manual review rather than approval.

**Measured accuracy** on 176 sample photos (real phone photos and scans, many rotated or blurry):
document type correct for 172 (the four misses are two scans too blurred to read by eye and two card
backs with no readable text). Against 442 hand-checked numbers and dates, 352 were read correctly,
17 wrongly and 73 not found — and of the values marked *verified*, 1 was wrong. The previous
generic OCR identified the type correctly for 30 of the 176.

The QR codes printed on the work, student and dependent permits in the samples all decode to the
same placeholder URL (`http://www.qrstuff.com/`), so they carry nothing that can be checked.

### Re-measuring accuracy

The sample photos live in `doc samples/`, which is git-ignored because they are real people's IDs.
With `labels.csv` (type of each image) and `ground_truth.json` (hand-checked numbers) in that
folder:

```bash
.venv/bin/python tools/evaluate_documents.py          # all samples
SHOW=1 .venv/bin/python tools/evaluate_documents.py   # also list every wrong reading
```

Run it after any change to `documents.py`; the number that matters most is *wrong but marked
verified*.

### Speed

Reading a document takes about 1.6 s on one core of a laptop (90% under 2.5 s, worst case ~10 s):
several OCR passes run instead of one. Render's free instance has only a fraction of a CPU, so
expect roughly ten times that there; a paid instance with a full CPU brings it back to laptop speed.

## Architecture map (what's in this repo vs. the full design)

| Pipeline stage | This prototype | Full design notes |
|---|---|---|
| Document OCR | Tesseract with per-type identification, targeted number crops, multi-read voting and MRZ parsing (see [Supported documents](#supported-documents)) | Same idea, with template homography for each card generation |
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
  documents.py      Document-type identification and ID-number extraction
  vision.py         Face detection/matching, liveness scoring, decision engine
  models/           YuNet + SFace ONNX weights (bundled, Apache-2.0)
  requirements.txt
frontend/
  index.html        Single-page demo UI (vanilla JS, no build step, no external calls)
tools/
  evaluate_documents.py   Accuracy report against the (git-ignored) sample set
```
