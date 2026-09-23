Paste everything below into Claude Code, in the root of this project, to continue the build.

---

I have a working prototype of a Bhutan document + liveness identity-verification pipeline in this
repo (`backend/app.py`, `backend/vision.py`, `frontend/index.html`). Read `README.md` first — it
lists exactly what's simplified and what's missing; treat that list as the actual backlog, not a
disclaimer to ignore.

Current state: FastAPI backend, in-memory verification store, OpenCV YuNet (detection) + SFace
(recognition) for face matching, Tesseract for OCR, a landmark-heuristic liveness check using only
the 5 points YuNet returns (head-turn/smile, no blink), and a three-band decision engine
(APPROVED / MANUAL_REVIEW / REJECTED). Everything is open-source and free to use commercially
(YuNet/SFace are Apache-2.0). This is a demo, not production — there's no auth, no persistence, no
encryption, and no connection to any Bhutanese government registry.

Constraints for this project, please keep these front of mind for every change:
- Prefer free/open-source components. If you introduce a new pretrained model (face recognition,
  anti-spoofing, OCR), check and tell me its license before wiring it in — some strong face-recognition
  weights are research-only even when the surrounding library is permissively licensed.
- This will eventually handle real national-ID numbers and biometric data. Don't add features that
  make that worse (e.g., logging raw images/video, storing embeddings in plaintext) without flagging it.
- Keep the manual-review path a first-class outcome, not an edge case — face-matching against a
  10-15-year-old ID photo is expected to land there often, by design, not as a bug.

Work through these in order, and after each one, tell me what you changed and what you verified
(don't just say it works — show me the test/curl output or describe what you ran):

1. **Persistence.** Replace the in-memory `STORE` dict in `app.py` with SQLite (or Postgres if you
   think it's warranted) so verification records survive a restart. Store every sub-score alongside
   the final decision — this is the audit trail a regulator would ask for later.

2. **Real blink detection.** The current liveness check can't do this because YuNet only returns 5
   landmark points. Add MediaPipe's Face Mesh (or another open landmark model if you hit the same
   network issue I did trying to fetch Google's model bundle) to get proper eye-aspect-ratio blink
   detection, and add "blink" to the challenge vocabulary in `vision.py`'s `CHALLENGE_POOL`.

3. **Document authenticity heuristics.** Add error-level analysis (ELA) to catch localized
   JPEG re-compression consistent with photo editing, and a basic screen/photo-recapture check
   (moiré pattern detection via FFT, or specular-glare heuristics). Surface both as additional
   `quality.flags` entries, not hard rejects — they should push toward manual review, not auto-reject.

4. **Per-document-type field extraction.** Right now OCR field-guessing is generic regex across any
   document. Bhutan's CID card has a fixed layout that differs across its 1st-4th generations
   (research this — the 4th-gen card has an NDI "biocrypto" QR code, which is a different
   verification path entirely, not something to OCR). Add template-based field-region extraction per
   known layout, with the current regex approach kept as the fallback for unrecognized layouts.

5. **API hardening.** Add basic auth (API key is fine for now), rate limiting, and input validation
   (file size/type limits on uploads — right now a malicious upload could be huge or not actually be
   an image/video). Add a `/api/verify/{id}` DELETE endpoint so a caller can purge a record.

6. **Tests.** Add a test suite (pytest) that exercises the full flow with synthetic fixtures — see how
   I smoke-tested this originally (a public-domain test photo, a synthetically generated head-turn
   clip via ffmpeg) if you need a pattern for generating fixtures without real biometric data.

7. **Bhutan NDI exploration (separate track, not blocking the above).** Look at the
   `Bhutan-NDI` GitHub org (`ngotag-platform`, `ngotag-agent-controller`) and sketch what a verifier
   integration would look like as an alternative/complementary path to this OCR+liveness pipeline for
   anyone already enrolled in NDI. Don't build it yet — just come back with a concrete integration plan
   and what access/onboarding it would require.

Ask me before making any change that would require a paid service, an account/API key I haven't
provided, or a license I haven't approved.
