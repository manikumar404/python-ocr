"""
Demo API for the Bhutan document + liveness verification prototype.

Run with:  uvicorn app:app --reload --port 8000
Then open http://localhost:8000 in a browser (webcam access needs
either localhost or HTTPS — localhost is fine).

This keeps verification state in memory (a plain dict) for simplicity.
A production version would use a real datastore, real auth, TLS, and
would not return raw embeddings/images across process restarts.
"""
import uuid
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

import vision

app = FastAPI(title="Bhutan KYC Prototype API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

STORE = {}  # verification_id -> dict
FRONTEND_INDEX = Path(__file__).parent.parent / "frontend" / "index.html"


@app.get("/", response_class=HTMLResponse)
def index():
    return FRONTEND_INDEX.read_text()


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.post("/api/verify/start")
def start_verification():
    vid = str(uuid.uuid4())
    STORE[vid] = {"document": None, "challenge": None, "liveness": None}
    return {"verification_id": vid}


def _get(vid):
    if vid not in STORE:
        raise HTTPException(404, "Unknown verification_id — call /api/verify/start first")
    return STORE[vid]


@app.post("/api/verify/{vid}/document")
async def upload_document(vid: str, file: UploadFile = File(...)):
    entry = _get(vid)
    image_bytes = await file.read()
    result = vision.analyze_document_image(image_bytes)
    if "error" in result:
        raise HTTPException(400, result["error"])
    entry["document"] = result  # keeps _embedding server-side
    public_result = {k: v for k, v in result.items() if not k.startswith("_")}
    return public_result


@app.get("/api/verify/{vid}/challenge")
def get_challenge(vid: str):
    entry = _get(vid)
    if entry["document"] is None:
        raise HTTPException(400, "Upload a document first")
    sequence = vision.new_challenge_sequence()
    entry["challenge"] = sequence
    return {
        "sequence": sequence,
        "labels": [vision.CHALLENGE_LABELS[c] for c in sequence],
    }


@app.post("/api/verify/{vid}/liveness")
async def upload_liveness(vid: str, file: UploadFile = File(...)):
    entry = _get(vid)
    if entry["document"] is None or entry["challenge"] is None:
        raise HTTPException(400, "Upload a document and fetch a challenge sequence first")
    video_bytes = await file.read()
    reference_embedding = entry["document"].get("_embedding")
    result = vision.analyze_liveness_video(video_bytes, entry["challenge"], reference_embedding)
    if "error" in result:
        raise HTTPException(400, result["error"])
    entry["liveness"] = result
    decision = vision.decide(entry["document"], result)
    entry["decision"] = decision
    public_result = {k: v for k, v in result.items()}
    return {"liveness": public_result, "decision": decision}


@app.get("/api/verify/{vid}/result")
def get_result(vid: str):
    entry = _get(vid)
    if entry.get("decision") is None:
        raise HTTPException(400, "No decision yet — complete the document and liveness steps first")
    doc_public = {k: v for k, v in entry["document"].items() if not k.startswith("_")}
    return {
        "document": doc_public,
        "liveness": entry["liveness"],
        "decision": entry["decision"],
    }
