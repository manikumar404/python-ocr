"""
Scores backend/documents.py against the labelled sample set.

    .venv/bin/python tools/evaluate_documents.py            # everything
    .venv/bin/python tools/evaluate_documents.py permit IMG_12   # files matching any filter
    SHOW=1 .venv/bin/python tools/evaluate_documents.py     # also list every wrong reading

Reads two files from "doc samples/" (git-ignored: they hold real people's
documents and numbers, and must never be committed):

    labels.csv          file, type, side, md5   (the document type of each image)
    ground_truth.json   {file: {field: expected value}} (numbers transcribed by hand)

Reports type accuracy and, per field, how many numbers were read correctly,
read wrongly, or not found, split by the confidence status the pipeline gave
them. "Silent" errors (wrong but marked verified) are the ones that matter
most: every change should keep that count at or near zero.
"""
import csv
import json
import os
import re
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "backend"))
import documents  # noqa: E402
import vision  # noqa: E402

SAMPLES = REPO / "doc samples"

# labels.csv uses coarser type names than documents.py; map them across.
FOREIGN_ID = {"FOREIGN": "hk_id", "Germany": "de_id", "India": "in_aadhaar"}


def expected_type(row):
    t, side, name = row["type"], row["side"], row["file"]
    if t == "bt_cid":
        return "bt_cid_" + side
    if t == "bt_special_residence":
        return "bt_srp_" + side
    if t == "foreign_id":
        return next(v for k, v in FOREIGN_ID.items() if k in name)
    if t == "not_id":
        return None
    return t


def norm(v):
    return re.sub(r"[^A-Z0-9]", "", str(v).upper()) if v else v


def main(filters):
    rows = list({r["md5"]: r for r in csv.DictReader(open(SAMPLES / "labels.csv"))}.values())
    if filters:
        rows = [r for r in rows if any(f in r["file"] for f in filters)]
    truth = json.loads((SAMPLES / "ground_truth.json").read_text())

    vision.get_models()
    lock = threading.Lock()  # the shared face detector isn't thread-safe

    def detect(img):
        with lock:
            return vision._detect_best_face(img)

    def run(row):
        start = time.time()
        _, _, result = documents.read_document(cv2.imread(str(SAMPLES / row["file"])), detect)
        return row, result, time.time() - start

    with ThreadPoolExecutor(int(os.environ.get("JOBS", 6))) as ex:
        results = list(ex.map(run, rows))

    type_ok, type_total, wrong_types = Counter(), Counter(), []
    fields, silent, wrong_reads = Counter(), [], []
    for row, res, _ in results:
        want, got = expected_type(row), res["document_type"]
        type_total[want] += 1
        type_ok[want] += want == got
        if want != got:
            wrong_types.append((row["file"], want, got))
        for key, value in truth.get(row["file"], {}).items():
            read, status = res["fields"].get(key), res["field_checks"].get(key)
            outcome = "missing" if not read else "correct" if norm(read) == norm(value) else "wrong"
            fields[(row["type"], key, outcome)] += 1
            if outcome == "wrong":
                wrong_reads.append((status, row["file"], key, value, read))
                if status in ("format_ok", "checksum_ok"):
                    silent.append((row["file"], key, value, read))

    print("Document type")
    for t in sorted(type_total, key=str):
        print(f"  {str(t):22s} {type_ok[t]:3d}/{type_total[t]:3d}")
    print(f"  TOTAL {sum(type_ok.values())}/{sum(type_total.values())}")
    for w in wrong_types:
        print("  misidentified:", w)

    print("\nNumbers (against ground_truth.json)")
    totals = Counter()
    for t, k in sorted({(t, k) for t, k, _ in fields}):
        c, w, m = (fields[(t, k, o)] for o in ("correct", "wrong", "missing"))
        totals.update(correct=c, wrong=w, missing=m)
        print(f"  {t:20s} {k:17s} correct {c:3d}  wrong {w:3d}  missing {m:3d}")
    print(f"  TOTAL correct {totals['correct']}  wrong {totals['wrong']}  missing {totals['missing']}")
    print(f"  wrong but marked verified: {len(silent)}")
    for s in silent:
        print("   ", s)
    if os.environ.get("SHOW"):
        for w in wrong_reads:
            print("  wrong:", w)

    secs = sorted(t for _, _, t in results)
    print(f"\nSeconds per document (with {os.environ.get('JOBS', 6)} in parallel): "
          f"median {secs[len(secs) // 2]:.2f}, max {secs[-1]:.2f}")


if __name__ == "__main__":
    main(sys.argv[1:])
