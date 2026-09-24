"""
Document-type identification and ID-number extraction.

The pipeline, in order:

1. Orientation. Phone photos of cards arrive rotated by 90/180/270 degrees
   about as often as not, and Tesseract's own orientation detection is
   unreliable on these cards (the Dzongkha header and guilloche patterns
   confuse it). Every supported document carries a portrait photo, and the
   photo is always printed upright relative to the card, so the rotation at
   which the face detector finds an upright face is the card's orientation.

2. Scale. The face also gives a scale: text size on these cards is roughly
   proportional to the photo, so resizing to a fixed face width puts the text
   near the size Tesseract reads best, whether the card fills the frame or is
   a small rectangle on a large table.

3. Classification. One sparse-text OCR pass over the whole card, matched
   against each type's printed title and field labels ("WORK PERMIT",
   "Citizenship ID No.", "Household No." ...). Matching is fuzzy because OCR
   routinely drops or swaps a letter in these headings.

4. Extraction. The ID number is re-read from a targeted crop with filtering
   chosen for that document type, because a whole-card pass reads numbers
   poorly: they sit on security patterns, and on the permits they're printed
   in red. The crop is OCR'd under several binarizations and scales and the
   readings vote; a reading only counts if it fits the number's structure
   (CID and household numbers start with a dzongkhag code 01-20; work permit
   numbers embed their processing date). Check digits are used where the
   document has them (passport MRZ, Hong Kong ID, Aadhaar).

Each field comes back with a status in `field_checks`:

    checksum_ok / format_ok   several readings agreed, or a check digit (or,
                              for dates, a whole validity term between issue
                              and expiry) confirmed it. On the sample set,
                              one such value was wrong.
    needs_review              a value was read but not confirmed; many of
                              these are wrong. Show it, but have a person
                              compare it with the document.
    checksum_failed           the check digit disagrees (a misread, or a
                              specimen document with a fake number).
    not_found / unchecked     nothing readable / no check exists (names).

Everything here is best-effort OCR on a prototype. A "verified" number was
read consistently; it has not been confirmed against any registry.
"""
import re
from dataclasses import dataclass, field
from datetime import date, timedelta
from difflib import SequenceMatcher

import cv2
import numpy as np
import pytesseract

# ---- Orientation and scale ---------------------------------------------------

_ROTATIONS = {
    0: None,
    90: cv2.ROTATE_90_CLOCKWISE,
    180: cv2.ROTATE_180,
    270: cv2.ROTATE_90_COUNTERCLOCKWISE,
}

# Face width (px) the card is resized to before OCR. Chosen on the sample set:
# at this size the body text on permits and CID cards is ~25-30px tall.
_TARGET_FACE_WIDTH = 130


def _rotate(img, angle):
    code = _ROTATIONS[angle]
    return img if code is None else cv2.rotate(img, code)


def orient(img, detect_face):
    """Returns (upright image, rotation applied, face row in upright coords).

    YuNet also fires on upside-down faces, just with a lower score, so the
    rotation with the best-scoring face wins rather than the first one found.
    Detection runs on a downscaled copy: four passes at full resolution would
    cost far more than the orientation decision needs.
    """
    s = min(1.0, 640 / max(img.shape[:2]))
    small = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else img
    best_angle, best_face = 0, None
    for angle in _ROTATIONS:
        face = detect_face(_rotate(small, angle))
        if face is not None and (best_face is None or face[-1] > best_face[-1]):
            best_angle, best_face = angle, face
    if best_face is not None:
        best_face = best_face.copy()
        best_face[:14] /= s
    return _rotate(img, best_angle), best_angle, best_face


def _normalize_scale(img, face):
    long_side = max(img.shape[:2])
    if face is not None:
        k = _TARGET_FACE_WIDTH / max(float(face[2]), 1.0)
        # How tightly the photo is framed varies by person, so the face is only
        # a rough ruler. Clamp the result: past ~2400px the upscaling blur
        # costs more than the size gains, and below ~1000px small print is lost.
        k = min(max(k, 1000 / long_side), 2400 / long_side)
    else:
        # No face (card backs, non-ID cards): fall back to the long side.
        k = 1400 / long_side
    k = min(k, 4.0)
    interp = cv2.INTER_CUBIC if k > 1 else cv2.INTER_AREA
    return cv2.resize(img, None, fx=k, fy=k, interpolation=interp), k


# ---- OCR helpers -------------------------------------------------------------

@dataclass
class Word:
    text: str
    conf: float
    x: int
    y: int
    w: int
    h: int
    line: tuple


def ocr_words(gray, psm=11, extra=""):
    data = pytesseract.image_to_data(gray, config=f"--psm {psm} {extra}", output_type=pytesseract.Output.DICT)
    words = []
    for i, t in enumerate(data["text"]):
        t = t.strip()
        if not t:
            continue
        words.append(Word(t, float(data["conf"][i]), data["left"][i], data["top"][i],
                          data["width"][i], data["height"][i],
                          (data["block_num"][i], data["par_num"][i], data["line_num"][i])))
    return words


def _lines(words):
    """Groups words into visual lines by vertical overlap.

    Tesseract's own line ids are unreliable in sparse mode (psm 11 puts nearly
    every word in its own block), so lines are rebuilt from geometry.
    """
    lines = []
    for w in sorted(words, key=lambda w: (w.y + w.h / 2, w.x)):
        cy = w.y + w.h / 2
        for ln in lines:
            ref = ln[0]
            if abs(cy - (ref.y + ref.h / 2)) < max(ref.h, w.h) * 0.6:
                ln.append(w)
                break
        else:
            lines.append([w])
    return [sorted(ln, key=lambda w: w.x) for ln in lines]


def _norm(s):
    return re.sub(r"[^A-Z0-9<]+", " ", s.upper()).strip()


def fuzzy_find(text, phrase, threshold=0.8):
    """Best similarity of `phrase` against any same-length window of `text`."""
    phrase = _norm(phrase)
    if phrase in text:
        return 1.0
    n = len(phrase)
    if len(text) < n:
        return SequenceMatcher(None, text, phrase).ratio()
    best = 0.0
    # A cheap prefilter keeps this fast: only score windows that share the
    # phrase's first or last letter at the expected position.
    for i in range(len(text) - n + 1):
        if text[i] != phrase[0] and text[i + n - 1] != phrase[-1]:
            continue
        r = SequenceMatcher(None, text[i:i + n], phrase).ratio()
        if r > best:
            best = r
            if best >= 0.99:
                break
    return best if best >= threshold else 0.0


# Characters OCR confuses with digits on these fonts.
_TO_DIGIT = str.maketrans({"O": "0", "o": "0", "D": "0", "Q": "0", "I": "1", "l": "1", "|": "1",
                           "i": "1", "L": "1", "Z": "2", "z": "2", "S": "5", "s": "5", "B": "8",
                           "G": "6", "b": "6", "T": "7", "g": "9", "q": "9", "A": "4"})


def digits_only(s):
    return re.sub(r"\D", "", s.translate(_TO_DIGIT))


def _fix_numeric_tokens(text):
    """Repairs letter/digit confusions inside tokens that are mostly digits.

    "BO000098685" -> "B0000098685". The first character is kept unless it is
    itself a digit look-alike, because several ID formats start with a letter
    (B/C card numbers, SP/TP/DP/MC permits).
    """
    def fix(m):
        tok = m.group(0)
        if sum(c.isdigit() for c in tok) * 2 < len(tok):
            return tok
        head = tok[0] if tok[0] not in "OIL" else tok[0].translate(_TO_DIGIT)
        if tok[:2] in ("SP", "TP", "DP", "MC"):
            return tok[:2] + tok[2:].translate(_TO_DIGIT)
        return head + tok[1:].translate(_TO_DIGIT)
    return re.sub(r"[A-Z0-9]{6,}", fix, text)


def _ocr_line(img, whitelist, psm=7):
    cfg = f"--psm {psm} -c tessedit_char_whitelist={whitelist}"
    return pytesseract.image_to_string(img, config=cfg).strip()


def _pad(img, px=12, value=255):
    return cv2.copyMakeBorder(img, px, px, px, px, cv2.BORDER_CONSTANT, value=value)


def _dark_text_mask(bgr):
    """Black print on a coloured security background.

    The guilloche patterns and tints on these cards are coloured, so at least
    one channel is bright; printed text is dark in all three. Thresholding the
    brightest channel keeps the text and drops most of the pattern, which is
    what defeats plain grayscale OCR on the CID number line.
    """
    mx = bgr.max(axis=2)
    t, _ = cv2.threshold(mx, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return np.where(mx <= t, 0, 255).astype(np.uint8)


def _binarizations(crop):
    """A few preprocessings of a field crop, cheapest-first."""
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    yield "otsu", cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    yield "dark", _dark_text_mask(crop)
    yield "gray", gray


def _scale_to_height(img, target):
    h = img.shape[0]
    if h == 0:
        return img
    k = target / h
    return cv2.resize(img, None, fx=k, fy=k, interpolation=cv2.INTER_CUBIC if k > 1 else cv2.INTER_AREA)


def _crop(img, x0, y0, x1, y1):
    H, W = img.shape[:2]
    x0, y0 = max(int(x0), 0), max(int(y0), 0)
    x1, y1 = min(int(x1), W), min(int(y1), H)
    if x1 - x0 < 4 or y1 - y0 < 4:
        return None
    return img[y0:y1, x0:x1]


# ---- Check digits ------------------------------------------------------------

_MRZ_VAL = {c: i for i, c in enumerate("0123456789")}
_MRZ_VAL.update({c: i + 10 for i, c in enumerate("ABCDEFGHIJKLMNOPQRSTUVWXYZ")})
_MRZ_VAL["<"] = 0


def mrz_check(data):
    return sum(_MRZ_VAL.get(c, 0) * (7, 3, 1)[i % 3] for i, c in enumerate(data)) % 10


def hkid_valid(s):
    m = re.fullmatch(r"([A-Z]{1,2})(\d{6})\(?([0-9A])\)?", s)
    if not m:
        return False
    letters, digits, check = m.groups()
    letters = letters.rjust(2, " ")
    vals = [36 if c == " " else ord(c) - 55 for c in letters] + [int(d) for d in digits]
    total = sum(v * w for v, w in zip(vals, range(9, 1, -1)))
    expected = (11 - total % 11) % 11
    return check == ("A" if expected == 10 else str(expected))


_VERHOEFF_D = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 2, 3, 4, 0, 6, 7, 8, 9, 5],
               [2, 3, 4, 0, 1, 7, 8, 9, 5, 6], [3, 4, 0, 1, 2, 8, 9, 5, 6, 7],
               [4, 0, 1, 2, 3, 9, 5, 6, 7, 8], [5, 9, 8, 7, 6, 0, 4, 3, 2, 1],
               [6, 5, 9, 8, 7, 1, 0, 4, 3, 2], [7, 6, 5, 9, 8, 2, 1, 0, 4, 3],
               [8, 7, 6, 5, 9, 3, 2, 1, 0, 4], [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]]
_VERHOEFF_P = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 5, 7, 6, 2, 8, 3, 0, 9, 4],
               [5, 8, 0, 3, 7, 9, 6, 1, 4, 2], [8, 9, 1, 6, 0, 4, 3, 5, 2, 7],
               [9, 4, 5, 3, 1, 2, 6, 8, 7, 0], [4, 2, 8, 6, 5, 7, 3, 9, 0, 1],
               [2, 7, 9, 3, 8, 0, 6, 4, 1, 5], [7, 0, 4, 6, 9, 1, 3, 2, 5, 8]]


def verhoeff_valid(num):
    c = 0
    for i, d in enumerate(reversed(num)):
        c = _VERHOEFF_D[c][_VERHOEFF_P[i % 8][int(d)]]
    return c == 0


# ---- Document types ----------------------------------------------------------

@dataclass
class DocType:
    key: str
    label: str
    # (phrase, weight): printed titles and field labels, matched fuzzily.
    phrases: list
    # Regexes over the whole-card text that are strong evidence on their own
    # (a number format only this document uses).
    patterns: list = field(default_factory=list)


DOC_TYPES = [
    DocType("bt_work_permit", "Bhutan work permit",
            [("WORK PERMIT", 3), ("JOB CATEGORY", 2), ("EMPLOYER", 1), ("DAY WORKER", 1),
             ("WORK SITE", 1), ("OCCUPATION", 1), ("VALID TILL", 0.5)]),
    DocType("bt_student_permit", "Bhutan student permit",
            [("STUDENT PERMIT", 4), ("INSTITUTION", 2)], [r"\bSP\s?\d{9}\b"]),
    DocType("bt_trader_permit", "Bhutan trader permit",
            [("TRADER PERMIT", 4), ("BUSINESS", 1.5)], [r"\bTP\s?\d{6}\b"]),
    DocType("bt_dependent_permit", "Bhutan dependent permit",
            [("DEPENDENT PERMIT", 4)], [r"\bDP\s?\d{8}\b"]),
    DocType("bt_immigration_card", "Bhutan immigration card",
            [("IMMIGRATION CARD", 4)], [r"\bMC\s?\d{6}\b"]),
    DocType("bt_cid_front", "Bhutan citizenship ID card (front)",
            [("CITIZENSHIP CARD", 3), ("CITIZENSHIP ID NO", 3), ("DATE OF BIRTH", 0.5), ("SEX", 0.3)]),
    DocType("bt_cid_back", "Bhutan citizenship ID card (back)",
            [("HOUSEHOLD NO", 3), ("DATE OF EXPIRY", 1), ("DATE OF ISSUE", 1)],
            [r"\b[B8]\s?0{3}\d{7}\b", r"\bC\s?0{3}\d{7}\b"]),
    DocType("bt_srp_front", "Bhutan special residence permit (front)",
            [("SPECIAL RESIDENCE", 3), ("SRP NO", 3)]),
    DocType("bt_srp_back", "Bhutan special residence permit (back)", []),
    DocType("bt_driving_license", "Bhutan driving licence",
            [("DRIVING LICENSE", 4), ("LICENSE NO", 2), ("BLOOD GROUP", 1), ("OFFENCES", 1)]),
    DocType("bt_voter_id", "Bhutan voter photo identity card",
            [("ELECTION COMMISSION", 3), ("VOTER PHOTO IDENTITY", 3), ("POLLING STATION", 1)]),
    DocType("bt_passport", "Bhutan passport",
            [("PASSPORT", 1), ("FOREIGN MINISTRY", 2), ("NAME OF BEARER", 1.5), ("BHUTANESE", 1)],
            [r"P<BTN", r"\bG\d{6}\b"]),
    DocType("foreign_passport", "Foreign passport", [("PASSPORT", 1)]),
    DocType("hk_id", "Hong Kong identity card",
            [("HONG KONG IDENTITY CARD", 4), ("FOR A PERSON OF THE AGE OF 18", 3)],
            [r"\b[A-Z]{1,2}\d{6}\s?\(\s?[0-9A]\s?\)"]),
    DocType("de_id", "German identity card",
            [("BUNDESREPUBLIK DEUTSCHLAND", 3), ("PERSONALAUSWEIS", 3), ("AUGENFARBE", 2), ("ANSCHRIFT", 1)]),
    DocType("in_aadhaar", "Indian Aadhaar card",
            [("UNIQUE IDENTIFICATION AUTHORITY", 4), ("GOVERNMENT OF INDIA", 3), ("AADHAAR", 2)],
            [r"\b\d{4}\s\d{4}\s\d{4}\b"]),
]
DOC_TYPES_BY_KEY = {t.key: t for t in DOC_TYPES}

_ACCEPT_THRESHOLD = 2.5
# A score this high means a printed title was read; no further OCR passes.
_CONFIDENT_SCORE = 4


def _is_bhutan_mrz(mrz, text):
    codes = {mrz.get("issuing_country") or "", mrz.get("nationality") or ""}
    if "BTN" in codes:
        return True
    # T is often read as I or 7 in the MRZ font; accept a one-letter slip
    # when the page itself says Bhutan.
    near = any(len(c) == 3 and sum(a != b for a, b in zip(c, "BTN")) == 1 for c in codes)
    return near and bool(fuzzy_find(text, "BHUTAN") or fuzzy_find(text, "BHUTANESE"))


def _card_tint(card):
    """Mean hue/saturation of the card's light background pixels."""
    hsv = cv2.cvtColor(card, cv2.COLOR_BGR2HSV)
    light = hsv[..., 2] > 150
    if light.sum() < 100:
        return None, None
    return float(np.median(hsv[..., 0][light])), float(np.median(hsv[..., 1][light]))


def classify(text, mrz, card):
    fixed = _fix_numeric_tokens(text)
    scores = {}
    for t in DOC_TYPES:
        s = sum(w * fuzzy_find(text, p) for p, w in t.phrases)
        s += sum(3 for pat in t.patterns if re.search(pat, fixed))
        scores[t.key] = s

    # Passports are identified by their MRZ, which is far more reliable than
    # OCR of the word "PASSPORT" in several scripts.
    if mrz and mrz["format"] == "TD3":
        key = "bt_passport" if _is_bhutan_mrz(mrz, text) else "foreign_passport"
        scores[key] += 6
    elif mrz and mrz["format"] == "TD1" and mrz["issuing_country"] == "D":
        scores["de_id"] += 6
    elif text.count("<<") >= 3 or "<<<<<" in text:
        # MRZ-looking lines that didn't parse: still a passport, not a card.
        key = "bt_passport" if fuzzy_find(text, "BHUTANESE") or "BTN" in text else "foreign_passport"
        scores[key] += 3

    # The SRP back and the new-style CID back share a layout (dates + QR or
    # barcode, no labelled number). The SRP card is printed on a green/teal
    # stock, the CID on white-lilac: hue ~80-100 with visible saturation.
    hue, sat = _card_tint(card)
    looks_back = fuzzy_find(text, "DATE OF EXPIRY") or fuzzy_find(text, "DATE OF ISSUE")
    if looks_back and hue is not None and 70 <= hue <= 100 and sat >= 25:
        scores["bt_srp_back"] = scores["bt_cid_back"] + 1

    best = max(scores, key=scores.get)
    if scores[best] < _ACCEPT_THRESHOLD:
        return None, scores
    return best, scores


# ---- MRZ ---------------------------------------------------------------------

_MRZ_ALPHA = str.maketrans({"0": "O", "1": "I", "5": "S", "8": "B", "2": "Z", "6": "G", "7": "T", "4": "A"})
_MRZ_NUM = str.maketrans({"O": "0", "D": "0", "Q": "0", "U": "0", "I": "1", "L": "1", "Z": "2", "S": "5",
                          "B": "8", "G": "6", "T": "7", "A": "4"})


def _mrz_clean(line):
    """Maps an OCR'd line onto the MRZ alphabet.

    Tesseract reads the '<' filler as lowercase c/e/k, «, or K/X in runs, so
    lowercase letters and runs of K/X/C next to '<' become filler.
    """
    l = re.sub(r"\s+", "", line)
    l = re.sub(r"[a-z«‹(\[{]", "<", l).upper()
    l = re.sub(r"[^A-Z0-9<]", "", l)
    # Filler misread as K/X/C/E: only the trailing run after the last name
    # part is rewritten, so names that start with those letters survive.
    l = re.sub(r"<[<KXCE]*$", lambda m: "<" * len(m.group(0)), l)
    return l


def _mrz_candidates(strings):
    out = []
    for raw in strings:
        l = _mrz_clean(raw)
        if len(l) >= 25 and (l.count("<") >= 2 or re.search(r"\d{6}", l)):
            out.append(l)
    return out


def _check(field_value, cd):
    return cd.isdigit() and mrz_check(field_value) == int(cd)


def _repair(value, cd, first_alpha=False):
    """Returns (value, ok): tries digit/letter swaps that make the check digit agree."""
    cd = cd.translate(_MRZ_NUM)
    if _check(value, cd):
        return value, True
    tries = [value.translate(_MRZ_NUM)]
    if first_alpha:
        tries += [value[:1].translate(_MRZ_ALPHA) + value[1:].translate(_MRZ_NUM),
                  value[:2].translate(_MRZ_ALPHA) + value[2:].translate(_MRZ_NUM)]
    for t in tries:
        if _check(t, cd):
            return t, True
    return value, False


def _parse_td3_line2(l2):
    """ICAO 9303 TD3 line 2 (44 chars). Tolerates a clipped or padded line.

    A line longer than 44 has junk around it: find the offset where the sex
    field and date shapes line up. A shorter one (a finger over the start) is
    aligned from the right, which keeps dates and the personal number (the
    CID on Bhutan passports) even though the document number is lost.
    """
    if len(l2) > 44:
        for off in range(len(l2) - 43):
            seg = l2[off:off + 44]
            if seg[20] in "MF<" and re.fullmatch(r"[0-9OIDQSBZ]{6}", seg[13:19]):
                l2 = seg
                break
        else:
            l2 = l2[:44]
    clipped = len(l2) < 44
    l2 = l2.rjust(44, "?")
    doc, doc_ok = _repair(l2[0:9], l2[9], first_alpha=True)
    dob, dob_ok = _repair(l2[13:19], l2[19])
    exp, exp_ok = _repair(l2[21:27], l2[27])
    dob_ok, exp_ok = dob_ok and dob.isdigit(), exp_ok and exp.isdigit()
    personal = l2[28:42]
    personal_num = personal.replace("<", "").translate(_MRZ_NUM)
    return {
        "format": "TD3",
        "document_number": None if (clipped and "?" in doc) else doc.replace("<", ""),
        "nationality": l2[10:13].translate(_MRZ_ALPHA).replace("<", ""),
        "date_of_birth": dob,
        "sex": l2[20],
        "expiry": exp,
        "personal_number": personal_num or None,
        "checks": {"document_number": doc_ok, "date_of_birth": dob_ok, "expiry": exp_ok,
                   "personal_number": _repair(personal, l2[42])[1] if personal.strip("<") else False},
    }


def _parse_td1_line1(l1):
    """ICAO 9303 TD1 (ID card, 3x30) line 1: type, issuer, document number."""
    l1 = (l1 + "<" * 30)[:30]
    doc, ok = _repair(l1[5:14], l1[14], first_alpha=True)
    return {"format": "TD1", "issuing_country": l1[2:5].replace("<", ""),
            "document_number": doc.replace("<", ""), "checks": {"document_number": ok}}


def _parse_mrz(lines):
    best = None
    for i, l in enumerate(lines):
        parsed = None
        if 26 <= len(l) <= 34 and l[0] in "IAC" and l[1:2] in ("D", "<", "I"):
            parsed = _parse_td1_line1(l)
        elif len(l) >= 36 and re.search(r"\d{6}.?[MF<]\d{6}", l.translate(_MRZ_NUM)[10:]):
            parsed = _parse_td3_line2(l)
            prev = lines[i - 1] if i > 0 else ""
            parsed["issuing_country"] = (prev[2:5].translate(_MRZ_ALPHA).replace("<", "")
                                         if prev.startswith("P") else parsed["nationality"])
            if prev.startswith("P") and "<<" in prev[5:]:
                surname, _, given = prev[5:].partition("<<")
                # Filler misread as letters leaves tokens like "SONAMK" or
                # "KKKK"; a name is only returned when the line parses cleanly.
                given_toks = [t for t in given.split("<") if t]
                surname_toks = [t for t in surname.split("<") if t]
                toks = given_toks + surname_toks
                if given_toks and surname_toks and all(re.fullmatch(r"[A-Z]{2,}", t) and "KK" not in t for t in toks):
                    parsed["name"] = " ".join(toks).title()
        if parsed:
            core = {k2: v for k2, v in parsed["checks"].items() if k2 != "personal_number"}
            parsed["checksum_valid"] = all(core.values())
            score = sum(parsed["checks"].values())
            if best is None or score > sum(best["checks"].values()):
                best = parsed
    return best


def read_mrz(card, words):
    """Parses the MRZ from the whole-card words, re-reading the band if needed."""
    cand = [w for w in words if "<<" in w.text or (len(w.text) >= 20 and re.fullmatch(r"[A-Za-z0-9<]+", w.text))]
    if not cand:
        return None
    # The whole-card pass often reads the MRZ better than a re-OCR of the band
    # does, so its lines are parsed first.
    # MRZ lines are often split into several words; keep every word on a line
    # that contains at least one MRZ-looking word.
    ids = {id(w) for w in cand}
    mrz_lines = [ln for ln in _lines(words) if any(id(w) in ids for w in ln)]
    best = _parse_mrz(_mrz_candidates(["".join(w.text for w in ln) for ln in mrz_lines]))
    if best and best["checksum_valid"]:
        return best

    y0 = min(w.y for w in cand)
    y1 = max(w.y + w.h for w in cand)
    h = max(w.h for w in cand)
    band = _crop(card, 0, y0 - 1.5 * h, card.shape[1], y1 + 1.5 * h)
    if band is None:
        return best
    for _, b in _binarizations(band):
        b = _pad(_scale_to_height(b, max(b.shape[0], 90)))
        raw = pytesseract.image_to_string(
            b, config="--psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789<")
        parsed = _parse_mrz(_mrz_candidates(raw.splitlines()))
        if parsed and (best is None or sum(parsed["checks"].values()) > sum(best["checks"].values())):
            best = parsed
        if best and best["checksum_valid"]:
            break
    return best


# ---- Field extraction --------------------------------------------------------

def _find_label(lines, phrase, threshold=0.75):
    """Returns (line words, index of the label's last word) for the best match."""
    target = _norm(phrase)
    best, best_r = None, threshold
    for ln in lines:
        toks = [_norm(w.text) for w in ln]
        for i in range(len(ln)):
            acc = ""
            for j in range(i, min(i + 5, len(ln))):
                acc = (acc + " " + toks[j]).strip()
                r = SequenceMatcher(None, acc, target).ratio()
                if r > best_r:
                    best, best_r = (ln, j), r
    return best


def _red_print_mask(bgr):
    """Pixels printed in red/pink ink.

    Fixed HSV ranges fail on these photos: the red print comes out anywhere
    from crimson to faded pink, while skin and wooden tables also read as
    "red". In Lab, the print is high on a* (red-green) without the matching
    b* (yellow) that wood and skin carry.
    """
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.int16)
    a, b = lab[..., 1] - 128, lab[..., 2] - 128
    return ((a > 10) & (a > b + 4)).astype(np.uint8)


def _permit_window(card, face, k):
    """The block under the photo holding the permit number and its dates.

    Measured on the samples relative to the detected face (x from -0.84 to
    +2.43 face widths, y from +1.34 to +3.21 face heights), with margin.
    """
    fx, fy, fw, fh = (float(v) * k for v in face[:4])
    return _crop(card, fx - 1.3 * fw, fy + 1.1 * fh, fx + 3.2 * fw, fy + 3.5 * fh)


def _red_line_boxes(win):
    """Bounding boxes (x0, y0, x1, y1) of the lines of red print, top down.

    On every permit-family card the first red line under the photo is the
    number, then the issue date, then the validity date. Blobs touching the
    window's top edge are the bottom of the photo (a red shirt, skin) and are
    dropped.
    """
    m = _red_print_mask(win)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(m)
    keep = np.zeros_like(m)
    H, W = m.shape
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if y <= 1 or area < 4 or h > H * 0.25:
            continue
        keep[labels == i] = 1
    on = keep.sum(axis=1) > max(2, W * 0.01)
    bands, start = [], None
    for y, v in enumerate(on):
        if v and start is None:
            start = y
        elif not v and start is not None:
            bands.append((start, y))
            start = None
    if start is not None:
        bands.append((start, H))
    boxes = []
    for y0, y1 in bands:
        if y1 - y0 < H * 0.04:
            continue
        cols = np.where(keep[y0:y1].sum(axis=0) > 0)[0]
        if len(cols) and cols[-1] - cols[0] > (y1 - y0) * 3:
            boxes.append((cols[0], y0, cols[-1] + 1, y1))
    return boxes


def _line_variants(line):
    """Several renderings of one line of red print, most reliable first.

    The green channel gives the best contrast: red ink absorbs green while
    the pale blue-grey card reflects it. Several heights are tried because
    Tesseract's digit errors on small, blurry print (5/8, 1/7, 0/9) change
    with scale, so they rarely repeat across variants while the true
    reading does.
    """
    green = line[..., 1]
    gray = cv2.cvtColor(line, cv2.COLOR_BGR2GRAY)
    for ch in (green, gray):
        for height in (64, 48, 88):
            im = _scale_to_height(ch, height)
            yield cv2.threshold(im, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
            yield im


def _clean_permit_token(raw, prefix):
    t = re.sub(r"[^A-Za-z0-9]", "", raw).upper()
    if prefix:
        return prefix + digits_only(t[2:]) if t[:2].translate(_MRZ_ALPHA) == prefix else t
    return digits_only(t)


_MONTHS = {m: i + 1 for i, m in enumerate(
    ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"])}
_ANY_DATE_RE = re.compile(
    r"\b(\d{1,2})\s?[/.\- ]\s?(\d{1,2}|[A-Z]{3})[A-Z]*\s?[/.\- ]\s?((?:19|20)\d{2})\b"
    r"|\b((?:19|20)\d{2})[- ](\d{2})[- ](\d{2})\b")


def _safe_date(y, m, d):
    try:
        v = date(int(y), int(m), int(d))
    except ValueError:
        return None
    return v if 1900 <= v.year <= 2060 else None


def dates_in_text(text):
    """Every date printed on the card, in any of the formats these cards use:
    04/03/2016, 26-04-2026, 20 03 2023 (Bhutan passport), 09 JUL 2025,
    2024-09-11 (older work permit)."""
    found = []
    for m in _ANY_DATE_RE.finditer(text.upper()):
        if m.group(4):
            v = _safe_date(m.group(4), m.group(5), m.group(6))
        else:
            mon = m.group(2)
            mon = _MONTHS.get(mon[:3]) if mon.isalpha() else int(mon)
            v = _safe_date(m.group(3), mon, m.group(1)) if mon else None
        if v and v not in found:
            found.append(v)
    return found


def _add_years(d, years):
    try:
        return d.replace(year=d.year + years)
    except ValueError:  # 29 February
        return d.replace(year=d.year + years, day=28)


def validity_pair(dates, expiry=None, terms=(10, 5), counts=None):
    """(issue, expiry) among `dates` whose gap is a whole validity term.

    Every CID card, SRP card and passport in the samples expires exactly 5
    or 10 years after issue, minus one day (the driving licence: exactly 10
    years). With `expiry` already known (a passport's MRZ, checked by its
    check digit), only the issue date is searched for. When several pairs
    fit (the same digit misread in both dates keeps the gap intact), the one
    with the most readings behind it (`counts`) wins.
    """
    counts = counts or {}
    pairs = []
    for exp in ([expiry] if expiry else dates):
        for iss in dates:
            if iss >= exp:
                continue
            for years in terms:
                end = _add_years(iss, years)
                if exp in (end, end - timedelta(days=1)):
                    pairs.append((counts.get(iss, 1) + counts.get(exp, 1), iss, exp))
    if not pairs:
        return None
    _, iss, exp = max(pairs)
    return iss, exp


def _mrz_date(yymmdd, future):
    """MRZ dates carry two-digit years: expiries are in this century, and a
    birth year later than this year belongs to the last one."""
    if not (yymmdd and re.fullmatch(r"\d{6}", yymmdd)):
        return None
    yy, mm, dd = int(yymmdd[:2]), yymmdd[2:4], yymmdd[4:]
    century = 2000 if future or yy <= date.today().year % 100 else 1900
    return _safe_date(century + yy, mm, dd)


def _read_date_region(region, text_height):
    """OCR a small region holding one date, under several binarizations."""
    if region is None:
        return {}
    k = 40 / max(text_height, 1)
    region = cv2.resize(region, None, fx=k, fy=k, interpolation=cv2.INTER_CUBIC if k > 1 else cv2.INTER_AREA)
    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    votes = {}
    for img, extra in ((_dark_text_mask(region), ""), (gray, "-c thresholding_method=2"),
                       (cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1], "")):
        raw = pytesseract.image_to_string(_pad(img, 20), config=f"--psm 7 {extra}")
        for d in dates_in_text(raw):
            votes[d] = votes.get(d, 0) + 1
    return votes


def _labelled_dates(card, pass_lines, labels, below=False):
    """{label: votes} for dates printed after (or under) each label."""
    out = {}
    for label in labels:
        votes = {}
        for lines in pass_lines:
            hit = _find_label(lines, label, threshold=0.85)
            if not hit:
                continue
            ln, j = hit
            first, last = ln[0], ln[j]
            for w in ln:
                if w.x <= last.x and _norm(w.text)[:2] == _norm(label)[:2]:
                    first = w
                    break
            h = max(w.h for w in ln[: j + 1])
            if below:
                region = _crop(card, first.x - h, last.y + 0.8 * h, last.x + last.w + 6 * h, last.y + 2.6 * h)
            else:
                region = _crop(card, last.x + last.w, last.y - 0.6 * h, last.x + last.w + 12 * h, last.y + 1.6 * h)
            for d, n in _read_date_region(region, h).items():
                votes[d] = votes.get(d, 0) + n
            if votes:
                break
        out[label] = votes
    return out


def _card_dates(card, pass_lines, text, issue_label, expiry_label, below=False, terms=(10, 5)):
    """Issue and expiry dates of a card, and whether each was confirmed.

    Dates come from the labelled regions and from the whole-card text. A
    pair a whole validity term apart confirms both; otherwise the most-read
    date after each label is returned unconfirmed.
    """
    labelled = _labelled_dates(card, pass_lines, (issue_label, expiry_label), below)
    counts = {}
    for segment in text.split(" | "):  # one segment per OCR pass
        for d in dates_in_text(segment):
            counts[d] = counts.get(d, 0) + 1
    for votes in labelled.values():
        for d, n in votes.items():
            counts[d] = counts.get(d, 0) + n
    pair = validity_pair(list(counts), terms=terms, counts=counts)
    if not pair:
        # Classification stops at the first OCR pass that reads a title,
        # which often hasn't caught both dates: try the other renderings.
        for name, image, extra in _classification_passes(card):
            if name != "dark":  # the only rendering that has recovered dates here
                continue
            for d in dates_in_text(_fix_numeric_tokens(_norm(" ".join(
                    w.text for w in ocr_words(image, psm=11, extra=extra))))):
                counts[d] = counts.get(d, 0) + 1
            pair = validity_pair(list(counts), terms=terms, counts=counts)
            if pair and counts[pair[0]] >= 2 and counts[pair[1]] >= 2:
                break
    if pair:
        # Confirmed only if each date was read at least twice: a single
        # misread can still land a whole term away from the other date.
        return pair[0], pair[1], counts[pair[0]] >= 2 and counts[pair[1]] >= 2
    best = lambda v: max(v, key=v.get) if v else None
    issue, expiry = best(labelled[issue_label]), best(labelled[expiry_label])
    if not (issue and expiry):
        # No labels read: issue and expiry are the earliest and latest
        # plausible dates on the card (the back carries no birth date).
        recent = sorted(d for d in counts if 1990 <= d.year)
        if len(recent) >= 2:
            issue, expiry = recent[0], recent[-1]
    if issue and expiry and issue >= expiry:
        issue = expiry = None
    return issue, expiry, False


_DATE_RE = re.compile(r"(\d{2})-?(\d{2})-?(\d{4})")


def _parse_date(raw):
    """dd-mm-yyyy from an OCR'd red date line, or None."""
    m = _DATE_RE.search(re.sub(r"[^0-9-]", "", raw))
    if not m:
        return None
    try:
        d = date(int(m.group(3)), int(m.group(2)), int(m.group(1)))
    except ValueError:
        return None
    return d if 2010 <= d.year <= 2045 else None


def _work_permit_valid(num, issued=None):
    """Work permit numbers are <1-2 digit prefix><YYMMDD><4-digit serial>.

    The embedded date is when the permit was processed, always on or a few
    weeks before the printed issue date, e.g. 162603190151 issued 19-03-2026.
    Checking it catches most single-digit misreads that plain length doesn't
    (a dropped digit shifts the date out of range).
    """
    if not re.fullmatch(r"\d{11,12}", num) or num[0] == "0":
        return False
    yy, mm, dd = int(num[-10:-8]), int(num[-8:-6]), int(num[-6:-4])
    try:
        d = date(2000 + yy, mm, dd)
    except ValueError:
        return False
    if not 2018 <= d.year <= 2040:
        return False
    if issued is not None:
        # Seen on the samples: from 7 days after the issue date to ~5 months
        # before it.
        return timedelta(days=-30) <= issued - d <= timedelta(days=200)
    return True


def _vote(votes):
    """Most frequent reading, how many reads gave it, and its share of all.

    A reading and the same reading with one extra leading digit are left to
    compete rather than merged: Tesseract both drops and invents a leading
    "1" on these lines, and the printed width doesn't tell 11 digits from 12
    reliably, so that disagreement has to lower confidence, not be resolved.
    """
    if not votes:
        return None, 0, 0.0
    best = max(votes, key=votes.get)
    return best, votes[best], votes[best] / sum(votes.values())


def _read_permit_block(card, face, k, pattern, prefix, valid):
    """Reads number + issue/valid dates from the red block under the photo.

    Returns (number, confident, issue_date, expiry_date, dates_confident). The number line is
    read under every variant from _line_variants; a reading counts only if it
    matches the type's format whole and passes `valid`. Reading stops early
    once one value has a clear majority.
    """
    win = _permit_window(card, face, k)
    if win is None:
        return None, 0.0, None, None
    boxes = _red_line_boxes(win)[:4]
    crops = []
    for x0, y0, x1, y1 in boxes:
        h = y1 - y0
        # Faded trailing digits can drop out of the colour mask, so the crop
        # runs to at least the width of a full 12-digit number.
        x1 = max(x1 + h, x0 + 8.5 * h)
        crops.append(_crop(win, x0 - h, y0 - 0.35 * h, x1, y1 + 0.35 * h))

    wl = "0123456789" + prefix
    votes, number_at = {}, None
    # The number is normally the first red line, but a red shirt or the
    # emblem can add a band above it: take the first line that reads as one.
    for i, c in enumerate(crops[:2]):
        for v in _line_variants(c):
            t = _clean_permit_token(_ocr_line(_pad(v, 20), wl), prefix)
            if re.fullmatch(pattern, t) and valid(t, None):
                votes[t] = votes.get(t, 0) + 1
                ranked = sorted(votes.values(), reverse=True) + [0]
                if ranked[0] >= 3 and ranked[0] - ranked[1] >= 2:
                    break
        if votes:
            number_at = i
            break

    issued = expires = first_date = None
    dates_sure = False
    if number_at is not None:
        # The two red lines after the number are the issue and validity
        # dates. The crop keeps a small margin: the colour mask can start
        # inside the first digit, while a full margin reaches the grey colon
        # of "Issue Date:" (read as 8 or 2, which the date parser skips).
        found = []
        for x0, y0, x1, y1 in boxes[number_at + 1:number_at + 3]:
            h = y1 - y0
            c = _crop(win, x0 - 0.6 * h, y0 - 0.35 * h, x1 + h, y1 + 0.35 * h)
            votes_d = {}
            for v in (_line_variants(c) if c is not None else ()):
                d = _parse_date(_ocr_line(_pad(v), "0123456789-"))
                if d:
                    votes_d[d] = votes_d.get(d, 0) + 1
                    ranked = sorted(votes_d.values(), reverse=True) + [0]
                    if ranked[0] >= 4 and ranked[0] - ranked[1] >= 3:
                        break
            found.append(_vote(votes_d))
        first_date = found[0][0] if found else None
        if len(found) == 2 and found[0][0] and found[1][0]:
            (issued, n1, a1), (expires, n2, a2) = found
            # Permits in the samples run from one month to 30 months.
            plausible = issued < expires <= _add_years(issued, 3)
            if not plausible:
                issued = expires = None
            else:
                dates_sure = n1 >= 3 and n2 >= 3 and min(a1, a2) >= 0.6
        elif found and found[0][0]:
            issued = found[0][0]

    vetoed = False
    # The first date line checks the number even when the dates themselves
    # didn't pass their own checks.
    if first_date:
        consistent = {t: n for t, n in votes.items() if valid(t, first_date)}
        # A misread issue date shouldn't veto every reading, but a number
        # that disagrees with the printed date isn't trusted either.
        vetoed = not consistent
        votes = consistent or votes
    if not votes:
        vetoed = True  # single sparse read below: never marked confident
        # Faded print the colour mask misses: sparse OCR of the whole window.
        big = _scale_to_height(win, max(win.shape[0], 220))
        gray = cv2.cvtColor(big, cv2.COLOR_BGR2GRAY)
        for extra in ("-c thresholding_method=2", ""):
            for w in ocr_words(_pad(gray), psm=11, extra=extra):
                t = _clean_permit_token(w.text, prefix)
                if re.fullmatch(pattern, t) and valid(t, issued):
                    votes[t] = votes.get(t, 0) + 1
            if votes:
                break
    number, count, agreement = _vote(votes)
    # Measured on the samples: 3+ agreeing readings that make up 70%+ of all
    # valid readings were never wrong; below that, about 1 in 8 was.
    confident = count >= 3 and agreement >= 0.7 and not vetoed
    if number and first_date and not valid(number, first_date):
        dates_sure = False  # the number and the issue date disagree
    return number, confident, issued, expires, dates_sure


def _permit_fields(card, face, k, text, pattern, prefix="", valid=None, number_key="permit_number"):
    valid = valid or (lambda t, issued: True)
    f = {number_key: None}
    if face is not None:
        number, confident, issued, expires, dates_sure = _read_permit_block(card, face, k, pattern, prefix, valid)
        f[number_key] = number
        f["issue_date"] = issued.isoformat() if issued else None
        f["expiry_date"] = expires.isoformat() if expires else None
        f["_confident"] = ([number_key] if number and confident else []) + \
            (["issue_date", "expiry_date"] if dates_sure else [])
    if not f[number_key]:
        for tok in _fix_numeric_tokens(text).split():
            if re.fullmatch(pattern, tok) and valid(tok, None):
                f[number_key] = tok
                break
    return f


HOUSEHOLD_RE = r"(?:0[1-9]|1\d|20)\d{7}"
SRP_RE = r"3(?:0[1-9]|1\d|20)\d{8}"


def _labelled_number(card, pass_lines, label, length, pattern, span=11):
    """Number printed right after a label ("Household No.", "SRP No." ...).

    The label is looked up in each OCR pass separately (merging passes
    duplicates words and garbles the line), and the region to its right is
    re-read under several binarizations with a vote.
    """
    for lines in pass_lines:
        hit = _find_label(lines, label)
        if not hit:
            continue
        ln, j = hit
        lw = ln[j]
        h = max(w.h for w in ln[: j + 1])
        region = _crop(card, lw.x + lw.w, lw.y - 0.6 * h, lw.x + lw.w + span * h, lw.y + 1.6 * h)
        number, confident = _read_number_region(region, length, lambda t: bool(re.fullmatch(pattern, t)), 40, h)
        if number:
            return number, confident
    return None, False


def _name(pass_lines, mrz):
    """Holder's name: the words after "Name:" on the card, or the MRZ."""
    if mrz and mrz.get("name"):
        return mrz["name"]
    for lines in pass_lines:
        hit = None
        for ln in lines:
            for j, w in enumerate(ln):
                if re.match(r"(?i)(full\s*)?name:?", w.text) and len(_norm(w.text.split(":")[0])) <= 5:
                    hit = (ln, j)
                    break
            if hit:
                break
        if not hit:
            continue
        ln, j = hit
        # "Name:Tenzin" is often one word: keep what follows the colon.
        tail = ln[j].text.split(":", 1)[1] if ":" in ln[j].text else ""
        words = [tail] if tail else []
        words += [w.text for w in ln[j + 1:] if re.fullmatch(r"[A-Za-z.'-]+", w.text)]
        name = " ".join(words).strip(" .")
        if len(name) >= 3:
            return name
    return None


def _extract(doc_key, card, face, k, pass_words, text, mrz):
    """Per-type extraction. Returns fields plus "_confident": the fields
    whose value was confirmed by agreeing reads or a check digit."""
    pass_lines = [_lines(w) for w in pass_words]
    fixed = _fix_numeric_tokens(text)
    f, sure = {}, []

    def put(key, value, confident):
        f[key] = value
        if value and confident:
            sure.append(key)

    def from_text(pattern):
        m = re.search(r"\b" + pattern + r"\b", fixed)
        return m.group(0) if m else None

    if doc_key == "bt_work_permit":
        # Older design: "WPC19-12408280004" in dark print, no red block.
        old = re.search(r"WP([A-Z0-9])(\d{2})\s?(\d{11})", fixed)
        if old:
            put("permit_number", f"WP{old.group(1).translate(_MRZ_ALPHA)}{old.group(2)}-{old.group(3)}", False)
            # "Issue Date: 2024-09-11 ... Valid Till: 2026-06-10"
            found = sorted(d for d in dates_in_text(fixed) if d.year >= 2015)
            if len(found) >= 2:
                put("issue_date", found[0].isoformat(), False)
                put("expiry_date", found[-1].isoformat(), False)
        else:
            f.update(_permit_fields(card, face, k, text, r"\d{11,12}", valid=_work_permit_valid))
    elif doc_key == "bt_student_permit":
        f.update(_permit_fields(card, face, k, text, r"SP\d{9}", "SP"))
    elif doc_key == "bt_trader_permit":
        f.update(_permit_fields(card, face, k, text, r"TP\d{6}", "TP"))
    elif doc_key == "bt_dependent_permit":
        f.update(_permit_fields(card, face, k, text, r"DP\d{8}", "DP"))
    elif doc_key == "bt_immigration_card":
        f.update(_permit_fields(card, face, k, text, r"MC\d{6}", "MC", number_key="card_number"))
    elif doc_key == "bt_cid_front":
        number, confident = _cid_front_number(card, face, k)
        put("cid_number", number or from_text(CID_RE), confident)
    elif doc_key == "bt_cid_back":
        number, confident = _cid_card_number(card, pass_words, fixed)
        put("card_number", number, confident)
        issue, expiry, sure_dates = _card_dates(card, pass_lines, fixed, "Date of Issue", "Date of Expiry")
        put("issue_date", issue and issue.isoformat(), sure_dates)
        put("expiry_date", expiry and expiry.isoformat(), sure_dates)
        if fuzzy_find(text, "HOUSEHOLD NO", 0.7) or not number or number.startswith("B"):
            number, confident = _labelled_number(card, pass_lines, "Household No.", 9, HOUSEHOLD_RE, span=9)
            put("household_number", number or from_text(HOUSEHOLD_RE), confident)
    elif doc_key == "bt_srp_back":
        issue, expiry, sure_dates = _card_dates(card, pass_lines, fixed, "Date of Issue", "Date of Expiry")
        put("issue_date", issue and issue.isoformat(), sure_dates)
        put("expiry_date", expiry and expiry.isoformat(), sure_dates)
    elif doc_key == "bt_srp_front":
        number, confident = _labelled_number(card, pass_lines, "SRP No.", 11, SRP_RE)
        put("srp_number", number or from_text(SRP_RE), confident)
    elif doc_key == "bt_driving_license":
        m = re.search(r"\b([A-Z])\s?-\s?(\d{6})\b", text)
        put("license_number", f"{m.group(1)}-{m.group(2)}" if m else None, False)
        number, confident = _labelled_number(card, pass_lines, "CID:", 11, CID_RE)
        put("cid_number", number or from_text(CID_RE), confident)
        # The licence prints its dates under the "Issued:" / "Validity:" labels.
        issue, expiry, sure_dates = _card_dates(card, pass_lines, fixed, "Issued:", "Validity:",
                                                below=True, terms=(10, 5))
        put("issue_date", issue and issue.isoformat(), sure_dates)
        put("expiry_date", expiry and expiry.isoformat(), sure_dates)
    elif doc_key == "bt_voter_id":
        number, confident = _labelled_number(card, pass_lines, "Citizen ID No:", 11, CID_RE)
        put("cid_number", number or from_text(CID_RE), confident)
    elif doc_key in ("bt_passport", "foreign_passport"):
        if mrz:
            checks = mrz["checks"]
            number = mrz["document_number"]
            if doc_key == "bt_passport":
                number = _bhutan_passport_number(number, text)
            put("passport_number", number, checks.get("document_number"))
            nationality = "BTN" if doc_key == "bt_passport" else mrz.get("nationality")
            put("nationality", nationality, checks.get("document_number"))
            dob = _mrz_date(mrz.get("date_of_birth"), future=False)
            expiry = _mrz_date(mrz.get("expiry"), future=True)
            put("date_of_birth", dob and dob.isoformat(), checks.get("date_of_birth"))
            put("expiry_date", expiry and expiry.isoformat(), checks.get("expiry"))
            # The MRZ has no issue date. The page prints it, alongside the
            # birth and expiry dates; the one a whole validity term before
            # the (check-digit confirmed) expiry is the issue date.
            pair = validity_pair(dates_in_text(fixed), expiry=expiry) if expiry and checks.get("expiry") else None
            put("issue_date", pair[0].isoformat() if pair else None, bool(pair))
            if doc_key == "bt_passport":
                pn = mrz.get("personal_number") or ""
                put("cid_number", pn if re.fullmatch(CID_RE, pn) else None, checks.get("personal_number"))
        elif doc_key == "bt_passport":
            put("passport_number", from_text(r"G\d{6}"), False)
        else:
            put("passport_number", from_text(r"[A-Z]{1,2}\d{6,8}"), False)
    elif doc_key == "hk_id":
        m = re.search(r"\b([A-Z]{1,2}\d{6})\s?([0-9A])\b", fixed)
        v = f"{m.group(1)}({m.group(2)})" if m else None
        put("id_number", v, bool(v) and hkid_valid(v))
    elif doc_key == "de_id":
        if mrz and mrz["format"] == "TD1":
            put("id_number", mrz["document_number"], mrz["checks"]["document_number"])
        else:
            put("id_number", from_text(r"[CFGHJKLMNPRTVWXYZ][CFGHJKLMNPRTVWXYZ0-9]{8}"), False)
    elif doc_key == "in_aadhaar":
        m = re.search(r"\b\d{4}\s\d{4}\s\d{4}\b", text)
        v = re.sub(r"\s", "", m.group(0)) if m else None
        put("aadhaar_number", v, bool(v) and verhoeff_valid(v))
    if doc_key in ("bt_passport", "foreign_passport"):
        # The printed page labels its name field "Name of Bearer", which the
        # label search would pick up; only the MRZ name is trusted here.
        f["name"] = (mrz or {}).get("name")
    elif doc_key not in ("bt_cid_back", "bt_srp_back"):
        f["name"] = _name(pass_lines, mrz)
    f["_confident"] = sure + f.pop("_confident", [])
    return f


def _bhutan_passport_number(mrz_number, text):
    """Bhutan passport numbers are one letter + 6 digits (G123456).

    The MRZ check digit can't settle the letter: G (value 16) and 6 contribute
    identically mod 10, and OCR reads G as 6 constantly. So the letter comes
    from the format, cross-checked against the number printed on the page.
    """
    printed = re.findall(r"\b([A-Z])(\d{6})\b", _fix_numeric_tokens(text))
    if mrz_number and re.fullmatch(r"[A-Z0-9]\d{6}", mrz_number):
        for letter, digits in printed:
            if digits == mrz_number[1:]:
                return letter + digits
        return mrz_number[0].translate(_MRZ_ALPHA) + mrz_number[1:]
    if printed:
        return printed[0][0] + printed[0][1]
    return mrz_number


# CID numbers: 1 + dzongkhag (01-20) + 8 digits. The six digits after the
# leading 1 are the household's area code and match the start of the
# household number on the card's back (CID 11105001833, household 110500214).
CID_RE = r"1(?:0[1-9]|1\d|20)\d{8}"


def _digit_runs(words, length):
    """Numbers of exactly `length` digits in OCR'd words.

    Adjacent tokens are joined too, because Tesseract often splits a long
    number into two words.
    """
    toks = []
    for w in words:
        t = re.sub(r"[^A-Za-z0-9]", "", w.text)
        if t and sum(c.isdigit() for c in t) * 2 >= len(t):
            toks.append(digits_only(t))
        else:
            toks.append(None)
    out = []
    for i, t in enumerate(toks):
        if t is None:
            continue
        if len(t) == length:
            out.append(t)
        elif i + 1 < len(toks) and toks[i + 1] and len(t) + len(toks[i + 1]) == length:
            out.append(t + toks[i + 1])
    return out


def _read_number_region(region, length, valid, target_text_height, text_height):
    """OCR a small region under several binarizations; vote on valid numbers."""
    if region is None:
        return None, False
    k = target_text_height / max(text_height, 1)
    region = cv2.resize(region, None, fx=k, fy=k, interpolation=cv2.INTER_CUBIC if k > 1 else cv2.INTER_AREA)
    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    variants = [
        (_dark_text_mask(region), ""),
        (gray, "-c thresholding_method=2"),
        (cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1], ""),
        (gray, ""),
        (cv2.createCLAHE(3.0, (8, 8)).apply(gray), "-c thresholding_method=2"),
    ]
    votes = {}
    for img, extra in variants:
        for psm in (6, 11):
            for t in _digit_runs(ocr_words(_pad(img, 20), psm=psm, extra=extra), length):
                if valid(t):
                    votes[t] = votes.get(t, 0) + 1
        ranked = sorted(votes.values(), reverse=True) + [0]
        if ranked[0] >= 4 and ranked[0] - ranked[1] >= 3:
            break
    number, count, agreement = _vote(votes)
    return number, bool(number) and count >= 3 and agreement >= 0.7


def _cid_front_number(card, face, k):
    """The number after "Citizenship ID No." along the card's bottom edge.

    Located by position rather than by the label, because OCR confuses the
    label with the "Citizenship Card" heading. Relative to the face, the
    number starts ~0.8-1.3 face widths right of it and sits 1.35-1.75 face
    heights below its top (measured on the samples).
    """
    if face is None:
        return None, False
    fx, fy, fw, fh = (float(v) * k for v in face[:4])
    region = _crop(card, fx + 0.35 * fw, fy + 1.15 * fh, fx + 4.6 * fw, fy + 2.05 * fh)
    return _read_number_region(region, 11, lambda t: bool(re.fullmatch(CID_RE, t)), 40, 0.15 * fh)


def _cid_card_number(card, pass_words, fixed_text):
    """B/C + 10 digits, printed large without a label.

    B-numbers (older backs with a household number) and C-numbers (newer
    backs with a QR code) always have three leading zeros, which is what
    makes a whole-card token recognisable. Each such token is re-read from
    its own crop under several binarizations, and all readings vote.
    """
    votes = {}

    def clean(raw):
        t = re.sub(r"[^A-Z0-9]", "", raw.upper())
        if len(t) < 10:
            return None
        head = "C" if t[0] in "CG" else "B" if t[0] in "B8" else None
        v = head + digits_only(t[1:]) if head else None
        return v if v and re.fullmatch(r"[BC]000\d{7}", v) else None

    for words in pass_words:
        for w in words:
            v = clean(w.text)
            if not v:
                continue
            votes[v] = votes.get(v, 0) + 1
            h = w.h
            crop = _crop(card, w.x - 0.6 * h, w.y - 0.35 * h, w.x + w.w + 0.8 * h, w.y + 1.35 * h)
            if crop is None:
                continue
            for _, b in _binarizations(crop):
                for height in (48, 64):
                    v2 = clean(_ocr_line(_pad(_scale_to_height(b, height), 20), "BC0123456789"))
                    if v2:
                        votes[v2] = votes.get(v2, 0) + 1
        ranked = sorted(votes.values(), reverse=True) + [0]
        if ranked[0] >= 4 and ranked[0] - ranked[1] >= 3:
            break
    number, count, agreement = _vote(votes)
    if number:
        return number, count >= 3 and agreement >= 0.7
    m = re.search(r"[BC]000\d{7}", fixed_text.replace(" ", ""))
    return (m.group(0) if m else None), False


# ---- Validation --------------------------------------------------------------

_FORMATS = {
    "permit_number": {"bt_work_permit": r"\d{11,12}|WP[A-Z0-9]{3}-?\d{10,12}", "bt_student_permit": r"SP\d{9}",
                      "bt_trader_permit": r"TP\d{6}", "bt_dependent_permit": r"DP\d{8}"},
    "card_number": {"bt_immigration_card": r"MC\d{6}", "bt_cid_back": r"[BC]\d{10}"},
    "cid_number": CID_RE,
    "household_number": HOUSEHOLD_RE,
    "srp_number": SRP_RE,
    "license_number": r"[A-Z]-\d{6}",
    "issue_date": r"\d{4}-\d{2}-\d{2}",
    "expiry_date": r"\d{4}-\d{2}-\d{2}",
    "name": r".+",
}


def _validate(doc_key, fields, mrz):
    status = {}
    sure = set(fields.get("_confident", []))
    for k, v in fields.items():
        if k.startswith("_"):
            continue
        if v is None:
            status[k] = "not_found"
            continue
        if k == "name":
            status[k] = "unchecked"
            continue
        if v is None:
            status[k] = "not_found"
            continue
        if k in ("passport_number", "nationality", "date_of_birth", "expiry_date", "cid_number") and mrz:
            status[k] = "checksum_ok" if k in sure else "checksum_failed"
        elif k == "id_number" and doc_key == "hk_id":
            status[k] = "checksum_ok" if hkid_valid(v) else "checksum_failed"
        elif k == "id_number" and doc_key == "de_id" and mrz:
            status[k] = "checksum_ok" if mrz["checksum_valid"] else "checksum_failed"
        elif k == "aadhaar_number":
            status[k] = "checksum_ok" if verhoeff_valid(v) else "checksum_failed"
        else:
            fmt = _FORMATS.get(k)
            if isinstance(fmt, dict):
                fmt = fmt.get(doc_key)
            if not (fmt and re.fullmatch(fmt, v)):
                status[k] = "unchecked"
            else:
                status[k] = "format_ok" if k in sure else "needs_review"
    return status


# ---- Entry point -------------------------------------------------------------

def _classification_passes(card):
    """Whole-card OCR variants, in the order they're worth trying.

    No single binarization reads every card: Sauvola (Tesseract's
    thresholding_method=2) handles the grey, low-contrast permit photos that
    the default global threshold wipes out; the dark-text mask strips the CID
    card's guilloche; CLAHE lifts faded print. Passes stop as soon as a
    printed title is read, so a clear card costs one pass.
    """
    gray = cv2.cvtColor(card, cv2.COLOR_BGR2GRAY)
    yield "sauvola", gray, "-c thresholding_method=2"
    yield "default", gray, ""
    yield "dark", _dark_text_mask(card), ""
    yield "clahe", cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(gray), ""


def read_document(img, detect_face):
    """Identify the document and extract its ID number(s).

    Returns the upright image (so callers can crop the face from it), the
    face row in that image's coordinates, and the extraction result.
    """
    upright, rotation, face = orient(img, detect_face)
    card, k = _normalize_scale(upright, face)

    words, pass_words_list, texts, mrz = [], [], [], None
    doc_key, scores, passes = None, {}, []
    for name, image, extra in _classification_passes(card):
        pass_words = ocr_words(image, psm=11, extra=extra)
        passes.append(name)
        words += pass_words
        pass_words_list.append(pass_words)
        texts.append(_norm(" ".join(w.text for w in pass_words)))
        text = " | ".join(texts)
        if (mrz is None or not mrz["checksum_valid"]) and ("<<" in texts[-1] or fuzzy_find(texts[-1], "PASSPORT")):
            m = read_mrz(card, pass_words)
            if m and (mrz is None or sum(m["checks"].values()) > sum(mrz["checks"].values())):
                mrz = m
        doc_key, scores = classify(text, mrz, card)
        if doc_key and scores[doc_key] >= _CONFIDENT_SCORE:
            break

    doc_key, scores = classify(text, mrz, card)
    fields = _extract(doc_key, card, face, k, pass_words_list, text, mrz) if doc_key else {}
    result = {
        "document_type": doc_key,
        "document_label": DOC_TYPES_BY_KEY[doc_key].label if doc_key else "Not a supported document",
        "rotation_applied": rotation,
        "fields": {k2: v for k2, v in fields.items() if not k2.startswith("_")},
        "field_checks": _validate(doc_key, fields, mrz),
        "mrz": mrz,
        "type_scores": {k2: round(v, 2) for k2, v in sorted(scores.items(), key=lambda kv: -kv[1])[:3]},
        "ocr_passes": passes,
        "raw_text": texts[0] if texts else "",
    }
    return upright, face, result
