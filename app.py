from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
import pandas as pd
import math
from io import StringIO
from Bio.Seq import Seq
from Bio import SeqIO, AlignIO
import subprocess
import os
import uvicorn
import Levenshtein

app = FastAPI(title="EscaPRRS scoring API")

# configuration
REPO_DATA_DIR = os.environ.get("REPO_DATA_DIR", "/data")
FLEVO_CSV_PATH = os.path.join(REPO_DATA_DIR, "prrsv_scaled_flevo_scores.csv")
DB_FASTA_PATH = os.path.join(REPO_DATA_DIR, "database.fasta")
REF_AA_SEQ = os.environ.get(
    "REF_AA_SEQ",
    "MLEKCLTAGYCSQLLFFWCIVPFCFAALVNAASNSSSHLQLIYNLTICELNGTDWLNQKFDWAVETFVIFPVLTHIVSYGALTTSHFLDTAGLITVSTAGYYHGRYVLSSIYAVFALAALICFVIRLTKNCMSWRYSCTRYTNFLLDTKGNLYRWRSPVVIERRGKVEVGDHLIDLKRVVLDGSAATPITKISAEQWGRP",
)

# Optional API key protection: set API_KEY env var to require header "x-api-key: <API_KEY>"
API_KEY = os.environ.get("API_KEY")

if not os.path.exists(FLEVO_CSV_PATH):
    raise RuntimeError(f"Missing flevo CSV at {FLEVO_CSV_PATH}. Place files into container data directory.")

evescape_df = pd.read_csv(FLEVO_CSV_PATH)
evescape_df["mutations"] = evescape_df["wt"] + evescape_df["i"].astype(str) + evescape_df["mut"]
evescape_dict_pos = dict(zip(evescape_df["mutations"], evescape_df["evescape"] - evescape_df["evescape"].min()))
evescape_dict_sigmoid = dict(zip(evescape_df["mutations"], evescape_df["evescape"].apply(lambda x: 1 / (1 + math.exp(-x)))))

if not os.path.exists(DB_FASTA_PATH):
    raise RuntimeError(f"Missing database FASTA at {DB_FASTA_PATH}. Place files into container data directory.")
db_records = list(SeqIO.parse(DB_FASTA_PATH, "fasta"))
_db_seq_index = {r.id: str(r.seq) for r in db_records}

def get_mutations(wt_seq, query_seq):
    with open("/tmp/temp_alignment.fasta", "w") as f:
        f.write(">WT\n" + wt_seq + "\n")
        f.write(">QUERY\n" + query_seq + "\n")
    try:
        result = subprocess.run(
            ["mafft", "--globalpair", "--maxiterate", "1000", "--quiet", "/tmp/temp_alignment.fasta"],
            capture_output=True, text=True, check=True, timeout=15
        )
        alignment = AlignIO.read(StringIO(result.stdout), "fasta")
    except Exception:
        # fallback naive alignment (no gap detection)
        alignment = None

    if alignment is None:
        mutations = []
        for i, (a, b) in enumerate(zip(wt_seq, query_seq), start=1):
            if a != b:
                mutations.append(f"{a}{i}{b}")
        return mutations

    wt_aligned = str(alignment[0].seq)
    query_aligned = str(alignment[1].seq)
    mutations = []
    wt_pos = 0
    for wt_res, q_res in zip(wt_aligned, query_aligned):
        if wt_res != "-":
            wt_pos += 1
        if wt_res == "-" and q_res != "-":
            mutations.append(f"{wt_aligned[wt_pos-1]}{wt_pos}ins{q_res}")
        elif wt_res != "-" and q_res == "-":
            mutations.append(f"{wt_res}{wt_pos}del")
        elif wt_res != "-" and q_res != "-" and wt_res != q_res:
            mutations.append(f"{wt_res}{wt_pos}{q_res}")
    return mutations

class SequenceQuery(BaseModel):
    sequence: str
    require_closest: bool = True
    max_len: int = 2000

def check_api_key(request: Request):
    if API_KEY:
        hdr = request.headers.get("x-api-key")
        if hdr != API_KEY:
            raise HTTPException(status_code=403, detail="Invalid API key")

@app.post("/score")
async def score(query: SequenceQuery, request: Request):
    check_api_key(request)
    seq = "".join(query.sequence.split()).replace('-', '').upper()
    if len(seq) == 0 or len(seq) > query.max_len:
        raise HTTPException(status_code=400, detail="Invalid sequence length")
    wt = REF_AA_SEQ
    lev_dist = Levenshtein.distance(wt, seq)
    similarity = (1 - lev_dist / len(wt)) * 100.0

    try:
        mutations = get_mutations(wt, seq)
    except Exception:
        # safe fallback
        mutations = []
        for i, (a, b) in enumerate(zip(wt, seq), start=1):
            if a != b:
                mutations.append(f"{a}{i}{b}")

    filtered_mutations = [m for m in mutations if 'del' not in m and 'ins' not in m]

    score_pos = sum(evescape_dict_pos.get(m, 0) for m in filtered_mutations)
    score_sigmoid = sum(evescape_dict_sigmoid.get(m, 0) for m in filtered_mutations)

    if similarity < 30:
        uncertainty = 100.0
    else:
        uncertainty = (lev_dist / len(wt)) * 100.0

    closest_id = None
    closest_distance = None
    if query.require_closest:
        for rid, rseq in _db_seq_index.items():
            d = Levenshtein.distance(rseq, seq)
            if closest_distance is None or d < closest_distance:
                closest_distance = d
                closest_id = rid
        similarity_db = (1 - closest_distance / len(wt)) * 100 if closest_distance is not None else None
    else:
        similarity_db = None

    minimum = float(os.environ.get("ESC_MIN", 1.1572728))
    maximum = float(os.environ.get("ESC_MAX", 204.2925904))
    mean_val = float(os.environ.get("ESC_MEAN", 92.8164662))

    response = {
        "levenshtein_distance": int(lev_dist),
        "similarity_pct": round(float(similarity), 6),
        "mutations": filtered_mutations,
        "score_pos": float(score_pos),
        "score_sigmoid": float(score_sigmoid),
        "uncertainty_pct": float(round(uncertainty, 6)),
        "relative_to": {
            "min_diff": float(round(abs(score_pos - minimum), 6)),
            "mean_diff": float(round(abs(score_pos - mean_val), 6)),
            "max_diff": float(round(abs(maximum - score_pos), 6)),
            "position_vs_mean": ("above median" if score_pos > mean_val else "equal to median" if score_pos == mean_val else "below median")
        },
        "closest_match": {
            "id": closest_id,
            "levenshtein_distance": int(closest_distance) if closest_distance is not None else None,
            "similarity_pct": float(round(similarity_db, 6)) if similarity_db is not None else None
        }
    }
    return response

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))