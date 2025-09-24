# fastapi_alignment.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import bert_embeddings.align_icd_cpt as align

app = FastAPI()

class AlignRequest(BaseModel):
    icd_code: Optional[str] = None
    cpt_code: Optional[str] = None
    threshold: Optional[float] = 0.6  # default threshold you can tune

@app.post("/check_alignment")
def check_alignment(req: AlignRequest):
    if not req.icd_code or not req.cpt_code:
        raise HTTPException(status_code=400, detail="icd_code and cpt_code are required")
    try:
        res = align.similarity_by_codes(req.icd_code, req.cpt_code)
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))

    aligned = res["Similarity"] >= req.threshold
    return {"aligned": aligned, **res}
