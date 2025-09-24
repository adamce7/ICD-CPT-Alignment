from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import re
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# ------------------------
# Load ICD & CPT reference data
# ------------------------
icd_df = pd.read_csv("data/icd10.csv")
cpt_df = pd.read_csv("data/cpt.csv")

icd_map = dict(zip(icd_df["Code"].astype(str), icd_df["Description"]))
cpt_map = dict(zip(cpt_df["Code"].astype(str), cpt_df["Description"]))

# ------------------------
# Load LLM (prototype)
# ------------------------
model_name = "m42-health/Llama3-Med42-8B"

tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",        # will try to put on GPU
    dtype=torch.float16 # use fp16 for speed on RTX cards
)

llm_pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer # force first GPU; fallback is CPU if unavailable
)

# ------------------------
# JSON Extraction Helper
# ------------------------
def extract_json(text: str):
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return {"Aligns": None, "Confidence": None, "Error": "No JSON found"}

    candidate = match.group(0)

    # Normalize booleans
    candidate = candidate.replace("True", "true").replace("False", "false")

    # Strip percentage signs
    candidate = re.sub(r"(\d+)\s*%", r"\1", candidate)

    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        return {"Aligns": None, "Confidence": None, "Error": "Invalid JSON"}

    return parsed

def normalize_icd(icd_code: str) -> str:
    """Normalize ICD-10 code: uppercase + remove dots."""
    return icd_code.strip().upper().replace(".", "")

# ------------------------
# FastAPI setup
# ------------------------
app = FastAPI(title="ICD ↔ CPT Alignment API")

class AlignmentRequest(BaseModel):
    icd_code: str
    cpt_code: str

@app.post("/check_alignment")
def check_alignment(req: AlignmentRequest):
    icd_code = normalize_icd(req.icd_code)
    cpt_code = req.cpt_code.strip()

    # Validate ICD
    if icd_code not in icd_map:
        raise HTTPException(status_code=400, detail=f"ICD-10 code '{icd_code}' not found")

    # Validate CPT
    if cpt_code not in cpt_map:
        raise HTTPException(status_code=400, detail=f"CPT code '{cpt_code}' not found")

    icd_desc = icd_map[icd_code]
    cpt_desc = cpt_map[cpt_code]

    # Prompt
    prompt = f"""
    You are a medical coding assistant.
    Given the following codes, determine if the CPT procedure is typically appropriate 
    for the ICD-10 diagnosis. Respond ONLY in JSON.

    ICD-10: {icd_code} - {icd_desc}
    CPT: {cpt_code} - {cpt_desc}

    Response format:
    {{
        "Aligns": true/false,
        "Confidence": number
    }}
    """

    raw_output = llm_pipe(
        prompt,
        max_new_tokens=100,
        temperature=0.2,
        return_full_text=False  # 🚀 only generated response
    )[0]["generated_text"]

    parsed = extract_json(raw_output)

    return {
        "icd_code": icd_code,
        "cpt_code": cpt_code,
        "result": parsed
    }
