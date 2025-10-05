# main.py
import logging
import json
from typing import List, Optional

import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

# ------------- Config -------------
MODEL_NAME = "medalpaca/medalpaca-13b"  # adjust if you use a different HF repo
CSV_PATH = "data/surgery_kits.csv"
# -----------------------------------

# ------------- Load dataset -------------
data = pd.read_csv(CSV_PATH)
# Convert the string representation of lists into Python lists
data["Required_Kit"] = data["Required_Kit"].apply(eval)

# ------------- FastAPI app -------------
app = FastAPI(title="Surgery Kit Compliance API")

class Surgery(BaseModel):
    surgery: str
    cpt_code: str

class Kit(BaseModel):
    items: List[str]

class ComplianceResult(BaseModel):
    compliance_degree: float
    missing_items: List[str]
    extra_items: List[str]

# ------------- Model init (global qa_pipeline) -------------
qa_pipeline = None
try:
    logging.info("Attempting to load MedAlpaca model. This may take a while...")
    # transformers imports here so server can start even if model fails to load
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="auto",
    )

    qa_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=256,
        temperature=0.2
    )
    logging.info("Model loaded successfully.")
except Exception as e:
    logging.exception("Model load failed or skipped. Running in deterministic fallback-only mode.")
    qa_pipeline = None

# ------------- Helper: ask MedAlpaca -------------
def ask_medalpaca(surgery: str, cpt_code: str, kit_items: List[str]) -> List[str]:
    """
    Ask the LLM to infer a required kit list. Returns a list of instruments.
    If model isn't loaded, returns [].
    """
    global qa_pipeline
    if qa_pipeline is None:
        return []

    prompt = f"""
You are a surgical compliance assistant.
Given a surgery and CPT code, determine the required surgical kit items.
Return ONLY a JSON array/list of lowercase instrument names.

Surgery: {surgery}
CPT Code: {cpt_code}
Provided Kit: {kit_items}

Example output:
["scalpel", "retractor", "suction"]
"""
    try:
        resp = qa_pipeline(prompt)[0]["generated_text"]
    except Exception:
        return []

    # Extract the last JSON-like list in the model output
    try:
        start = resp.rfind("[")
        end = resp.rfind("]") + 1
        if start == -1 or end == -1:
            return []
        raw = resp[start:end]
        parsed = json.loads(raw)
        # Normalize to lowercase stripped strings
        parsed = [str(x).strip().lower() for x in parsed]
        return parsed
    except Exception:
        # fallback: return empty list on parse failure
        logging.exception("Failed to parse model output")
        return []

# ------------- Endpoint -------------
@app.post("/check_compliance", response_model=ComplianceResult)
def check_compliance(surgery: Surgery, kit: Kit):
    """
    1) Try match by CPT code
    2) Then try match by surgery name (case-insensitive)
    3) If not found, ask model (if available)
    4) Return compliance % and lists
    """
    # Find dataset row
    row = data[data["CPT_Code"] == surgery.cpt_code]
    if row.empty:
        row = data[data["Surgery"].str.lower() == surgery.surgery.lower()]

    if row.empty:
        required_kit = ask_medalpaca(surgery.surgery, surgery.cpt_code, kit.items)
    else:
        required_kit = [str(x).strip().lower() for x in row.iloc[0]["Required_Kit"]]

    # Normalize provided kit
    provided_set = set([str(x).strip().lower() for x in kit.items])
    required_set = set(required_kit)

    # If required_set is empty (model failed to infer and no CSV match), treat required as unknown:
    if not required_set:
        # In this case, return 0 compliance and mark all provided as extra
        return ComplianceResult(
            compliance_degree=0.0,
            missing_items=[],
            extra_items=list(provided_set)
        )

    missing_items = sorted(list(required_set - provided_set))
    extra_items = sorted(list(provided_set - required_set))

    compliance_degree = 100.0 * (len(required_set) - len(missing_items)) / len(required_set)

    return ComplianceResult(
        compliance_degree=round(compliance_degree, 2),
        missing_items=missing_items,
        extra_items=extra_items
    )
