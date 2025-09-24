from fastapi import FastAPI
from pydantic import BaseModel, Extra
import pandas as pd
from rapidfuzz import process

# Load ICD-10 codes CSV
df = pd.read_csv("data/icd_10codes.csv")

app = FastAPI()

class PatientData(BaseModel):
    Diagnosis: str | None = None
    class Config:
        extra = 'allow'   # allow all extra fields

def find_icd10_from_csv(diagnosis: str):
    if not diagnosis:
        return {"error": "Diagnosis field is empty."}
    
    choices = df["Description"].tolist()
    match, score, idx = process.extractOne(diagnosis, choices)
    
    if score > 70:
        return {"ICD10_Code": df.iloc[idx]["Code"]}
    else:
        return {"ICD10_Code": None}

@app.post("/map_icd10")
def map_icd10(data: PatientData):
    if not data.Diagnosis:
        return {"error": "Diagnosis field is empty."}
    return find_icd10_from_csv(data.Diagnosis)
