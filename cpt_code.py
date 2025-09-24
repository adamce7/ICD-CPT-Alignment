from fastapi import FastAPI
from pydantic import BaseModel, Extra
import pandas as pd
from rapidfuzz import process

# Load CPT codes CSV
df_cpt = pd.read_csv("data/cpt_code.csv")

app = FastAPI()

class PatientData(BaseModel):
    Surgeries: list[str] | None = None
    
    class Config:
        extra = 'allow'   # allow other fields like Diagnosis, Age, etc.

def find_cpt_from_csv(surgery: str, df=df_cpt):
    if not surgery:
        return {"error": "Surgery field is empty."}
    
    choices = df["Procedure Code Descriptions"].astype(str).tolist()
    match, score, idx = process.extractOne(surgery, choices)
    
    if score > 70:
        return {
            "CPT_Code": df.iloc[idx]["CPT Codes"],
        }
    else:
        return {"CPT_Code": None}

@app.post("/map_cpt")
def map_cpt(data: PatientData):
    if not data.Surgeries or len(data.Surgeries) == 0:
        return {"error": "No surgeries provided."}
    
    # Map each surgery in the list
    results = []
    for surgery in data.Surgeries:
        results.append(find_cpt_from_csv(surgery))
    
    return {"Surgeries": results}
