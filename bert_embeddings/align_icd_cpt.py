# align_icd_cpt.py
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import os

EMB_DIR = "embeddings"

def load_embeddings(npz_path):
    data = np.load(npz_path, allow_pickle=True)
    return data["embeddings"], data["codes"], data["descriptions"]

# load both at module import
icd_emb, icd_codes, icd_descs = load_embeddings(os.path.join(EMB_DIR, "icd_embeddings.npz"))
cpt_emb, cpt_codes, cpt_descs = load_embeddings(os.path.join(EMB_DIR, "cpt_embeddings.npz"))

# create dictionaries for quick lookup
icd_codes = [str(c).strip() for c in icd_codes]
icd_index = {c: i for i, c in enumerate(icd_codes)}
cpt_codes = [str(c).strip() for c in cpt_codes]
cpt_index = {c: i for i, c in enumerate(cpt_codes)}


def cosine(u, v):
    # u, v are 1D numpy arrays (L2-normalized already)
    # If not normalized, cosine_similarity will still work.
    return float(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-9))

def similarity_by_codes(icd_code, cpt_code):
    if icd_code not in icd_index:
        raise ValueError("ICD code not found")
    if cpt_code not in cpt_index:
        raise ValueError("CPT code not found")
    i = icd_index[icd_code]
    j = cpt_index[cpt_code]
    sim = cosine(icd_emb[i], cpt_emb[j])
    return {
        "ICD_Code": icd_code,
        "ICD_Desc": icd_descs[i],
        "CPT_Code": cpt_code,
        "CPT_Desc": cpt_descs[j],
        "Similarity": sim
    }

def top_cpts_for_icd(icd_code, top_n=5):
    if icd_code not in icd_index:
        raise ValueError("ICD code not found")
    i = icd_index[icd_code]
    sims = (cpt_emb @ icd_emb[i])  # (N_cpt,)
    idxs = np.argsort(-sims)[:top_n]
    return [{
        "CPT_Code": cpt_codes[k],
        "CPT_Desc": cpt_descs[k],
        "Similarity": float(sims[k])
    } for k in idxs]

def top_icds_for_cpt(cpt_code, top_n=5):
    if cpt_code not in cpt_index:
        raise ValueError("CPT code not found")
    j = cpt_index[cpt_code]
    sims = (icd_emb @ cpt_emb[j])
    idxs = np.argsort(-sims)[:top_n]
    return [{
        "ICD_Code": icd_codes[k],
        "ICD_Desc": icd_descs[k],
        "Similarity": float(sims[k])
    } for k in idxs]

# # Example quick test:
# if __name__ == "__main__":
#     # sample: E11.9 and CPT 66984 (you provided)
#     icd_sample = "E11.9"
#     cpt_sample = "66984"
#     try:
#         r = similarity_by_codes(icd_sample, cpt_sample)
#         print(r)
#     except Exception as e:
#         print("Lookup error:", e)
