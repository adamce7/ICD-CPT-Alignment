#embed_icd_cpt.py
import os
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

MODEL_NAME = "emilyalsentzer/Bio_ClinicalBERT"  # clinical BERT
ICD_CSV = "data/icd_10codes.csv"   # columns: 'Code', 'Description'
CPT_CSV = "data/CPT_Codes_List.csv"      # columns: 'Codes', 'Description'
OUT_DIR = "embeddings"
BATCH_SIZE = 32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME)
    model.to(DEVICE)
    model.eval()
    return tokenizer, model

def mean_pooling(last_hidden_state, attention_mask):
    # last_hidden_state: (batch, seq_len, dim)
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, dim=1)
    sum_mask = input_mask_expanded.sum(dim=1).clamp(min=1e-9)
    return sum_embeddings / sum_mask

def embed_texts(texts, tokenizer, model, batch_size=BATCH_SIZE):
    all_embeds = []
    for i in tqdm(range(0, len(texts), batch_size)):
        batch_texts = texts[i:i+batch_size]
        enc = tokenizer(batch_texts,
                        padding=True,
                        truncation=True,
                        max_length=256,
                        return_tensors="pt")
        input_ids = enc["input_ids"].to(DEVICE)
        attention_mask = enc["attention_mask"].to(DEVICE)
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            # use last_hidden_state and mean-pool
            pooled = mean_pooling(outputs.last_hidden_state, attention_mask)
            # normalize
            pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
            all_embeds.append(pooled.cpu().numpy())
    all_embeds = np.vstack(all_embeds)
    return all_embeds  # shape: (n_texts, hidden_size)

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    tokenizer, model = load_model()

    # Load ICD
    df_icd = pd.read_csv(ICD_CSV, dtype=str).fillna("")
    icd_texts = df_icd["Description"].astype(str).tolist()
    icd_codes = df_icd["Code"].astype(str).tolist()
    # icd_codes = df_icd["Code"].str.replace('.', '', regex=False).astype(str).tolist()
    print(f"Embedding {len(icd_texts)} ICD descriptions...")
    icd_emb = embed_texts(icd_texts, tokenizer, model)
    np.savez_compressed(os.path.join(OUT_DIR, "icd_embeddings.npz"),
                        embeddings=icd_emb, codes=icd_codes, descriptions=icd_texts)
    print("Saved ICD embeddings.")

    # Load CPT
    df_cpt = pd.read_csv(CPT_CSV, dtype=str).fillna("")
    cpt_codes = df_cpt["Code"].astype(str).str.strip().tolist()
    # cpt_codes = df_cpt["Code"].str.replace('.', '', regex=False).astype(str).tolist()
    cpt_texts = df_cpt["Description"].astype(str).tolist()
    
    print(f"Embedding {len(cpt_texts)} CPT descriptions...")
    cpt_emb = embed_texts(cpt_texts, tokenizer, model)
    np.savez_compressed(os.path.join(OUT_DIR, "cpt_embeddings.npz"),
                        embeddings=cpt_emb, codes= cpt_codes, descriptions=cpt_texts)
    print("Saved CPT embeddings.")

if __name__ == "__main__":
    main()
