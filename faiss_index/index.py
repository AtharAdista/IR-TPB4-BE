import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import os
import json

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

csv_path = os.path.join("data", "corpus_dpr.csv")
index_path = os.path.join("data", "faiss_index.bin")
metadata_path = os.path.join("data", "metadata.json")

def build_faiss_index():
    data = pd.read_csv(csv_path)
    if 'doc_id' not in data.columns or 'title' not in data.columns or 'text' not in data.columns:
        raise ValueError("CSV harus memiliki kolom: 'doc_id', 'title', dan 'text'.")

    doc_ids = []
    doc_texts = []
    doc_embeddings = []

    for _, row in data.iterrows():
        text = f"{row['title']} {row['text']}"  
        embedding = model.encode(text, convert_to_numpy=True)  
        doc_ids.append(row['doc_id'])
        doc_texts.append(text)
        doc_embeddings.append(embedding)

    doc_embeddings = np.array(doc_embeddings).astype("float32")

    embedding_dim = doc_embeddings.shape[1]
    index = faiss.IndexFlatL2(embedding_dim)  
    index.add(doc_embeddings)  

    faiss.write_index(index, index_path)
    with open(metadata_path, "w") as f:
        json.dump({"doc_ids": doc_ids, "doc_texts": doc_texts}, f)

    print(f"Indeks FAISS dan metadata berhasil disimpan di folder 'data/'")

if __name__ == "__main__":
    build_faiss_index()
