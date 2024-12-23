import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import os
import json

# Load model Cross-Encoder
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Path ke file
csv_path = os.path.join("data", "corpus_dpr.csv")
index_path = os.path.join("data", "faiss_index.bin")
metadata_path = os.path.join("data", "metadata.json")

def build_faiss_index():
    # Load data dari CSV
    data = pd.read_csv(csv_path)
    if 'doc_id' not in data.columns or 'title' not in data.columns or 'text' not in data.columns:
        raise ValueError("CSV harus memiliki kolom: 'doc_id', 'title', dan 'text'.")

    # Generate embedding dokumen
    doc_ids = []
    doc_texts = []
    doc_embeddings = []

    for _, row in data.iterrows():
        text = f"{row['title']} {row['text']}"  # Gabungkan judul dan konten
        embedding = model.encode(text, convert_to_numpy=True)  # Generate embedding
        doc_ids.append(row['doc_id'])
        doc_texts.append(text)
        doc_embeddings.append(embedding)

    # Konversi embedding ke numpy array
    doc_embeddings = np.array(doc_embeddings).astype("float32")

    # Buat FAISS index
    embedding_dim = doc_embeddings.shape[1]
    index = faiss.IndexFlatL2(embedding_dim)  # L2 (Euclidean distance)
    index.add(doc_embeddings)  # Tambahkan embedding ke indeks

    # Simpan indeks FAISS dan metadata
    faiss.write_index(index, index_path)
    with open(metadata_path, "w") as f:
        json.dump({"doc_ids": doc_ids, "doc_texts": doc_texts}, f)

    print(f"Indeks FAISS dan metadata berhasil disimpan di folder 'data/'")

if __name__ == "__main__":
    build_faiss_index()
