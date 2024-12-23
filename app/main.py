from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import faiss
import numpy as np
import json
import os
from sentence_transformers import SentenceTransformer
from fastapi.middleware.cors import CORSMiddleware



# Initialize FastAPI app
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Mengizinkan semua origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Path ke file
index_path = os.path.join("faiss_index", "data", "faiss_index.bin")
metadata_path = os.path.join("faiss_index", "data", "metadata.json")


# Load model SentenceTransformer
model = SentenceTransformer("all-MiniLM-L6-v2")

# Load FAISS index dan metadata
def load_faiss_index():
    # Load FAISS index
    index = faiss.read_index(index_path)
    
    # Load metadata
    with open(metadata_path, "r") as f:
        metadata = json.load(f)
        
    return index, metadata

index, metadata = load_faiss_index()

# Pydantic model untuk menerima input query
class Query(BaseModel):
    query: str

# Endpoint untuk pencarian berdasarkan query
@app.post("/search/")
async def search(query: Query):
    # Encode query text
    query_embedding = model.encode(query.query).astype("float32").reshape(1, -1)

    # Lakukan pencarian menggunakan FAISS
    D, I = index.search(query_embedding, k=5)  # Menampilkan 5 hasil teratas (k=5)

    # Ambil hasil metadata berdasarkan ID dokumen yang ditemukan
    results = []
    for idx in I[0]:
        if idx != -1:  # -1 berarti tidak ada hasil
            result = {
                "doc_id": metadata["doc_ids"][idx],
                "title": metadata["doc_texts"][idx].split(" ")[0],  # Menampilkan judul dokumen
                "content": metadata["doc_texts"][idx],
            }
            results.append(result)
    
    if not results:
        raise HTTPException(status_code=404, detail="No documents found")
    
    return {"results": results}

@app.get("/search/{doc_id}")
async def get_search_result(doc_id: int):
    # Cari dokumen berdasarkan doc_id dalam metadata
    if doc_id not in metadata["doc_ids"]:
        raise HTTPException(status_code=404, detail="Document not found")
    
    # Ambil index dari doc_id
    doc_idx = metadata["doc_ids"].index(doc_id)
    
    # Ambil informasi dokumen berdasarkan index
    document = {
        "doc_id": metadata["doc_ids"][doc_idx],
        "title": metadata["doc_texts"][doc_idx].split(" ")[0],  # Menampilkan judul dokumen
        "content": metadata["doc_texts"][doc_idx],
    }
    
    return document

# Menjalankan server menggunakan Uvicorn (bisa dijalankan dari terminal)
# uvicorn app.main:app --reload
