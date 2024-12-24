from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import faiss
import numpy as np
import json
import os
from sentence_transformers import SentenceTransformer
from fastapi.middleware.cors import CORSMiddleware
from .retriever import HybridRetriever
from .LTRModel import LTRModel

import pandas as pd


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

semantic_path = os.path.join("faiss_index", "data", "faiss_index.bin")
bm25_path = os.path.join("bm25_index", "b25_index.pkl")
ltr_path = os.path.join("ltr_model", "ltr_model.json")
metadata_path = os.path.join("faiss_index", "data", "metadata.json")

doc_path = os.path.join("faiss_index", "data", "corpus_dpr.csv")

doc = pd.read_csv(doc_path)
retriever = HybridRetriever(documents=doc, bm25_index_path=bm25_path, semantic_index_path=semantic_path, metadata_path=metadata_path)
ltr_model = LTRModel(retriever)
ltr_model.load_model(ltr_path)

def load_faiss_index(path:str, metadata_path):
        index = faiss.read_index(path)
        
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
        
        return index, metadata

index, metadata = load_faiss_index(semantic_path, metadata_path)
class Query(BaseModel):
    query: str

@app.post("/search/")
async def search(query: Query):
    results = ltr_model.search(query.query)
    
    if not results:
        raise HTTPException(status_code=404, detail="No documents found")
    
    results = [{key: int(value) if isinstance(value, np.int64) else value for key, value in result.items()} for result in results]
    
    return {"results": results}


@app.get("/search/{doc_id}")
async def get_search_result(doc_id: int):
    if doc_id not in metadata["doc_ids"]:
        raise HTTPException(status_code=404, detail="Document not found")
    
    doc_idx = metadata["doc_ids"].index(doc_id)
    
    document = {
        "doc_id": metadata["doc_ids"][doc_idx],
        "title": metadata["doc_texts"][doc_idx].split(" ")[0],  # Menampilkan judul dokumen
        "content": metadata["doc_texts"][doc_idx],
    }
    
    return document

