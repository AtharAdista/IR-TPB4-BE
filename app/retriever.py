import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from rank_bm25 import BM25Okapi
import numpy as np
from typing import List, Dict, Tuple
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from collections import defaultdict
import faiss
import pandas as pd
from nltk.stem.snowball import SnowballStemmer
import pickle
import os
import json
from sentence_transformers import SentenceTransformer

# Download NLTK data
nltk.download('punkt_tab')
nltk.download('stopwords')


class SemanticEncoder(nn.Module):
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", embedding_dim: int = 384):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.projection = nn.Linear(self.bert.config.hidden_size, embedding_dim)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0]
        return self.projection(pooled_output)


class PassageDataset(Dataset):
    def __init__(self, passages: List[str], tokenizer, max_length: int = 512):
        self.passages = passages
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.passages)
    
    def __getitem__(self, idx):
        passage = self.passages[idx]
        encoding = self.tokenizer(
            passage,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'passage': passage
        }


class HybridRetriever:
    def __init__(self, 
                 documents: pd.DataFrame,
                 model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 embedding_dim: int = 384,
                 batch_size: int = 32,
                 device: str = "cuda" if torch.cuda.is_available() else "cpu",
                 bm25_index_path: str = None,
                 semantic_index_path: str = None,
                 metadata_path:str=None):
        
        self.documents = documents
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self._load_bm25_index(bm25_index_path)
        
        self.index, self.metadata = self._load_faiss_index(semantic_index_path, metadata_path)
        self.stemmer = SnowballStemmer("english")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.semantic_encoder = SemanticEncoder(model_name, embedding_dim)
    
    def _load_bm25_index(self, path: str):
        """Load the BM25 index from a file."""
        with open(path, 'rb') as f:
            self.bm25 = pickle.load(f)

    def _load_faiss_index(self, path:str, metadata_path):
        index = faiss.read_index(path)
        
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
        
        return index, metadata
    

    
    def _lexical_search(self, query: str, k: int = 100) -> List[Tuple[int, float]]:
        """Perform BM25 search"""
        tokenized_query = [self.stemmer.stem(w.lower())  for w in word_tokenize(query)]
        scores = self.bm25.get_scores(tokenized_query)
        top_n = np.argsort(scores)[-k:][::-1]
        return [(idx, scores[idx]) for idx in top_n]
    
    def _semantic_search(self, query: str, k: int = 100) -> List[Dict]:
        """Perform semantic search using FAISS without manipulating the scores."""
        query_embedding = self.model.encode(query).astype("float32").reshape(1, -1)

        D, I = self.index.search(query_embedding, k)  # Getting the k top results

        results = []
        for i, idx in enumerate(I[0]):
            if idx != -1:  # -1 means no result
                result = (idx, float(D[0][i]))#{
                #     "doc_id": self.metadata["doc_ids"][idx],
                #     "title": self.metadata["doc_texts"][idx].split(" ")[0],  # Display the first word as title
                #     "content": self.metadata["doc_texts"][idx], # Directly use the FAISS distance as score
                #     "score": float(D[0][i])  
                # }
                results.append(result)
    
        return results
    
    def search(self, query: str, k: int = 100, alpha: float = 0.5) -> List[Dict]:
        lexical_results = self._lexical_search(query, k)
        semantic_results = self._semantic_search(query, k)

        print("Raw Semantic Results:", semantic_results)
        if isinstance(semantic_results, list):
            print("First entry in Semantic Results:", semantic_results[0])

        combined_scores = defaultdict(float)
        lexical_scores = defaultdict(float)
        semantic_scores = defaultdict(float)

        print("Lexical Results:", lexical_results)

        max_lexical = max((score for _, score in lexical_results), default=1e-8)  # Prevent zero division
        for idx, score in lexical_results:
            if score != 0:
                combined_scores[idx] += alpha * (score / max_lexical)
            else:
                combined_scores[idx] += 1e-8
            lexical_scores[idx] = score

                
        max_semantic = max((score for _, score in semantic_results), default=1e-8)  # Prevent zero division
        min_semantic = min((score for _, score in semantic_results), default=0)  # For proper scaling

        for idx, score in semantic_results:
            normalized_score = (max_semantic - score) / (max_semantic - min_semantic + 1e-8)
            combined_scores[idx] += (1 - alpha) * normalized_score
            semantic_scores[idx] = normalized_score

        results = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
        print("Combined Results:", results)

        return [
            {
                'text': self.documents['text'][idx],
                'score': score,
                'title': self.documents['title'][idx],
                'doc_id': self.documents['doc_id'][idx],
                'lexical_score': lexical_scores[idx],
                'semantic_score': semantic_scores[idx]
            }
            for idx, score in results
        ]



