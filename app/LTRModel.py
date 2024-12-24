import xgboost as xgb
import numpy as np
import pandas as pd
from typing import List
from dataclasses import dataclass

@dataclass
class Document:
    title: str
    doc_id: str
    text: str
    score: float
    lexical_score: float
    semantic_score: float

class LTRModel:
    def __init__(self, retriever):
        self.retriever = retriever
        self.model = xgb.XGBRanker(
            objective='rank:ndcg',
            eval_metric=["ndcg@5"],
            learning_rate=0.1,
            n_estimators=100,
            max_depth=5,
            min_child_weight=1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )

    def save_model(self, path: str):
        self.model.save_model(path)
        print(f"Model saved to {path}")

    def load_model(self, path: str):
        self.model.load_model(path)
        print(f"Model loaded from {path}")

    def extract_features(self, query: str, doc: Document) -> List[float]:
        features = [
            doc.score,  
            len(query.split()),  
            len(doc.text.split()),  
            self._term_overlap(query, doc.text),
            self._normalized_term_overlap(query, doc.text),
            doc.lexical_score,
            doc.semantic_score,
        ]
        return features

    def _term_overlap(self, query: str, doc_text: str) -> float:
        query_terms = set(query.lower().split())
        doc_terms = set(doc_text.lower().split())
        return len(query_terms.intersection(doc_terms))

    def _normalized_term_overlap(self, query: str, doc_text: str) -> float:
        query_terms = set(query.lower().split())
        doc_terms = set(doc_text.lower().split())
        if not query_terms:
            return 0.0
        return len(query_terms.intersection(doc_terms)) / len(query_terms)

    def prepare_training_data(self, queries: pd.DataFrame, qrels: pd.DataFrame, k: int = 30):
        data = []
        features_list = []

        for _, row in queries.iterrows():
            results = self.retriever.search(row['text'], k=k)
            query_features = []

            for result in results:
                doc = Document(
                    title=result['title'],
                    doc_id=result['doc_id'],
                    text=result['text'],
                    score=result['score'],
                    lexical_score=result['lexical_score'],
                    semantic_score=result['semantic_score']
                )

                features = self.extract_features(result['title'] + " " + row['text'], doc)
                relevance = self.get_relevance(row['qid'], doc.doc_id, qrels)

                data.append({
                    'qid': row['qid'],
                    'doc_id': doc.doc_id,
                    'relevance': relevance
                })
                query_features.append(features)

            features_list.extend(query_features)

        df = pd.DataFrame(data)
        X = np.array(features_list)
        y = df['relevance'].values
        query_ids = df['qid'].values
        group_sizes = df.groupby('qid').size().tolist()

        return X, y, query_ids, group_sizes

    def get_relevance(self, qid: str, doc_id: str, qrels: pd.DataFrame) -> int:
        relevance = qrels[(qrels['qid'] == qid) & (qrels['doc_id'] == doc_id)]
        if not relevance.empty:
            return relevance['relevance'].values[0]
        return 0

    def train(self, train_queries: pd.DataFrame, dev_queries, qrels_train: pd.DataFrame, qrels_dev: pd.DataFrame, k: int = 30):

        # Prepare training and validation data
        X_train, y_train, _, group_sizes_train = self.prepare_training_data(train_queries, qrels_train, k)
        X_val, y_val, _, group_sizes_val = self.prepare_training_data(dev_queries, qrels_dev, k)

        # Train the model with validation
        self.model.fit(
            X_train, y_train,
            group=group_sizes_train,
            eval_set=[(X_val, y_val)],
            eval_group=[group_sizes_val],
            verbose=True
        )

    def search(self, query_text: str, k: int = 30):
        results = self.retriever.search(query_text, k=k)

        documents = [
            Document(
                title=r['title'],
                doc_id=r['doc_id'],
                text=r['title'] + ' ' + r['text'],
                score=r['score'],
                lexical_score=r['lexical_score'],
                semantic_score=r['semantic_score']
            ) for r in results
        ]

        features = []
        for doc in documents:
            feat = self.extract_features(query_text, doc)
            features.append(feat)

        X = np.array(features)
        scores = self.model.predict(X)

        ranked_results = []
        for doc, score in zip(documents, scores):
            ranked_results.append({
                'doc_id': doc.doc_id,
                'title': doc.title,
                'text': doc.text,
                'score': float(score),
                'original_score': doc.score
            })

        ranked_results.sort(key=lambda x: x['score'], reverse=True)

        return ranked_results[:k]
