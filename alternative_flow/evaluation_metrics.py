from sklearn.metrics import precision_score, recall_score
import numpy as np
from sentence_transformers import util
import regex as re

class Evaluator:
    def __init__(self, embed_model):
        self.embed_model = embed_model
        
    def semantic_similarity(self, pred, true):
        """Compare answer with ground truth using embeddings"""
        pred_emb = self.embed_model.encode(pred)
        true_emb = self.embed_model.encode(true)
        return util.cos_sim(pred_emb, true_emb).item()

    def citation_relevance(self, answer, context):
        """Check if citations match retrieved context"""
        cited_titles = re.findall(r'\[(.*?)\]', answer)
        context_titles = [doc.metadata['title'] for doc in context]
        return len(set(cited_titles) & set(context_titles)) / len(cited_titles) if cited_titles else 0

    def hallucination_score(self, answer, context):
        """Detect unsupported claims"""
        answer_emb = self.embed_model.encode(answer)
        context_emb = self.embed_model.encode(" ".join(context))
        return 1 - util.cos_sim(answer_emb, context_emb).item()