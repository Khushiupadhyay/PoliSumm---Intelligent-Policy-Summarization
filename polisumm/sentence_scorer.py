"""
Sentence Scoring and Ranking Module
Uses TF-IDF and transformer embeddings for scoring
"""

from typing import List, Dict
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import warnings
warnings.filterwarnings('ignore')


class SentenceScorer:
    """Score and rank sentences for extractive summarization"""
    
    def __init__(self, embedding_model: str = "sentence-transformers/all-mpnet-base-v2"):
        """
        Initialize the sentence scorer
        
        Args:
            embedding_model: Model name for semantic embeddings
        """
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        
        try:
            self.embedding_model = SentenceTransformer(embedding_model)
        except Exception as e:
            print(f"Warning: Could not load embedding model: {e}")
            self.embedding_model = None
    
    def compute_tfidf_scores(self, sentences: List[str]) -> np.ndarray:
        """
        Compute TF-IDF scores for sentences
        
        Args:
            sentences: List of sentence texts
            
        Returns:
            TF-IDF matrix
        """
        try:
            tfidf_matrix = self.tfidf_vectorizer.fit_transform(sentences)
            return tfidf_matrix
        except Exception as e:
            print(f"Error computing TF-IDF: {e}")
            return np.zeros((len(sentences), 1))
    
    def compute_tfidf_importance_scores(self, sentences: List[str]) -> List[float]:
        """
        Compute importance scores using TF-IDF
        
        Args:
            sentences: List of sentence texts
            
        Returns:
            List of importance scores
        """
        tfidf_matrix = self.compute_tfidf_scores(sentences)
        
        # Compute scores as sum of TF-IDF values per sentence
        scores = np.array(tfidf_matrix.sum(axis=1)).flatten()
        
        # Normalize
        if scores.max() > 0:
            scores = scores / scores.max()
        
        return scores.tolist()
    
    def compute_semantic_similarity_scores(self, sentences: List[str]) -> List[float]:
        """
        Compute importance scores using semantic similarity to centroid
        
        Args:
            sentences: List of sentence texts
            
        Returns:
            List of importance scores
        """
        if self.embedding_model is None:
            # Fallback to TF-IDF if embeddings not available
            return self.compute_tfidf_importance_scores(sentences)
        
        try:
            # Generate embeddings
            embeddings = self.embedding_model.encode(sentences)
            
            # Compute centroid
            centroid = np.mean(embeddings, axis=0)
            
            # Compute similarity to centroid
            similarities = []
            for emb in embeddings:
                similarity = cosine_similarity([emb], [centroid])[0][0]
                similarities.append(similarity)
            
            # Normalize
            similarities = np.array(similarities)
            if similarities.max() > 0:
                similarities = (similarities - similarities.min()) / (similarities.max() - similarities.min())
            
            return similarities.tolist()
        
        except Exception as e:
            print(f"Error computing semantic scores: {e}")
            return [0.0] * len(sentences)
    
    def compute_combined_scores(self, sentences: List[str], 
                               tfidf_weight: float = 0.3,
                               semantic_weight: float = 0.7) -> List[float]:
        """
        Compute combined scores using both TF-IDF and semantic similarity
        
        Args:
            sentences: List of sentence texts
            tfidf_weight: Weight for TF-IDF scores
            semantic_weight: Weight for semantic similarity scores
            
        Returns:
            List of combined scores
        """
        tfidf_scores = self.compute_tfidf_importance_scores(sentences)
        semantic_scores = self.compute_semantic_similarity_scores(sentences)
        
        # Normalize both to [0, 1]
        tfidf_scores = np.array(tfidf_scores)
        semantic_scores = np.array(semantic_scores)
        
        # Combine with weights
        combined = tfidf_weight * tfidf_scores + semantic_weight * semantic_scores
        
        # Normalize final scores
        if combined.max() > 0:
            combined = (combined - combined.min()) / (combined.max() - combined.min())
        
        return combined.tolist()
    
    def rank_sentences(self, processed_doc: Dict[str, any], 
                      top_n: int = 10,
                      scoring_method: str = 'combined') -> List[Dict[str, any]]:
        """
        Rank sentences by importance and select top N
        
        Args:
            processed_doc: Processed document from NLP pipeline
            top_n: Number of top sentences to select
            scoring_method: 'tfidf', 'semantic', or 'combined'
            
        Returns:
            List of top sentences with scores
        """
        sentences = [sent['text'] for sent in processed_doc['sentences']]
        
        # Compute scores
        if scoring_method == 'tfidf':
            scores = self.compute_tfidf_importance_scores(sentences)
        elif scoring_method == 'semantic':
            scores = self.compute_semantic_similarity_scores(sentences)
        else:  # combined
            scores = self.compute_combined_scores(sentences)
        
        # Create list of sentences with scores and indices
        scored_sentences = []
        for idx, (sentence, score) in enumerate(zip(processed_doc['sentences'], scores)):
            scored_sentences.append({
                'sentence': sentence,
                'text': sentence['text'],
                'idx': sentence['idx'],
                'score': score,
                'length': sentence['length']
            })
        
        # Sort by score
        scored_sentences.sort(key=lambda x: x['score'], reverse=True)
        
        # Select top N
        top_sentences = scored_sentences[:top_n]
        
        # Sort by original position to preserve structure
        top_sentences.sort(key=lambda x: x['idx'])
        
        return top_sentences
    
    def extract_key_sentences(self, processed_doc: Dict[str, any], 
                             summary_ratio: float = 0.2) -> List[str]:
        """
        Extract key sentences for extractive summarization
        
        Args:
            processed_doc: Processed document
            summary_ratio: Ratio of sentences to extract (0.0 to 1.0)
            
        Returns:
            List of extracted sentence texts in original order
        """
        total_sentences = len(processed_doc['sentences'])
        top_n = max(1, int(total_sentences * summary_ratio))
        
        ranked_sentences = self.rank_sentences(processed_doc, top_n=top_n)
        
        return [sent['text'] for sent in ranked_sentences]

