"""
Evaluation Module
Compute ROUGE, BERTScore, SARI, and readability metrics
"""

from typing import Dict, List, Optional
import numpy as np
from rouge_score import rouge_scorer
try:
    from bert_score import score as bert_score
except ImportError:
    bert_score = None
try:
    import textstat
except ImportError:
    textstat = None
import re


class SummaryEvaluator:
    """Evaluate summary quality using multiple metrics"""
    
    def __init__(self):
        """Initialize evaluator"""
        self.rouge_scorer = rouge_scorer.RougeScorer(
            ['rouge1', 'rouge2', 'rougeL'],
            use_stemmer=True
        )
    
    def evaluate(self, original: str, summary: str, 
                reference: Optional[str] = None) -> Dict[str, float]:
        """
        Comprehensive evaluation of summary
        
        Args:
            original: Original document text
            summary: Generated summary
            reference: Reference summary (optional)
            
        Returns:
            Dictionary with evaluation scores
        """
        scores = {}
        
        # ROUGE scores (if reference provided)
        if reference:
            rouge_scores = self._compute_rouge(reference, summary)
            scores.update(rouge_scores)
        
        # Semantic similarity using BERTScore
        bert_scores = self._compute_bertscore(original, summary)
        scores.update(bert_scores)
        
        # SARI (if reference provided)
        if reference:
            sari_score = self._compute_sari(original, reference, summary)
            scores['sari'] = sari_score
        
        # Readability scores
        readability = self._compute_readability(original, summary)
        scores.update(readability)
        
        # Word overlap
        overlap_score = self._compute_word_overlap(original, summary)
        scores.update(overlap_score)
        
        # Preservation score
        preservation = self._compute_preservation_score(original, summary)
        scores['preservation_score'] = preservation
        
        return scores
    
    def _compute_rouge(self, reference: str, summary: str) -> Dict[str, float]:
        """Compute ROUGE scores"""
        scores = self.rouge_scorer.score(reference, summary)
        
        return {
            'rouge1': scores['rouge1'].fmeasure,
            'rouge2': scores['rouge2'].fmeasure,
            'rougeL': scores['rougeL'].fmeasure,
            'rouge1_precision': scores['rouge1'].precision,
            'rouge1_recall': scores['rouge1'].recall,
        }
    
    def _compute_bertscore(self, original: str, summary: str) -> Dict[str, float]:
        """Compute BERTScore for semantic similarity"""
        if bert_score is None:
            return {
                'bertscore_precision': 0.0,
                'bertscore_recall': 0.0,
                'bertscore_f1': 0.0
            }
        try:
            P, R, F1 = bert_score([summary], [original], lang='en', verbose=False)
            return {
                'bertscore_precision': float(P[0]),
                'bertscore_recall': float(R[0]),
                'bertscore_f1': float(F1[0])
            }
        except Exception as e:
            print(f"Error computing BERTScore: {e}")
            return {
                'bertscore_precision': 0.0,
                'bertscore_recall': 0.0,
                'bertscore_f1': 0.0
            }
    
    def _compute_sari(self, original: str, reference: str, summary: str) -> float:
        """
        Compute SARI score (simplification metric)
        Simplified implementation
        """
        try:
            # Split into sentences
            original_sents = self._split_sentences(original)
            reference_sents = self._split_sentences(reference)
            summary_sents = self._split_sentences(summary)
            
            # Tokenize
            original_tokens = [self._tokenize(s) for s in original_sents]
            reference_tokens = [self._tokenize(s) for s in reference_sents]
            summary_tokens = [self._tokenize(s) for s in summary_sents]
            
            # Compute SARI components
            # This is a simplified version
            total_score = 0.0
            count = 0
            
            for sum_tokens in summary_tokens:
                # Compute with reference
                for ref_tokens in reference_tokens:
                    # F1-score-like metric
                    overlap = len(set(sum_tokens) & set(ref_tokens))
                    precision = overlap / len(sum_tokens) if len(sum_tokens) > 0 else 0
                    recall = overlap / len(ref_tokens) if len(ref_tokens) > 0 else 0
                    
                    if precision + recall > 0:
                        f1 = 2 * precision * recall / (precision + recall)
                        total_score += f1
                        count += 1
                        break
            
            return total_score / count if count > 0 else 0.0
        
        except Exception as e:
            return 0.0
    
    def _compute_readability(self, original: str, summary: str) -> Dict[str, float]:
        """Compute readability metrics"""
        if textstat is None:
            return {
                'original_fkgl': 0.0,
                'summary_fkgl': 0.0,
                'original_readability': 0.0,
                'summary_readability': 0.0,
                'readability_improvement': 0.0
            }
        return {
            'original_fkgl': textstat.flesch_kincaid_grade(original),
            'summary_fkgl': textstat.flesch_kincaid_grade(summary),
            'original_readability': textstat.flesch_reading_ease(original),
            'summary_readability': textstat.flesch_reading_ease(summary),
            'readability_improvement': (
                textstat.flesch_reading_ease(summary) - 
                textstat.flesch_reading_ease(original)
            )
        }
    
    def _compute_word_overlap(self, original: str, summary: str) -> Dict[str, float]:
        """Compute word overlap ratio"""
        # Tokenize and lowercase
        original_tokens = set(word.lower() for word in self._tokenize(original))
        summary_tokens = set(word.lower() for word in self._tokenize(summary))
        
        # Common words
        common_words = original_tokens & summary_tokens
        
        # Ratios
        overlap_ratio = len(common_words) / len(summary_tokens) if len(summary_tokens) > 0 else 0
        coverage_ratio = len(common_words) / len(original_tokens) if len(original_tokens) > 0 else 0
        
        return {
            'word_overlap_ratio': overlap_ratio,
            'coverage_ratio': coverage_ratio
        }
    
    def _compute_preservation_score(self, original: str, summary: str) -> float:
        """
        Compute preservation score (semantic similarity)
        Simplified version using word overlap and length
        """
        # Use BERTScore F1 as preservation score
        if bert_score is None:
            # Fallback to word overlap
            original_tokens = set(self._tokenize(original))
            summary_tokens = set(self._tokenize(summary))
            common = len(original_tokens & summary_tokens)
            total = len(summary_tokens)
            return common / total if total > 0 else 0.0
        
        try:
            _, _, F1 = bert_score([summary], [original], lang='en', verbose=False)
            return float(F1[0])
        except:
            # Fallback to word overlap
            original_tokens = set(self._tokenize(original))
            summary_tokens = set(self._tokenize(summary))
            common = len(original_tokens & summary_tokens)
            total = len(summary_tokens)
            return common / total if total > 0 else 0.0
    
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple word tokenization"""
        # Remove punctuation and split
        text = re.sub(r'[^\w\s]', ' ', text)
        return text.lower().split()
    
    def format_evaluation_report(self, scores: Dict[str, float]) -> str:
        """Format evaluation scores as a readable report"""
        report = "=== Evaluation Report ===\n\n"
        
        # ROUGE scores
        if 'rouge1' in scores:
            report += "ROUGE Scores:\n"
            report += f"  ROUGE-1: {scores['rouge1']:.3f}\n"
            report += f"  ROUGE-2: {scores['rouge2']:.3f}\n"
            report += f"  ROUGE-L: {scores['rougeL']:.3f}\n\n"
        
        # BERTScore
        if 'bertscore_f1' in scores:
            report += "BERTScore (Semantic Similarity):\n"
            report += f"  F1: {scores['bertscore_f1']:.3f}\n\n"
        
        # Readability
        if 'readability_improvement' in scores:
            report += "Readability:\n"
            report += f"  Original FKGL: {scores['original_fkgl']:.1f}\n"
            report += f"  Summary FKGL: {scores['summary_fkgl']:.1f}\n"
            report += f"  Improvement: {scores['readability_improvement']:.2f} points\n\n"
        
        # Preservation
        if 'preservation_score' in scores:
            report += f"Preservation Score: {scores['preservation_score']:.3f}\n"
        
        return report

