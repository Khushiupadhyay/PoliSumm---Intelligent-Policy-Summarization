"""
Main Pipeline Module
Orchestrates the complete NLP workflow
"""

from typing import Dict, List, Optional, Tuple
import yaml
import json
from pathlib import Path

from polisumm.text_extractor import TextExtractor
from polisumm.text_cleaner import TextCleaner
from polisumm.nlp_pipeline import NLPPipeline
from polisumm.sentence_scorer import SentenceScorer
from polisumm.summarizer import DocumentSummarizer
from polisumm.evaluator import SummaryEvaluator
from polisumm.translator import Translator
from polisumm.pdf_generator import PDFGenerator


class PoliSummPipeline:
    """Main pipeline for document summarization"""
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize the PoliSumm pipeline
        
        Args:
            config_path: Path to configuration file
        """
        self.config = self._load_config(config_path)
        self.text_extractor = TextExtractor()
        self.text_cleaner = TextCleaner()
        self.nlp = NLPPipeline(self.config['models']['ner_model'])
        self.sentence_scorer = SentenceScorer(self.config['models']['embedding_model'])
        self.summarizer = DocumentSummarizer(self.config['models']['summarization_model'])
        self.evaluator = SummaryEvaluator()
        self.pdf_generator = PDFGenerator()
        
        # Initialize translator if needed
        self.translator = None
        if self.config.get('multilingual', {}).get('target_languages', []):
            self.translator = Translator(
                self.config['multilingual']['target_languages']
            )
    
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file"""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            print(f"Config file {config_path} not found. Using defaults.")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict:
        """Get default configuration"""
        return {
            'models': {
                'summarization_model': 'google/long-t5-tglobal-base',
                'embedding_model': 'sentence-transformers/all-mpnet-base-v2',
                'ner_model': 'en_core_web_sm'
            },
            'summarization': {
                'max_summary_length': 250,
                'min_summary_length': 50,
                'num_sentences_to_extract': 10
            },
            'evaluation': {
                'similarity_threshold': 0.7
            }
        }
    
    def process(self, input_file: str = None, input_text: str = None) -> Dict:
        """
        Process a document through the complete pipeline
        
        Args:
            input_file: Path to input file (PDF, DOCX, or TXT)
            input_text: Direct text input
            
        Returns:
            Complete processing results dictionary
        """
        results = {}
        
        # Step 1: Extract text
        print("Step 1: Extracting text...")
        if input_file:
            extracted = self.text_extractor.extract(input_file)
        elif input_text:
            extracted = self.text_extractor.extract_from_text(input_text)
        else:
            raise ValueError("Either input_file or input_text must be provided")
        
        original_text = extracted['text']
        results['metadata'] = extracted['metadata']
        results['sections'] = extracted['sections']
        
        # Step 2: Clean text
        print("Step 2: Cleaning text...")
        cleaned_text = self.text_cleaner.clean(original_text)
        results['original_length'] = len(original_text)
        results['cleaned_length'] = len(cleaned_text)
        
        # Step 3-11: NLP Processing
        print("Step 3-11: NLP processing...")
        processed_doc = self.nlp.process_document(cleaned_text)
        results['statistics'] = self.nlp.get_document_statistics(processed_doc)
        
        # Step 12: Sentence scoring and ranking
        print("Step 12: Scoring and ranking sentences...")
        extracted_sentences = self.sentence_scorer.extract_key_sentences(
            processed_doc,
            summary_ratio=0.3
        )
        results['extracted_sentences'] = extracted_sentences
        
        # Step 13: Summarization
        print("Step 13: Generating summary...")
        summary_result = self.summarizer.simplify_for_legal_text(
            cleaned_text,
            extracted_sentences=extracted_sentences
        )
        results['summary'] = summary_result['summary']
        results['summary_length'] = summary_result['length']
        
        # Step 14: Evaluation
        print("Step 14: Evaluating summary...")
        evaluation_scores = self.evaluator.evaluate(cleaned_text, results['summary'])
        results['evaluation'] = evaluation_scores
        
        # Step 15: Quality control
        print("Step 15: Quality control...")
        if evaluation_scores.get('preservation_score', 0) < self.config['evaluation']['similarity_threshold']:
            print("Warning: Preservation score below threshold. Using extractive fallback.")
            results['summary'] = self._extractive_fallback(extracted_sentences)
        
        # Add disclaimer
        results['summary'] += "\n\n⚠️ This summary is for informational purposes only and not legal advice."
        
        # Optional: Translation
        if self.translator:
            results['translations'] = {}
            for lang in self.config['multilingual']['target_languages']:
                if self.translator.is_available(lang):
                    print(f"Translating to {lang}...")
                    translation = self.translator.translate_summary(results['summary'], lang)
                    results['translations'][lang] = translation
        
        return results
    
    def _extractive_fallback(self, sentences: List[str]) -> str:
        """Generate extractive summary when preservation score is low"""
        # Join top sentences
        summary = ". ".join(sentences)
        return summary
    
    def format_output(self, results: Dict, format: str = 'json') -> str:
        """
        Format results for output
        
        Args:
            results: Processing results dictionary
            format: Output format ('json', 'markdown', 'txt')
            
        Returns:
            Formatted string
        """
        if format == 'json':
            return json.dumps(results, indent=2)
        elif format == 'markdown':
            return self._format_markdown(results)
        else:  # txt
            return self._format_text(results)
    
    def _format_markdown(self, results: Dict) -> str:
        """Format results as Markdown"""
        output = "# PoliSumm Summary Report\n\n"
        
        # Metadata
        output += "## Document Information\n\n"
        output += f"**Title:** {results['metadata'].get('title', 'N/A')}\n"
        output += f"**Author:** {results['metadata'].get('author', 'N/A')}\n"
        output += f"**Pages:** {results['metadata'].get('pages', 'N/A')}\n\n"
        
        # Summary
        output += "## Summary\n\n"
        output += results['summary'] + "\n\n"
        
        # Statistics
        output += "## Document Statistics\n\n"
        stats = results['statistics']
        output += f"- Total Sentences: {stats['total_sentences']}\n"
        output += f"- Total Words: {stats['total_words']}\n"
        output += f"- Average Sentence Length: {stats['avg_sentence_length']:.1f}\n\n"
        
        # Evaluation
        output += "## Evaluation Metrics\n\n"
        eval_scores = results['evaluation']
        
        if 'bertscore_f1' in eval_scores:
            output += f"- **Preservation Score (BERTScore):** {eval_scores['bertscore_f1']:.3f}\n"
        if 'readability_improvement' in eval_scores:
            output += f"- **Readability Improvement:** {eval_scores['readability_improvement']:.2f} points\n"
        
        return output
    
    def _format_text(self, results: Dict) -> str:
        """Format results as plain text"""
        output = "POLISUMM SUMMARY REPORT\n"
        output += "=" * 50 + "\n\n"
        output += "SUMMARY:\n"
        output += "-" * 50 + "\n"
        output += results['summary'] + "\n\n"
        output += "=" * 50 + "\n"
        return output
    
    def save_results(self, results: Dict, output_path: str):
        """Save results to file"""
        output_format = 'json'
        
        if output_path.endswith('.md'):
            output_format = 'markdown'
        elif output_path.endswith('.txt'):
            output_format = 'txt'
        
        formatted = self.format_output(results, output_format)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(formatted)
        
        print(f"Results saved to {output_path}")
    
    def generate_summary_pdf(self, results: Dict) -> bytes:
        """
        Generate PDF for summary
        
        Args:
            results: Processing results dictionary
            
        Returns:
            PDF content as bytes
        """
        return self.pdf_generator.generate_summary_pdf(results)
    
    def generate_key_sentences_pdf(self, results: Dict) -> bytes:
        """
        Generate PDF for key extracted sentences
        
        Args:
            results: Processing results dictionary
            
        Returns:
            PDF content as bytes
        """
        return self.pdf_generator.generate_key_sentences_pdf(results)
    
    def save_summary_pdf(self, results: Dict, output_path: str):
        """
        Save summary as PDF
        
        Args:
            results: Processing results dictionary
            output_path: Path to save the PDF
        """
        pdf_content = self.generate_summary_pdf(results)
        self.pdf_generator.save_pdf(pdf_content, output_path)
    
    def save_key_sentences_pdf(self, results: Dict, output_path: str):
        """
        Save key sentences as PDF
        
        Args:
            results: Processing results dictionary
            output_path: Path to save the PDF
        """
        pdf_content = self.generate_key_sentences_pdf(results)
        self.pdf_generator.save_pdf(pdf_content, output_path)

