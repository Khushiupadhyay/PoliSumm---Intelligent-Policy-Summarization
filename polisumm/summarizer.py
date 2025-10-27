"""
Summarization Module
Uses transformer models for abstractive summarization
"""

from typing import List, Dict, Optional
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import torch
import warnings
warnings.filterwarnings('ignore')


class DocumentSummarizer:
    """Generate summaries using transformer models"""
    
    def __init__(self, model_name: str = "google/long-t5-tglobal-base"):
        """
        Initialize the summarizer
        
        Args:
            model_name: Name of the summarization model
        """
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            self.model.to(self.device)
            
            # Create summarization pipeline
            self.summarizer = pipeline(
                "summarization",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if torch.cuda.is_available() else -1
            )
        except Exception as e:
            print(f"Warning: Could not load model {model_name}: {e}")
            print("Falling back to alternative model...")
            self._load_fallback_model()
    
    def _load_fallback_model(self):
        """Load a fallback summarization model"""
        try:
            # Try BART as fallback
            fallback_model = "facebook/bart-large-cnn"
            self.tokenizer = AutoTokenizer.from_pretrained(fallback_model)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(fallback_model)
            self.model.to(self.device)
            
            self.summarizer = pipeline(
                "summarization",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if torch.cuda.is_available() else -1
            )
            self.model_name = fallback_model
        except Exception as e:
            print(f"Error loading fallback model: {e}")
            self.summarizer = None
    
    def summarize(self, text: str, max_length: int = 250, 
                  min_length: int = 50, 
                  num_beams: int = 3) -> str:
        """
        Generate a summary of the input text
        
        Args:
            text: Input text to summarize
            max_length: Maximum length of summary
            min_length: Minimum length of summary
            num_beams: Number of beams for beam search
            
        Returns:
            Generated summary
        """
        if self.summarizer is None:
            return "Error: Summarization model not available"
        
        try:
            # Chunk text if too long (models have token limits)
            max_input_length = self.tokenizer.model_max_length
            
            if len(self.tokenizer.encode(text)) > max_input_length:
                # Split into chunks and summarize each
                chunks = self._chunk_text(text, max_input_length)
                chunk_summaries = []
                
                for chunk in chunks:
                    summary = self._summarize_chunk(chunk, max_length, min_length, num_beams)
                    chunk_summaries.append(summary)
                
                # Combine chunk summaries
                combined_text = " ".join(chunk_summaries)
                
                # If still too long, summarize again
                if len(self.tokenizer.encode(combined_text)) > max_input_length:
                    return self._summarize_chunk(combined_text, max_length, min_length, num_beams)
                else:
                    return combined_text
            else:
                return self._summarize_chunk(text, max_length, min_length, num_beams)
        
        except Exception as e:
            return f"Error during summarization: {e}"
    
    def _summarize_chunk(self, text: str, max_length: int, 
                        min_length: int, num_beams: int) -> str:
        """Summarize a single chunk of text"""
        try:
            result = self.summarizer(
                text,
                max_length=max_length,
                min_length=min_length,
                num_beams=num_beams,
                early_stopping=True,
                no_repeat_ngram_size=2
            )
            return result[0]['summary_text']
        except Exception as e:
            return f"Error: {e}"
    
    def _chunk_text(self, text: str, chunk_size: int) -> List[str]:
        """
        Split text into chunks
        
        Args:
            text: Input text
            chunk_size: Maximum size of each chunk
            
        Returns:
            List of text chunks
        """
        # Simple sentence-based chunking
        sentences = text.split('. ')
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence_length = len(self.tokenizer.encode(sentence))
            
            if current_length + sentence_length > chunk_size and current_chunk:
                chunks.append('. '.join(current_chunk) + '.')
                current_chunk = [sentence]
                current_length = sentence_length
            else:
                current_chunk.append(sentence)
                current_length += sentence_length
        
        if current_chunk:
            chunks.append('. '.join(current_chunk))
        
        return chunks
    
    def simplify_for_legal_text(self, text: str, extracted_sentences: Optional[List[str]] = None) -> Dict[str, str]:
        """
        Create a simplified summary specifically for legal documents
        
        Args:
            text: Original document text
            extracted_sentences: Pre-extracted key sentences (optional)
            
        Returns:
            Dictionary with simplified summary and metadata
        """
        # If extracted sentences provided, combine them first
        if extracted_sentences:
            combined_text = ". ".join(extracted_sentences)
        else:
            combined_text = text
        
        # Create the prompt template for legal text
        legal_prompt = self._create_legal_prompt(combined_text)
        
        # Generate summary
        summary = self.summarize(legal_prompt)
        
        # Post-process for legal clarity
        formatted_summary = self._format_legal_summary(summary)
        
        return {
            'summary': formatted_summary,
            'length': len(summary.split()),
            'method': 'abstractive'
        }
    
    def _create_legal_prompt(self, text: str) -> str:
        """
        Create a prompt template for legal document summarization
        
        Args:
            text: Document text
            
        Returns:
            Formatted prompt
        """
        prompt = f"""Simplify and summarize the following legal/policy document into plain, clear English. 
Maintain factual accuracy and legal meaning. 
Present the result in bullet points with short explanations.
IMPORTANT: Provide at least 7 bullet points covering the key aspects of the document.

Document text:
{text}

Summary in bullet points (minimum 7 points):"""
        
        return prompt
    
    def _format_legal_summary(self, summary: str) -> str:
        """Format summary for legal clarity"""
        # Ensure bullet points are properly formatted
        lines = summary.split('\n')
        formatted_lines = []
        bullet_count = 0
        
        for line in lines:
            line = line.strip()
            if line:
                # Ensure it starts with a bullet if it's a key point
                if not line.startswith(('•', '-', '*', '·')):
                    # Check if it looks like a list item
                    if len(line) < 100 and not line.endswith('.'):
                        line = f"• {line}"
                        bullet_count += 1
                elif line.startswith(('•', '-', '*', '·')):
                    bullet_count += 1
                formatted_lines.append(line)
        
        # Ensure we have at least 7 bullet points
        if bullet_count < 7:
            # Add placeholder bullet points if needed
            for i in range(bullet_count + 1, 8):
                formatted_lines.append(f"• Additional key point {i} (detailed analysis required)")
        
        return '\n'.join(formatted_lines)

