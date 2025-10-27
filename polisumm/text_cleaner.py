"""
Text Cleaning and Preprocessing Module
"""

import re
import ftfy
from typing import List, Tuple
import unicodedata


class TextCleaner:
    """Clean and normalize text for NLP processing"""
    
    def __init__(self):
        # Patterns for cleaning
        self.url_pattern = re.compile(
            r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        )
        self.email_pattern = re.compile(
            r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
        )
        self.emoji_pattern = re.compile(
            "["
            "\U0001F600-\U0001F64F"  # emoticons
            "\U0001F300-\U0001F5FF"  # symbols & pictographs
            "\U0001F680-\U0001F6FF"  # transport & map symbols
            "\U0001F1E0-\U0001F1FF"  # flags
            "\U00002702-\U000027B0"
            "\U000024C2-\U0001F251"
            "]+", flags=re.UNICODE
        )
    
    def clean(self, text: str, normalize_unicode: bool = True) -> str:
        """
        Clean text by removing unwanted elements
        
        Args:
            text: Raw text to clean
            normalize_unicode: Whether to normalize Unicode characters
            
        Returns:
            Cleaned text
        """
        if not text:
            return ""
        
        # Fix encoding issues
        if normalize_unicode:
            text = ftfy.fix_text(text, normalization='NFKC')
        
        # Remove HTML/XML tags
        text = self._remove_html_tags(text)
        
        # Remove URLs
        text = self.url_pattern.sub(' ', text)
        
        # Remove email addresses
        text = self.email_pattern.sub(' ', text)
        
        # Remove emojis
        text = self.emoji_pattern.sub(' ', text)
        
        # Remove special symbols but keep punctuation that's important for legal text
        text = self._clean_special_chars(text)
        
        # Remove extra whitespace
        text = self._normalize_whitespace(text)
        
        # Fix OCR artifacts
        text = self._fix_ocr_artifacts(text)
        
        return text.strip()
    
    def _remove_html_tags(self, text: str) -> str:
        """Remove HTML/XML tags"""
        # Basic tag removal
        text = re.sub(r'<[^>]+>', '', text)
        return text
    
    def _clean_special_chars(self, text: str) -> str:
        """
        Remove special characters but preserve important punctuation
        Keep: . , ; : ! ? ( ) [ ] { } - " '
        Remove: § ¶ © ® ™ etc. (but keep if it's a legal symbol)
        """
        # Keep legal symbols that might be important
        legal_symbols = r'§¶©®™'
        # Remove non-printable characters
        text = ''.join(char for char in text if unicodedata.category(char)[0] != 'C' or char in '\n\r\t')
        return text
    
    def _normalize_whitespace(self, text: str) -> str:
        """Normalize whitespace to single spaces, preserve newlines"""
        # Replace multiple spaces with single space
        text = re.sub(r'[ \t]+', ' ', text)
        # Normalize newlines
        text = re.sub(r'\r\n|\r', '\n', text)
        # Remove multiple consecutive newlines
        text = re.sub(r'\n{3,}', '\n\n', text)
        # Remove trailing spaces
        text = re.sub(r' +\n', '\n', text)
        return text
    
    def _fix_ocr_artifacts(self, text: str) -> str:
        """Fix common OCR errors"""
        replacements = {
            'ﬁ': 'fi',
            'ﬂ': 'fl',
            'ﬀ': 'ff',
            'ﬃ': 'ffi',
            'ﬄ': 'ffl',
            '…': '...',
            '"': '"',
            '"': '"',
            ''': "'",
            ''': "'",
            '–': '-',
            '—': '-',
        }
        
        for old, new in replacements.items():
            text = text.replace(old, new)
        
        return text
    
    def to_lowercase(self, text: str) -> str:
        """Convert text to lowercase"""
        return text.lower()
    
    def remove_punctuation(self, text: str, keep_important: bool = True) -> str:
        """
        Remove punctuation
        
        Args:
            text: Text to process
            keep_important: If True, keep punctuation important for legal text
        """
        if keep_important:
            # Keep punctuation marks
            text = re.sub(r'[^\w\s.,;:!?()[]{}\-"\']', '', text)
        else:
            text = re.sub(r'[^\w\s]', '', text)
        
        return text
    
    def remove_numbers(self, text: str, keep_important: bool = True) -> str:
        """
        Remove numbers
        
        Args:
            text: Text to process
            keep_important: If True, keep dates and amounts
        """
        if keep_important:
            # Keep patterns like dates, percentages, amounts
            # This is complex, so for now we'll keep numbers
            return text
        
        # Remove all digits
        text = re.sub(r'\d+', '', text)
        return text
    
    def normalize_unicode(self, text: str) -> str:
        """Normalize Unicode characters"""
        return unicodedata.normalize('NFKC', text)

