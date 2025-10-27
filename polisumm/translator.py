"""
Translation Module
Optional multilingual support for summaries
"""

from typing import Optional, List, Dict
from transformers import pipeline


class Translator:
    """Translate summaries to other languages"""
    
    def __init__(self, target_languages: List[str] = None):
        """
        Initialize translator
        
        Args:
            target_languages: List of target languages (e.g., ['hi'])
        """
        self.target_languages = target_languages or []
        self.translators = {}
        
        # Initialize translators for each target language
        for lang in self.target_languages:
            try:
                if lang == 'hi':  # Hindi
                    # Use opus-mt or indic-trans model
                    try:
                        self.translators[lang] = pipeline(
                            "translation",
                            model="Helsinki-NLP/opus-mt-en-hi"
                        )
                    except:
                        # Fallback to other model
                        print(f"Could not load translation model for {lang}")
                # Add more languages as needed
            except Exception as e:
                print(f"Error initializing translator for {lang}: {e}")
    
    def translate(self, text: str, target_lang: str) -> Optional[str]:
        """
        Translate text to target language
        
        Args:
            text: Text to translate
            target_lang: Target language code (e.g., 'hi')
            
        Returns:
            Translated text or None if translation fails
        """
        if target_lang not in self.translators:
            return None
        
        try:
            translator = self.translators[target_lang]
            result = translator(text)
            
            if isinstance(result, list) and len(result) > 0:
                return result[0]['translation_text']
            elif isinstance(result, dict):
                return result.get('translation_text', text)
            else:
                return text
        except Exception as e:
            print(f"Error translating to {target_lang}: {e}")
            return text
    
    def translate_summary(self, summary: str, target_lang: str = 'hi') -> Dict[str, str]:
        """
        Translate a summary to the target language
        
        Args:
            summary: English summary text
            target_lang: Target language code
            
        Returns:
            Dictionary with original and translated text
        """
        translated = self.translate(summary, target_lang)
        
        return {
            'original': summary,
            'translated': translated or summary,
            'target_language': target_lang
        }
    
    def is_available(self, target_lang: str) -> bool:
        """Check if translation is available for a language"""
        return target_lang in self.translators

