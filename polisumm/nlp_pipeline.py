"""
NLP Pipeline Module
Handles tokenization, POS tagging, NER, lemmatization
"""

import spacy
import nltk
from typing import List, Dict, Tuple, Optional
import re


# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('tokenizers/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)


class NLPPipeline:
    """Main NLP processing pipeline"""
    
    def __init__(self, model_name: str = "en_core_web_sm"):
        """
        Initialize the NLP pipeline
        
        Args:
            model_name: spaCy model name
        """
        try:
            self.nlp = spacy.load(model_name)
        except OSError:
            print(f"Model {model_name} not found. Please install it with: python -m spacy download {model_name}")
            raise
        
        self.stopwords = set(nltk.corpus.stopwords.words('english'))
        self.legal_stopwords = {'must', 'shall', 'will', 'may', 'should', 'required'}
    
    def segment_sentences(self, text: str) -> List[str]:
        """
        Segment text into sentences
        
        Args:
            text: Input text
            
        Returns:
            List of sentences
        """
        # Use spaCy for sentence segmentation
        doc = self.nlp(text)
        sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
        return sentences
    
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into words
        
        Args:
            text: Input text
            
        Returns:
            List of tokens
        """
        doc = self.nlp(text)
        return [token.text for token in doc]
    
    def process_document(self, text: str, include_pos: bool = True, 
                        include_ner: bool = True) -> Dict[str, any]:
        """
        Process a document through the full NLP pipeline
        
        Args:
            text: Input text
            include_pos: Whether to include POS tags
            include_ner: Whether to include NER
            
        Returns:
            Dictionary with processed sentences and metadata
        """
        # Segment sentences
        sentences = self.segment_sentences(text)
        
        processed_sentences = []
        
        for idx, sentence in enumerate(sentences):
            processed = self.process_sentence(
                sentence, 
                sentence_idx=idx,
                include_pos=include_pos,
                include_ner=include_ner
            )
            processed_sentences.append(processed)
        
        return {
            'sentences': processed_sentences,
            'total_sentences': len(sentences),
            'total_words': sum(len(s['tokens']) for s in processed_sentences)
        }
    
    def process_sentence(self, sentence: str, sentence_idx: int = 0,
                        include_pos: bool = True, include_ner: bool = True) -> Dict[str, any]:
        """
        Process a single sentence
        
        Args:
            sentence: Sentence text
            sentence_idx: Index of sentence in document
            include_pos: Whether to include POS tags
            include_ner: Whether to include NER
            
        Returns:
            Dictionary with processed sentence data
        """
        doc = self.nlp(sentence)
        
        # Tokenization
        tokens = [token.text for token in doc]
        
        # POS tagging
        pos_tags = [token.pos_ for token in doc] if include_pos else []
        pos_detailed = [(token.text, token.pos_, token.tag_) for token in doc] if include_pos else []
        
        # NER
        entities = []
        if include_ner:
            entities = [
                {
                    'text': ent.text,
                    'label': ent.label_,
                    'start': ent.start_char,
                    'end': ent.end_char
                }
                for ent in doc.ents
            ]
        
        # Lemmatization
        lemmas = [token.lemma_ for token in doc]
        
        # Identify legal keywords
        legal_keywords = self._extract_legal_keywords(doc, pos_detailed)
        
        # Extract obligations and rights
        obligations = self._extract_obligations(doc, pos_detailed)
        
        return {
            'text': sentence,
            'idx': sentence_idx,
            'tokens': tokens,
            'lemmas': lemmas,
            'pos_tags': pos_tags,
            'pos_detailed': pos_detailed,
            'entities': entities,
            'legal_keywords': legal_keywords,
            'obligations': obligations,
            'length': len(tokens)
        }
    
    def remove_stopwords(self, tokens: List[str], keep_negations: bool = True,
                        keep_legal: bool = True) -> List[str]:
        """
        Remove stopwords from tokens
        
        Args:
            tokens: List of tokens
            keep_negations: Keep negation words (not, no, never, etc.)
            keep_legal: Keep legal stopwords (must, shall, will, etc.)
            
        Returns:
            Filtered tokens
        """
        negation_words = {'not', 'no', 'never', 'none', 'nothing', 'nobody', 
                         'nowhere', 'neither', 'nor'}
        
        filtered = []
        for token in tokens:
            token_lower = token.lower()
            
            # Keep if it's a legal term and keep_legal is True
            if keep_legal and token_lower in self.legal_stopwords:
                filtered.append(token)
            # Keep if it's a negation and keep_negations is True
            elif keep_negations and token_lower in negation_words:
                filtered.append(token)
            # Skip if it's a regular stopword
            elif token_lower not in self.stopwords:
                filtered.append(token)
        
        return filtered
    
    def _extract_legal_keywords(self, doc, pos_tags: List[Tuple[str, str, str]]) -> List[Dict[str, str]]:
        """Extract legal and financial keywords"""
        keywords = []
        
        # Legal terms
        legal_terms = {'must', 'shall', 'will', 'agree', 'agreement', 'contract', 
                      'liability', 'damages', 'breach', 'terminate', 'obligation',
                      'party', 'parties', 'applicable', 'jurisdiction'}
        
        # Financial terms
        financial_terms = {'payment', 'refund', 'fee', 'cost', 'price', 'amount',
                          'balance', 'deposit', 'charge', 'billing'}
        
        for token in doc:
            token_lower = token.text.lower()
            if token_lower in legal_terms:
                keywords.append({'word': token.text, 'type': 'legal', 'importance': 'high'})
            elif token_lower in financial_terms:
                keywords.append({'word': token.text, 'type': 'financial', 'importance': 'high'})
        
        return keywords
    
    def _extract_obligations(self, doc, pos_tags: List[Tuple[str, str, str]]) -> List[Dict[str, str]]:
        """Extract obligations, rights, and deadlines from sentence"""
        obligations = []
        
        # Look for patterns like "must X", "shall X", "required to X"
        obligation_patterns = [
            (r'(must|shall)\s+(\w+\s+)*(\w+)', 'obligation'),
            (r'(required|mandatory)\s+to\s+(\w+)', 'obligation'),
            (r'(\w+)\s+(will|shall)\s+(\w+)', 'obligation'),
        ]
        
        sentence_text = doc.text
        
        for pattern, obligation_type in obligation_patterns:
            matches = re.finditer(pattern, sentence_text, re.IGNORECASE)
            for match in matches:
                obligations.append({
                    'type': obligation_type,
                    'text': match.group(0),
                    'start': match.start(),
                    'end': match.end()
                })
        
        # Extract dates and deadlines
        date_entities = [ent for ent in doc.ents if ent.label_ in ['DATE', 'TIME']]
        for ent in date_entities:
            obligations.append({
                'type': 'deadline',
                'text': ent.text,
                'label': ent.label_
            })
        
        # Extract monetary amounts
        money_entities = [ent for ent in doc.ents if ent.label_ == 'MONEY']
        for ent in money_entities:
            obligations.append({
                'type': 'financial',
                'text': ent.text,
                'label': 'MONEY'
            })
        
        return obligations
    
    def get_document_statistics(self, processed_doc: Dict[str, any]) -> Dict[str, any]:
        """
        Calculate statistics for a processed document
        
        Args:
            processed_doc: Output from process_document
            
        Returns:
            Dictionary with statistics
        """
        all_entities = []
        all_keywords = []
        
        for sentence in processed_doc['sentences']:
            all_entities.extend(sentence['entities'])
            all_keywords.extend(sentence.get('legal_keywords', []))
        
        # Count entity types
        entity_counts = {}
        for entity in all_entities:
            entity_type = entity['label']
            entity_counts[entity_type] = entity_counts.get(entity_type, 0) + 1
        
        # Count keywords
        keyword_counts = {}
        for keyword in all_keywords:
            keyword_word = keyword['word'].lower()
            keyword_counts[keyword_word] = keyword_counts.get(keyword_word, 0) + 1
        
        return {
            'total_sentences': processed_doc['total_sentences'],
            'total_words': processed_doc['total_words'],
            'avg_sentence_length': processed_doc['total_words'] / processed_doc['total_sentences'] if processed_doc['total_sentences'] > 0 else 0,
            'entity_counts': entity_counts,
            'keyword_counts': dict(sorted(keyword_counts.items(), key=lambda x: x[1], reverse=True)[:20]),
            'total_entities': len(all_entities),
            'total_keywords': len(all_keywords)
        }

