"""
Test script for PoliSumm pipeline
Run this to verify your installation is working correctly
"""

import sys
import traceback


def test_imports():
    """Test if all modules can be imported"""
    print("Testing imports...")
    
    try:
        from polisumm.text_extractor import TextExtractor
        print("  [OK] text_extractor")
        
        from polisumm.text_cleaner import TextCleaner
        print("  [OK] text_cleaner")
        
        from polisumm.nlp_pipeline import NLPPipeline
        print("  [OK] nlp_pipeline")
        
        from polisumm.sentence_scorer import SentenceScorer
        print("  [OK] sentence_scorer")
        
        from polisumm.summarizer import DocumentSummarizer
        print("  [OK] summarizer")
        
        from polisumm.evaluator import SummaryEvaluator
        print("  [OK] evaluator")
        
        from polisumm.translator import Translator
        print("  [OK] translator")
        
        from polisumm.pipeline import PoliSummPipeline
        print("  [OK] pipeline")
        
        return True
        
    except Exception as e:
        print(f"  [FAIL] Import failed: {e}")
        traceback.print_exc()
        return False


def test_text_extraction():
    """Test text extraction"""
    print("\nTesting text extraction...")
    
    try:
        from polisumm.text_extractor import TextExtractor
        
        extractor = TextExtractor()
        
        # Test with sample text
        sample_text = "This is a test document. It has multiple sentences. Each sentence is important."
        result = extractor.extract_from_text(sample_text)
        
        print(f"  [OK] Text length: {len(result['text'])} characters")
        print(f"  [OK] Sections found: {len(result['sections'])}")
        
        return True
        
    except Exception as e:
        print(f"  [FAIL] Text extraction failed: {e}")
        traceback.print_exc()
        return False


def test_text_cleaning():
    """Test text cleaning"""
    print("\nTesting text cleaning...")
    
    try:
        from polisumm.text_cleaner import TextCleaner
        
        cleaner = TextCleaner()
        
        dirty_text = "This is a test  with   extra spaces and http://example.com and emails@test.com"
        clean_text = cleaner.clean(dirty_text)
        
        print(f"  [OK] Original length: {len(dirty_text)}")
        print(f"  [OK] Cleaned length: {len(clean_text)}")
        
        return True
        
    except Exception as e:
        print(f"  [FAIL] Text cleaning failed: {e}")
        traceback.print_exc()
        return False


def test_nlp_pipeline():
    """Test NLP pipeline"""
    print("\nTesting NLP pipeline...")
    
    try:
        from polisumm.nlp_pipeline import NLPPipeline
        
        nlp = NLPPipeline()
        
        sample_text = "The user must provide accurate information. The Company shall not be liable."
        doc = nlp.process_document(sample_text)
        
        print(f"  [OK] Processed sentences: {doc['total_sentences']}")
        print(f"  [OK] Total words: {doc['total_words']}")
        print(f"  [OK] First sentence entities: {len(doc['sentences'][0]['entities'])}")
        
        return True
        
    except Exception as e:
        print(f"  [FAIL] NLP pipeline failed: {e}")
        traceback.print_exc()
        return False


def test_sentence_scorer():
    """Test sentence scoring"""
    print("\nTesting sentence scoring...")
    
    try:
        from polisumm.nlp_pipeline import NLPPipeline
        from polisumm.sentence_scorer import SentenceScorer
        
        sample_text = """
        The Company provides services to users.
        Users must agree to the terms and conditions.
        Payment is due within 30 days of invoice date.
        Refunds will be processed within 14 business days.
        All content is protected by copyright laws.
        """
        
        nlp = NLPPipeline()
        processed = nlp.process_document(sample_text)
        
        scorer = SentenceScorer()
        key_sentences = scorer.extract_key_sentences(processed, summary_ratio=0.3)
        
        print(f"  [OK] Extracted {len(key_sentences)} key sentences")
        
        return True
        
    except Exception as e:
        print(f"  [FAIL] Sentence scoring failed: {e}")
        traceback.print_exc()
        return False


def test_full_pipeline():
    """Test the full pipeline"""
    print("\nTesting full pipeline...")
    
    try:
        from polisumm.pipeline import PoliSummPipeline
        
        sample_text = """
        TERMS OF SERVICE
        
        This Agreement is entered into between the User and the Company.
        By using the service, you agree to these terms.
        
        USER OBLIGATIONS
        
        You must provide accurate information.
        You shall not use the service for illegal purposes.
        You are required to maintain account confidentiality.
        
        PAYMENT TERMS
        
        All fees are due within 30 days.
        Refunds will be processed within 14 business days.
        Prices are subject to change with 30 days notice.
        
        LIABILITY
        
        The service is provided "as is" without warranties.
        We shall not be liable for indirect damages.
        Our liability is limited to the amount paid in the past 12 months.
        """
        
        pipeline = PoliSummPipeline()
        results = pipeline.process(input_text=sample_text)
        
        print(f"  [OK] Summary generated: {len(results['summary'])} characters")
        print(f"  [OK] Statistics computed")
        print(f"  [OK] Evaluation scores calculated")
        
        return True
        
    except Exception as e:
        print(f"  [FAIL] Full pipeline failed: {e}")
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("="*60)
    print("PoliSumm Pipeline Test Suite")
    print("="*60)
    
    tests = [
        ("Import Tests", test_imports),
        ("Text Extraction", test_text_extraction),
        ("Text Cleaning", test_text_cleaning),
        ("NLP Pipeline", test_nlp_pipeline),
        ("Sentence Scoring", test_sentence_scorer),
        ("Full Pipeline", test_full_pipeline),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n{test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "[OK] PASS" if result else "[FAIL] FAIL"
        print(f"{status}: {test_name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\nAll tests passed! Installation is working correctly.")
        return 0
    else:
        print(f"\n{total - passed} test(s) failed. Please check the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())

