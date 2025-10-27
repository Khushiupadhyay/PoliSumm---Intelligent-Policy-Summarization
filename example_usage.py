"""
Example usage of PoliSumm pipeline
"""

from polisumm.pipeline import PoliSummPipeline


def example_file_processing():
    """Example: Process a file"""
    print("Example 1: Processing a file...")
    
    # Initialize pipeline
    pipeline = PoliSummPipeline()
    
    # Process file (make sure you have a test document)
    results = pipeline.process(input_file="sample_document.pdf")
    
    # Print summary
    print("\nSummary:")
    print(results['summary'])
    
    # Save results
    pipeline.save_results(results, "example_output.json")


def example_text_processing():
    """Example: Process direct text input"""
    print("\nExample 2: Processing text input...")
    
    sample_text = """
    TERMS OF SERVICE
    
    By using our service, you agree to the following terms:
    
    1. You must be at least 18 years old to use this service.
    2. You are responsible for maintaining the confidentiality of your account.
    3. We reserve the right to terminate accounts that violate these terms.
    4. Refunds will be processed within 30 days of request.
    5. We are not liable for any indirect damages arising from use of the service.
    
    For questions, contact us at support@example.com.
    """
    
    # Initialize pipeline
    pipeline = PoliSummPipeline()
    
    # Process text
    results = pipeline.process(input_text=sample_text)
    
    # Display results
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(results['summary'])
    
    print("\n" + "="*60)
    print("DOCUMENT STATISTICS")
    print("="*60)
    stats = results['statistics']
    print(f"Total sentences: {stats['total_sentences']}")
    print(f"Total words: {stats['total_words']}")
    print(f"Average sentence length: {stats['avg_sentence_length']:.1f}")
    
    print("\n" + "="*60)
    print("EVALUATION METRICS")
    print("="*60)
    eval_scores = results['evaluation']
    print(f"Preservation score: {eval_scores.get('preservation_score', 'N/A'):.3f}")
    print(f"Readability improvement: {eval_scores.get('readability_improvement', 'N/A'):.2f} points")


def example_custom_config():
    """Example: Using custom configuration"""
    print("\nExample 3: Custom configuration...")
    
    # You can create a custom config.yaml or pass a different path
    pipeline = PoliSummPipeline(config_path="config.yaml")
    
    # Process with custom settings
    results = pipeline.process(
        input_text="Your document text here..."
    )
    
    # Output in different formats
    print("\n--- JSON Format ---")
    print(pipeline.format_output(results, format='json')[:200] + "...")
    
    print("\n--- Markdown Format ---")
    print(pipeline.format_output(results, format='markdown')[:200] + "...")


def example_advanced_usage():
    """Example: Advanced usage with individual modules"""
    print("\nExample 4: Advanced module usage...")
    
    from polisumm.text_extractor import TextExtractor
    from polisumm.text_cleaner import TextCleaner
    from polisumm.nlp_pipeline import NLPPipeline
    from polisumm.sentence_scorer import SentenceScorer
    from polisumm.summarizer import DocumentSummarizer
    
    sample_text = "Your legal document text here..."
    
    # Step-by-step processing
    cleaner = TextCleaner()
    cleaned = cleaner.clean(sample_text)
    
    nlp = NLPPipeline()
    processed = nlp.process_document(cleaned)
    
    scorer = SentenceScorer()
    key_sentences = scorer.extract_key_sentences(processed, top_n=5)
    
    summarizer = DocumentSummarizer()
    summary = summarizer.simplify_for_legal_text(cleaned, key_sentences)
    
    print("Key sentences extracted:")
    for i, sent in enumerate(key_sentences[:3], 1):
        print(f"{i}. {sent}")
    
    print(f"\nGenerated summary:\n{summary['summary']}")


if __name__ == "__main__":
    print("="*60)
    print("PoliSumm Example Usage")
    print("="*60)
    print("\nNote: These examples demonstrate how to use PoliSumm.")
    print("Please ensure you have the necessary dependencies installed.")
    print("\nRunning example_text_processing()...\n")
    
    example_text_processing()
    
    # Uncomment to run other examples:
    # example_file_processing()
    # example_custom_config()
    # example_advanced_usage()

