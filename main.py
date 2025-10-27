"""
CLI Interface for PoliSumm
Command-line interface for document summarization
"""

import argparse
import sys
import os
from pathlib import Path

from polisumm.pipeline import PoliSummPipeline


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="PoliSumm - Legal & Policy Document Simplifier",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process a PDF file
  python main.py --input document.pdf --output summary.json
  
  # Process text directly
  python main.py --text "Your text here" --output summary.txt
  
  # Format as Markdown
  python main.py --input document.pdf --format markdown --output summary.md
        """
    )
    
    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--input', '-i', type=str, help='Input file path (PDF, DOCX, or TXT)')
    input_group.add_argument('--text', '-t', type=str, help='Direct text input')
    
    # Output options
    parser.add_argument('--output', '-o', type=str, help='Output file path')
    parser.add_argument('--format', '-f', choices=['json', 'markdown', 'txt'], 
                       default='json', help='Output format')
    parser.add_argument('--config', '-c', type=str, default='config.yaml',
                       help='Path to configuration file')
    
    # Options
    parser.add_argument('--quiet', '-q', action='store_true',
                       help='Suppress progress output')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose output')
    
    args = parser.parse_args()
    
    # Display banner
    if not args.quiet:
        print("=" * 60)
        print("PoliSumm - Legal & Policy Document Simplifier")
        print("=" * 60)
        print()
    
    try:
        # Initialize pipeline
        if not args.quiet:
            print("Initializing PoliSumm pipeline...")
        
        pipeline = PoliSummPipeline(args.config)
        
        # Process document
        if args.input:
            if not args.quiet:
                print(f"Processing file: {args.input}")
            
            results = pipeline.process(input_file=args.input)
            
        elif args.text:
            if not args.quiet:
                print("Processing text input...")
            
            results = pipeline.process(input_text=args.text)
        
        # Display summary
        if not args.quiet:
            print("\n" + "=" * 60)
            print("SUMMARY")
            print("=" * 60)
            print()
            print(results['summary'])
            print()
        
        # Save output
        if args.output:
            pipeline.save_results(results, args.output)
        elif not args.quiet:
            # Print to stdout
            print("=" * 60)
            print("FULL RESULTS")
            print("=" * 60)
            print()
            print(pipeline.format_output(results, args.format))
        
        # Display evaluation metrics
        if args.verbose:
            from polisumm.evaluator import SummaryEvaluator
            evaluator = SummaryEvaluator()
            report = evaluator.format_evaluation_report(results['evaluation'])
            print()
            print(report)
        
        if not args.quiet:
            print("\nProcessing complete!")
        
        return 0
    
    except FileNotFoundError as e:
        print(f"Error: File not found: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

