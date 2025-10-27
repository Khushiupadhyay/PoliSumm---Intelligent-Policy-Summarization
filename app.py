"""
Streamlit Dashboard for PoliSumm
Interactive web interface for document summarization
"""

import streamlit as st
import os
import tempfile
from pathlib import Path
import plotly.graph_objects as go
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Import PoliSumm modules
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from polisumm.pipeline import PoliSummPipeline

# Page config
st.set_page_config(
    page_title="PoliSumm - Legal & Policy Simplifier",
    page_icon="📘",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .summary-box {
        background-color: #f0f0f0;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
        color: #000;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

def main():
    """Main Streamlit app"""
    
    # Header
    st.markdown('<div class="main-header">📘 PoliSumm</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Legal & Policy Document Simplifier</div>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Input method
        input_method = st.radio(
            "Choose Input Method:",
            ["File Upload", "Text Input", "Demo Mode"]
        )
        
        # Model selection
        st.subheader("Model Settings")
        use_fast_model = st.checkbox("Use Faster Model (BART instead of LongT5)", False)
        
        # Advanced options
        with st.expander("Advanced Options"):
            max_summary_length = st.slider("Max Summary Length (words)", 100, 500, 250)
            num_sentences = st.slider("Extract Top N Sentences", 5, 20, 10)
            show_detailed_stats = st.checkbox("Show Detailed Statistics", True)
    
    # Main content area
    if input_method == "File Upload":
        handle_file_upload(use_fast_model, max_summary_length, num_sentences, show_detailed_stats)
    elif input_method == "Text Input":
        handle_text_input(use_fast_model, max_summary_length, num_sentences, show_detailed_stats)
    else:
        handle_demo_mode(use_fast_model, max_summary_length, num_sentences, show_detailed_stats)


def handle_file_upload(use_fast_model, max_len, num_sents, show_stats):
    """Handle file upload input"""
    st.header("📄 Upload Document")
    
    uploaded_file = st.file_uploader(
        "Choose a file",
        type=['pdf', 'docx', 'txt'],
        help="Upload a PDF, DOCX, or TXT file"
    )
    
    if uploaded_file is not None:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name
        
        # Process button
        if st.button("🚀 Generate Summary", type="primary"):
            process_document(tmp_file_path, use_fast_model, max_len, num_sents, show_stats)
        
        # Clean up
        if os.path.exists(tmp_file_path):
            os.unlink(tmp_file_path)


def handle_text_input(use_fast_model, max_len, num_sents, show_stats):
    """Handle direct text input"""
    st.header("✍️ Enter Text")
    
    text_input = st.text_area(
        "Paste or type your document text here:",
        height=300,
        help="Enter the text you want to summarize"
    )
    
    if st.button("🚀 Generate Summary", type="primary"):
        if text_input.strip():
            process_text(text_input, use_fast_model, max_len, num_sents, show_stats)
        else:
            st.error("Please enter some text to summarize")


def handle_demo_mode(use_fast_model, max_len, num_sents, show_stats):
    """Handle demo mode with sample document"""
    st.header("🎯 Demo Mode")
    st.info("This demo uses a sample legal document for demonstration purposes.")
    
    # Sample legal text
    sample_text = """
    TERMS AND CONDITIONS
    
    SECTION 1: GENERAL TERMS
    
    This Agreement ("Agreement") is entered into between the User ("you") and 
    PoliSumm LLC ("Company", "we", "us") effective as of the date you accept 
    these terms. By accessing or using our services, you agree to be bound by 
    these Terms and Conditions.
    
    SECTION 2: USER OBLIGATIONS
    
    You must provide accurate information when using our services. You shall not 
    use our services for any illegal purposes or in any way that could harm the 
    Company or other users. You are required to maintain the confidentiality of 
    your account credentials.
    
    SECTION 3: INTELLECTUAL PROPERTY
    
    All content, features, and functionality of our services are owned by the 
    Company and are protected by international copyright laws. You may not 
    reproduce, distribute, or create derivative works without express written 
    permission.
    
    SECTION 4: LIABILITY AND WARRANTIES
    
    Our services are provided "as is" without warranties of any kind. We shall 
    not be liable for any indirect, incidental, or consequential damages arising 
    from your use of our services. Our liability is limited to the amount you 
    paid for the service in the past 12 months.
    
    SECTION 5: TERMINATION
    
    We reserve the right to terminate your access to our services at any time 
    for violation of these terms. You may terminate your account at any time by 
    contacting customer support.
    """
    
    if st.button("🚀 Process Demo Document", type="primary"):
        process_text(sample_text, use_fast_model, max_len, num_sents, show_stats)


def process_document(file_path, use_fast_model, max_len, num_sents, show_stats):
    """Process a document file"""
    with st.spinner("Processing document..."):
        try:
            # Initialize pipeline
            pipeline = PoliSummPipeline()
            
            # Process document
            results = pipeline.process(input_file=file_path)
            
            # Display results
            display_results(results, show_stats)
            
        except Exception as e:
            st.error(f"Error processing document: {str(e)}")


def process_text(text, use_fast_model, max_len, num_sents, show_stats):
    """Process direct text input"""
    with st.spinner("Processing text..."):
        try:
            # Initialize pipeline
            pipeline = PoliSummPipeline()
            
            # Process text
            results = pipeline.process(input_text=text)
            
            # Display results
            display_results(results, show_stats)
            
        except Exception as e:
            st.error(f"Error processing text: {str(e)}")


def display_results(results, show_stats):
    """Display summarization results"""
    
    # Summary section
    st.header("📝 Summary")
    st.markdown(f'<div class="summary-box">{results["summary"]}</div>', unsafe_allow_html=True)
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Original Words", results.get('original_length', 0))
    with col2:
        st.metric("Summary Words", results.get('summary_length', 0))
    with col3:
        st.metric("Compression Ratio", 
                 f"{100 - (results.get('summary_length', 0) / max(results.get('original_length', 1), 1) * 100):.0f}%")
    with col4:
        eval_scores = results.get('evaluation', {})
        st.metric("Preservation Score", 
                 f"{eval_scores.get('preservation_score', 0):.2f}")
    
    # Statistics section
    if show_stats:
        st.header("📊 Document Statistics")
        
        stats = results.get('statistics', {})
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Basic Statistics")
            st.write(f"- Total Sentences: {stats.get('total_sentences', 0)}")
            st.write(f"- Total Words: {stats.get('total_words', 0)}")
            st.write(f"- Average Sentence Length: {stats.get('avg_sentence_length', 0):.1f} words")
        
        with col2:
            st.subheader("Named Entities")
            entity_counts = stats.get('entity_counts', {})
            if entity_counts:
                for entity_type, count in list(entity_counts.items())[:5]:
                    st.write(f"- {entity_type}: {count}")
    
    # Evaluation metrics
    st.header("🎯 Evaluation Metrics")
    
    eval_scores = results.get('evaluation', {})
    
    if eval_scores:
        col1, col2 = st.columns(2)
        
        with col1:
            # Preservation and similarity
            st.subheader("Preservation & Similarity")
            
            if 'bertscore_f1' in eval_scores:
                st.progress(eval_scores['bertscore_f1'])
                st.caption(f"BERTScore F1: {eval_scores['bertscore_f1']:.3f}")
            
            if 'preservation_score' in eval_scores:
                st.progress(eval_scores['preservation_score'])
                st.caption(f"Preservation Score: {eval_scores['preservation_score']:.3f}")
        
        with col2:
            # Readability
            st.subheader("Readability Improvement")
            
            if 'readability_improvement' in eval_scores:
                improvement = eval_scores['readability_improvement']
                color = "green" if improvement > 0 else "red"
                st.markdown(
                    f"<h3 style='color: {color};'>"
                    f"{improvement:+.1f} points</h3>",
                    unsafe_allow_html=True
                )
                st.caption("Flesch Reading Ease improvement")
    
    # Key clauses table
    if 'extracted_sentences' in results:
        st.header("🔑 Key Extracted Sentences")
        
        extracted = results['extracted_sentences']
        
        for i, sentence in enumerate(extracted[:5], 1):
            st.markdown(f"**{i}.** {sentence}")
    
    # Download options
    st.header("💾 Export Options")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Generate Summary PDF
        try:
            from polisumm.pdf_generator import PDFGenerator
            pdf_gen = PDFGenerator()
            summary_pdf = pdf_gen.generate_summary_pdf(results)
            
            st.download_button(
                label="📥 Download Summary as PDF",
                data=summary_pdf,
                file_name="summary.pdf",
                mime="application/pdf"
            )
        except Exception as e:
            st.error(f"Error generating summary PDF: {str(e)}")
    
    with col2:
        # Generate Key Sentences PDF
        try:
            from polisumm.pdf_generator import PDFGenerator
            pdf_gen = PDFGenerator()
            key_sentences_pdf = pdf_gen.generate_key_sentences_pdf(results)
            
            st.download_button(
                label="📥 Download Key Sentences as PDF",
                data=key_sentences_pdf,
                file_name="key_sentences.pdf",
                mime="application/pdf"
            )
        except Exception as e:
            st.error(f"Error generating key sentences PDF: {str(e)}")


if __name__ == "__main__":
    main()

