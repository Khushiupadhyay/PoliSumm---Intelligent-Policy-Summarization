"""
PDF Generation Module
Generates PDF reports for summaries and key extracted sentences
"""

from typing import Dict, List
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
import io
import os


class PDFGenerator:
    """Generate PDF reports for PoliSumm results"""
    
    def __init__(self):
        """Initialize the PDF generator"""
        self.styles = getSampleStyleSheet()
        self._setup_custom_styles()
    
    def _setup_custom_styles(self):
        """Setup custom paragraph styles"""
        # Title style
        self.title_style = ParagraphStyle(
            'CustomTitle',
            parent=self.styles['Heading1'],
            fontSize=18,
            spaceAfter=30,
            alignment=TA_CENTER,
            textColor=colors.darkblue
        )
        
        # Subtitle style
        self.subtitle_style = ParagraphStyle(
            'CustomSubtitle',
            parent=self.styles['Heading2'],
            fontSize=14,
            spaceAfter=20,
            textColor=colors.darkblue
        )
        
        # Bullet point style
        self.bullet_style = ParagraphStyle(
            'BulletPoint',
            parent=self.styles['Normal'],
            fontSize=11,
            spaceAfter=12,
            leftIndent=20,
            bulletIndent=10,
            bulletText='•'
        )
        
        # Key sentence style
        self.key_sentence_style = ParagraphStyle(
            'KeySentence',
            parent=self.styles['Normal'],
            fontSize=11,
            spaceAfter=15,
            leftIndent=20,
            alignment=TA_JUSTIFY
        )
        
        # Metadata style
        self.metadata_style = ParagraphStyle(
            'Metadata',
            parent=self.styles['Normal'],
            fontSize=10,
            spaceAfter=8,
            textColor=colors.grey
        )
    
    def generate_summary_pdf(self, results: Dict) -> bytes:
        """
        Generate PDF for summary
        
        Args:
            results: Processing results dictionary
            
        Returns:
            PDF content as bytes
        """
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=72, leftMargin=72, 
                               topMargin=72, bottomMargin=18)
        
        story = []
        
        # Title
        story.append(Paragraph("PoliSumm Summary Report", self.title_style))
        story.append(Spacer(1, 20))
        
        # Document metadata
        metadata = results.get('metadata', {})
        if metadata:
            story.append(Paragraph("Document Information", self.subtitle_style))
            story.append(Paragraph(f"<b>Title:</b> {metadata.get('title', 'N/A')}", self.metadata_style))
            story.append(Paragraph(f"<b>Author:</b> {metadata.get('author', 'N/A')}", self.metadata_style))
            story.append(Paragraph(f"<b>Pages:</b> {metadata.get('pages', 'N/A')}", self.metadata_style))
            story.append(Spacer(1, 20))
        
        # Summary section
        story.append(Paragraph("Document Summary", self.subtitle_style))
        
        # Format summary with bullet points
        summary = results.get('summary', '')
        summary_lines = summary.split('\n')
        
        for line in summary_lines:
            line = line.strip()
            if line:
                if line.startswith(('•', '-', '*', '·')):
                    # Remove bullet and add proper formatting
                    clean_line = line[1:].strip()
                    story.append(Paragraph(f"• {clean_line}", self.bullet_style))
                else:
                    story.append(Paragraph(line, self.bullet_style))
        
        story.append(Spacer(1, 20))
        
        # Statistics
        stats = results.get('statistics', {})
        if stats:
            story.append(Paragraph("Document Statistics", self.subtitle_style))
            story.append(Paragraph(f"• Total Sentences: {stats.get('total_sentences', 'N/A')}", self.bullet_style))
            story.append(Paragraph(f"• Total Words: {stats.get('total_words', 'N/A')}", self.bullet_style))
            story.append(Paragraph(f"• Average Sentence Length: {stats.get('avg_sentence_length', 0):.1f} words", self.bullet_style))
            story.append(Spacer(1, 20))
        
        # Evaluation metrics
        eval_scores = results.get('evaluation', {})
        if eval_scores:
            story.append(Paragraph("Evaluation Metrics", self.subtitle_style))
            if 'preservation_score' in eval_scores:
                story.append(Paragraph(f"• Preservation Score: {eval_scores['preservation_score']:.3f}", self.bullet_style))
            if 'readability_improvement' in eval_scores:
                story.append(Paragraph(f"• Readability Improvement: {eval_scores['readability_improvement']:+.2f} points", self.bullet_style))
            story.append(Spacer(1, 20))
        
        # Disclaimer
        story.append(Paragraph("⚠️ This summary is for informational purposes only and not legal advice.", 
                              self.metadata_style))
        
        # Build PDF
        doc.build(story)
        buffer.seek(0)
        return buffer.getvalue()
    
    def generate_key_sentences_pdf(self, results: Dict) -> bytes:
        """
        Generate PDF for key extracted sentences
        
        Args:
            results: Processing results dictionary
            
        Returns:
            PDF content as bytes
        """
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=72, leftMargin=72, 
                               topMargin=72, bottomMargin=18)
        
        story = []
        
        # Title
        story.append(Paragraph("Key Extracted Sentences", self.title_style))
        story.append(Spacer(1, 20))
        
        # Document metadata
        metadata = results.get('metadata', {})
        if metadata:
            story.append(Paragraph("Document Information", self.subtitle_style))
            story.append(Paragraph(f"<b>Title:</b> {metadata.get('title', 'N/A')}", self.metadata_style))
            story.append(Paragraph(f"<b>Author:</b> {metadata.get('author', 'N/A')}", self.metadata_style))
            story.append(Paragraph(f"<b>Pages:</b> {metadata.get('pages', 'N/A')}", self.metadata_style))
            story.append(Spacer(1, 20))
        
        # Key sentences section
        story.append(Paragraph("Key Extracted Sentences", self.subtitle_style))
        
        extracted_sentences = results.get('extracted_sentences', [])
        if extracted_sentences:
            for i, sentence in enumerate(extracted_sentences, 1):
                story.append(Paragraph(f"{i}. {sentence}", self.key_sentence_style))
        else:
            story.append(Paragraph("No key sentences extracted.", self.bullet_style))
        
        story.append(Spacer(1, 20))
        
        # Statistics
        stats = results.get('statistics', {})
        if stats:
            story.append(Paragraph("Document Statistics", self.subtitle_style))
            story.append(Paragraph(f"• Total Sentences: {stats.get('total_sentences', 'N/A')}", self.bullet_style))
            story.append(Paragraph(f"• Total Words: {stats.get('total_words', 'N/A')}", self.bullet_style))
            story.append(Paragraph(f"• Extracted Sentences: {len(extracted_sentences)}", self.bullet_style))
            story.append(Spacer(1, 20))
        
        # Disclaimer
        story.append(Paragraph("⚠️ These extracted sentences are for informational purposes only and not legal advice.", 
                              self.metadata_style))
        
        # Build PDF
        doc.build(story)
        buffer.seek(0)
        return buffer.getvalue()
    
    def save_pdf(self, pdf_content: bytes, file_path: str):
        """
        Save PDF content to file
        
        Args:
            pdf_content: PDF content as bytes
            file_path: Path to save the PDF
        """
        with open(file_path, 'wb') as f:
            f.write(pdf_content)
        print(f"PDF saved to {file_path}")
