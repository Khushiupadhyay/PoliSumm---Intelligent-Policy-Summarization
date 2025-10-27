# PoliSumm – Legal & Policy Simplifier

## 📘 Overview

PoliSumm is a robust NLP pipeline that automatically reads, analyzes, and summarizes legal or policy documents into clear, plain-language summaries. It supports multiple document formats and provides comprehensive evaluation metrics.

## 🚀 Features

- **Multiple Input Formats**: TXT, PDF, DOCX
- **Advanced NLP Pipeline**: Tokenization, POS tagging, NER, lemmatization
- **Transformer-Based Summarization**: Uses LongT5 for long documents
- **Evaluation Metrics**: ROUGE, BERTScore, SARI, FKGL
- **Interactive Dashboard**: Streamlit-based visualization
- **Multilingual Support**: English and optional Hindi translation

## 🛠️ Installation

1. Clone the repository or extract the project folder
2. Install Python 3.9+
3. Install dependencies:
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

## 💻 Usage

### Command Line
```bash
python main.py --input document.pdf --output summary.json
```

### Interactive Dashboard
```bash
streamlit run app.py
```

## 📁 Project Structure

```
nlpproject/
├── polisumm/
│   ├── __init__.py
│   ├── text_extractor.py      # Document parsing
│   ├── text_cleaner.py         # Text preprocessing
│   ├── nlp_pipeline.py         # NLP operations
│   ├── sentence_scorer.py      # Sentence ranking
│   ├── summarizer.py           # Summarization models
│   ├── evaluator.py            # Evaluation metrics
│   └── translator.py           # Translation (optional)
├── app.py                      # Streamlit dashboard
├── main.py                     # CLI interface
└── requirements.txt
```

## 🔧 Configuration

Edit `config.yaml` to customize:
- Model choices (LongT5, BART, etc.)
- Summarization ratios
- Evaluation thresholds
- Output formats

## ⚠️ Disclaimer

This summary is for informational purposes only and not legal advice.

