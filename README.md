# Academic Abstract Classification System

## Overview
This system automatically classifies academic abstracts into multiple categories based on their content. It uses Natural Language Processing (NLP) and machine learning techniques to analyze titles and abstracts of academic works in Portuguese, assigning them to appropriate categories such as Science Area, Educational Modality, School Level, and Focus.

## Features
- Pre-processes text data in Portuguese (tokenization, stopword removal, stemming)
- Trains multi-label classification models for multiple categories
- Processes batch files containing academic abstracts
- Generates detailed analysis and classification reports
- Provides comprehensive evaluation metrics

## Requirements
- Python 3.6 or higher
- Required libraries:
  - pandas
  - numpy
  - scikit-learn
  - nltk
  - re
  - pickle
  - Google Colab (for file uploads/downloads)

## Installation
```python
# Install required packages
pip install pandas numpy scikit-learn nltk

# Download NLTK resources for Portuguese
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('rslp')
```

## Usage

### Training Models
1. Run the system in Google Colab or other Python environment
2. Upload a training dataset when prompted
3. The system will:
   - Pre-process the text data
   - Extract features using TF-IDF
   - Train multi-label classification models
   - Save trained models for future use

### Processing New Abstracts
1. Upload an Excel file containing titles and abstracts to be classified
2. Specify the column names for titles and abstracts
3. The system will:
   - Classify each abstract into multiple categories
   - Generate a new Excel file with predictions
   - Provide a summary of classification results

## Data Format
### Training Data
The training file should be a CSV or Excel file with the following columns:
- `título`: Title of the academic work
- `resumo`: Abstract of the academic work
- `Área de Ciências`: Science areas (semicolon-separated for multiple values)
- `Modalidade Educacional`: Educational modalities (semicolon-separated)
- `Nível Escolar`: School levels (semicolon-separated)
- `Foco`: Focus areas (semicolon-separated)

### Input Data for Classification
The input file should be an Excel file containing at minimum:
- A column for the title of each work
- A column for the abstract of each work

## Technical Details
- Text preprocessing: lowercase conversion, special character removal, tokenization, stopword removal, and stemming
- Feature extraction: TF-IDF vectorization with up to 5000 features
- Classification: Multi-output Logistic Regression with fallback to DummyClassifier for categories with insufficient data
- Evaluation: Micro and Macro F1-scores

## Output
The system produces an Excel file with the original data plus additional columns:
- `Previsto: Área de Ciências`: Predicted science areas
- `Previsto: Modalidade Educacional`: Predicted educational modalities
- `Previsto: Nível Escolar`: Predicted school levels
- `Previsto: Foco`: Predicted focus areas

## Performance
The system reports performance metrics including:
- Processing speed (records per second)
- Estimated time remaining
- Classification distribution summary

## Files Generated
- `classificador_resumos_vectorizer.pkl`: TF-IDF vectorizer
- `classificador_resumos_área_de_ciências.pkl`: Science area classifier
- `classificador_resumos_modalidade_educacional.pkl`: Educational modality classifier
- `classificador_resumos_nível_escolar.pkl`: School level classifier
- `classificador_resumos_foco.pkl`: Focus classifier
- `binarizer_*.pkl`: Label binarizers for each category
- `resultados_classificacao.xlsx`: Final classification results
