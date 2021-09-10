""" 
This is a shared library for project-2.
"""

import re
import nltk
nltk.download(['punkt', 'wordnet'])
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

def extract_category_names(df):
    """Extracts multiple category names of target variables from input dataframe.

    Args:
    df: Input dataframe containing target variables

    Returns:
    Target variable (column) names in a string array
    """

    # Define the filtering columns not belong to target variable column names
    filter_cols = ['index', 'id', 'message', 'original', 'genre']
    
    # Find all columns which do not belong to the filtering columns
    category_names = []
    for col in df.columns.values:
        if col not in filter_cols:
            category_names.append(str(col))
            
    return category_names


def tokenize(text):
    """Text processing functions including normalization, word tokenization, lemmatization.

    Args:
    text : A sentence of text.

    Returns:
    Tokenized words in a string array.
    """
    
    # Punctuation removal
    text = re.sub(r"[^a-zA-Z0-9]", " ", text)
    
    # Word tokenization
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip() #Case normalization and lemmatization
        clean_tokens.append(clean_tok)

    return clean_tokens