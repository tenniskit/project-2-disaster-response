
# Import common libraries
import pandas as pd
from sqlalchemy import create_engine


import re
import nltk
nltk.download(['punkt', 'wordnet'])
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

from sklearn.pipeline import Pipeline

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

import pickle
import sys


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
    

def load_data(database_filepath):
    """Load the dataframe from SQLite database.

    Args:
    database_filepath : Target Sqlite database file path.

    Returns:
    X: Independent variable in numpy array
    Y: Dependent variables in numpy array
    category_names:
    """
    
    # Load data from database
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('Message', engine)  
    
    # Extract target variable names from the loaded data
    category_names = extract_category_names(df)

    # Extract independent variable
    X = df["message"].values
    
    # Extract dependent variables (target variable)
    Y = df[category_names].values

    # Return all extracted values
    return X, Y, category_names




def build_model():
    """Build a model using Pipeline, GridSearchCV and a set of hyperparameters for grid search.

    Args:
    None.
    
    Returns:
    A model with GridSearchCV.
    """
    
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    
    parameters = {
        'clf__estimator__n_estimators': [99, 100] # Lowers the range of hyperparameters in order to prevent it from taking too long to run.
    }

    return GridSearchCV(pipeline, param_grid=parameters)



def evaluate_model(model, X_test, Y_test, category_names):
    """Build a model using Pipeline and GridSearchCV

    Args:
    None.
    
    Returns:
    A model with GridSearchCV.
    """
    
    print("\nBest Parameters:", model.best_params_)
    
    # predict on test data
    y_pred = model.predict(X_test)

    for i in range(len(category_names)):
    
        accuracy = accuracy_score(Y_test[:, i] , y_pred[:, i])
        precision =  precision_score(Y_test[:, i] , y_pred[:, i], average="weighted")
        recall =  recall_score(Y_test[:, i] , y_pred[:, i], average="weighted")
        
        print(category_names[i])
        print("\t Accuracy1=%.4f"  %  accuracy + "\t  %% Precision=%.4f"  %  precision + "\t  %% Recall=%.4f"  %  recall + "\r\n")
        
        


def save_model(model, model_filepath):
    """Save the model to a physical file.

    Args:
    Model: A model
    model_filepath: Target file path.
    
    Returns:
    None.
    """
    
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()