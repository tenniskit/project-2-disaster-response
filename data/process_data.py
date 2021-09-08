import sys

# import libraries
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """Loads the messages and categories datasets from CSV files and merge them. (Extract of ETL)

    Args:
    messages_filepath : File path of messages dataset which includes plain text messages.
    categories_filepath : File path of categories dataset which includes classified labels.

    Returns:
    The merged datasets (in Pandas's DataFrame) with messages and categories joined.
    """

    # load messages dataset
    messages = pd.read_csv(messages_filepath)

    # load categories dataset
    categories = pd.read_csv(categories_filepath)
    
    # merge datasets
    return messages.merge(categories, how='inner', on=['id'])


def clean_data(df):
    """Clean and transform data (Transsform of ETL)

    Args:
    df : The merged dataframe combined from messages and categories datasets.

    Returns:
    df: A cleaned and transformed dataframe.
    """
    
    # create a dataframe of the 36 individual category columns
    categories = df["categories"].str.split(";", expand=True)
    
    # select the first row of the categories dataframe
    row = categories[0:1]

    # use this row to extract a list of new column names for categories.
    # one way is to apply a lambda function that takes everything 
    # up to the second to last character of each string with slicing
    category_colnames = row.apply(lambda x : x.str.split("-").str.get(0)).values[0]
        
    # rename the columns of `categories`
    categories.columns = category_colnames

    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].apply(lambda s: s[-1:] )
        
        # convert column from string to numeric
        categories[column] = categories[column].apply(lambda s: int(s) )
        
    # drop the original categories column from `df`
    df.drop("categories", axis=1, inplace=True)
    
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1)
    
    # drop duplicates
    df.drop_duplicates(inplace=True)

    return df


def save_data(df, database_filename):
    """Save the cleaned and transformed dataframe into Sqlite DB. (Load of ETL)

    Args:
    df : A cleaned and transformed dataframe.
    database_filename : Target Sqlite database file name.

    Returns:
    None
    """
    
    # Create db engine and save dataframe to table
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('Message', engine, index=False)


def main():    
    if len(sys.argv) == 4:
        
        # Extract parmeters from commend line input
        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]
        
        # Load data from CSV files into DataFrame (E of ETL)
        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        # Clean and transform the data (T of ETL)
        print('Cleaning data...')
        df = clean_data(df)
        
        # Save the data (L of ETL)
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()