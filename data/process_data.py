import sys
import pandas as pd
from sqlalchemy import create_engine



def load_data(messages_filepath, categories_filepath):
    df_messages = pd.read_csv(messages_filepath)
    df_categories = pd.read_csv(categories_filepath)

    # dataframe includes both messages and categories
    df = df_messages.merge(df_categories, how='inner', on='id')
    
    return df


def clean_data(df):
    categories = df['categories'].str.split(';', expand=True)
    
    rows = categories.iloc[0,:]
    category_colNames = rows.str.split('-').str[0]
    categories.columns = category_colNames
    
    for column in categories.columns:
        categories[column] = categories[column].str.split('-').str[1]
        categories[column] = categories[column].astype('int')
     
    df.drop('categories', axis=1, inplace=True)
    df = pd.concat([df, categories], axis=1)

    df['related'] = df['related'].apply(lambda x: 1 if x==2 else x)
    df.drop('child_alone', axis=1, inplace=True)
    df.drop_duplicates(keep='first', inplace=True)
    
    return df
    


def save_data(df):
    engine = create_engine('sqlite:///DisasterResponse.db')
    df.to_sql('DisasterResponse', engine, index=False, if_exists='replace')  


def main():
    if len(sys.argv) == 3:
        messages_filepath, categories_filepath= sys.argv[1:]
        df = load_data(messages_filepath, categories_filepath)
        df = clean_data(df)
        save_data(df)
        print('Clean data successfully!')
    else:
        print('Unable to clean data. Please provide filepath in correct order')

if __name__ == '__main__':
    main()

