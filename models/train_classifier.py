import pandas as pd 
import numpy as np
from sqlalchemy import create_engine 
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
nltk.download(['punkt', 'wordnet'])
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import sys
import re
import pickle

def load_data():
    # load cleaned data
    engine = create_engine('sqlite:///DisasterResponse.db')
    df = pd.read_sql_table('DisasterResponse',engine)
    X = df['message']
    y = df.loc[:,'related':]
    
    return X, y


def tokenize(text):
    # replace url text
    regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_url = re.findall(regex, text)
    for each in detected_url:
        text = text.replace(each, 'urlplaceholder')
    
    # produce clean token
    token = word_tokenize(text)
    lemma = WordNetLemmatizer()
    cleaned_token = []
    for n in token:
        clean_token = lemma.lemmatize(n).lower().strip()
        cleaned_token.append(clean_token)
        
    return cleaned_token


def build_model():
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', RandomForestClassifier())
    ])
    
    return pipeline

def evaluate_model(y_test, y_pred):
    labels = np.unique(y_pred)
    accuracy = (y_pred == y_test).mean()

    print('Labels: ', labels)
    print("Accuracy:", accuracy)
    
    df_ypred = pd.DataFrame(y_pred, columns = y_test.columns)
    for col in y_test.columns:
        print("Category: ", col)
        print(classification_report(y_test[col], df_ypred[col]))


def save_model(pipeline):
    pickle.dump(pipeline, open('models/classifier.pkl', 'wb'))


def main():
        X, y = load_data()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        
        model = build_model()
        model.fit(X_train, y_train)

        # display results
        y_pred = model.predict(X_test)
        evaluate_model(y_test, y_pred)

        save_model(model)


if __name__ == '__main__':
    main()
