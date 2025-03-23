import streamlit as st 
import numpy as np
import re
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load data
news_df = pd.read_csv('train.csv')

# Print column names to verify the presence of 'author'
print("Available columns:", news_df.columns)

# Handle missing 'author' column
if 'author' in news_df.columns:
    news_df['author'] = news_df['author'].fillna('')  
else:
    news_df['author'] = ''  # If 'author' is missing, create an empty column

# Ensure 'title' column exists
if 'title' in news_df.columns:
    news_df['title'] = news_df['title'].fillna('')
else:
    raise KeyError("The 'title' column is missing from train.csv. Please check the dataset.")

news_df['content'] = news_df['author'] + ' ' + news_df['title']

# Ensure 'label' column exists before dropping
if 'label' not in news_df.columns:
    raise KeyError("The 'label' column is missing from train.csv. Please check the dataset.")

X = news_df.drop('label', axis=1)
y = news_df['label']

# Define stemming function
ps = PorterStemmer()
def stemming(content):
    stemmed_content = re.sub('[^a-zA-Z]', ' ', content)
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()
    stemmed_content = [ps.stem(word) for word in stemmed_content if word not in stopwords.words('english')]
    return ' '.join(stemmed_content)

# Apply stemming function to content column
news_df['content'] = news_df['content'].apply(stemming)

# Vectorize data
X = news_df['content'].values
y = news_df['label'].values
vector = TfidfVectorizer()
vector.fit(X)
X = vector.transform(X)

# Split data into train and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=2)

# Fit logistic regression model
model = LogisticRegression()
model.fit(X_train, Y_train)

# Website
st.title('Fake News Detector')
input_text = st.text_input('Enter news Article')

def prediction(input_text):
    input_data = vector.transform([input_text])
    prediction = model.predict(input_data)
    return prediction[0]

if input_text:
    pred = prediction(input_text)
    st.write('The News is Fake' if pred == 1 else 'The News Is Real')
