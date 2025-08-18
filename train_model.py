import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import RegexpStemmer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score


df=pd.read_csv('D:\\Desktop\\Gen AI 2.0\\text.csv')
print(df.head())

print(df.columns)  # should show ['text', 'label']
print(df['label'].value_counts()) 

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')

stop_words = set(stopwords.words("english"))
negations = {"no", "not", "nor", "don't", "didn't", "won't", "can't", "shan't", "isn't", 
             "aren't", "couldn't", "wouldn't", "shouldn't", "wasn't", "weren't"}
filtered_stopwords = stop_words - negations

rs = RegexpStemmer('ed$|ing$|s$|es$|er$|ly$|ness$|ment$|tion$|able$|al$|ful$')

def clean_text_nltk(text):
    text = text.lower()                          # lowercase
    tokens = word_tokenize(text)                 # tokenize
    words = [word for word in tokens if word.isalpha()]  # remove punctuations and numbers
    words = [word for word in words if word not in filtered_stopwords]  # remove stopwords
    stemmed = [rs.stem(word) for word in words]  # apply custom stemming
    return ' '.join(stemmed)

df['text'] = df['text'].apply(clean_text_nltk)

# Show cleaned data
print(df.head())

X = df['text']   
y = df['label'] 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

model = MultinomialNB()
model.fit(X_train_vec, y_train)

y_pred = model.predict(X_test_vec)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))


# from nrclex import NRCLex

# text_object = NRCLex("I am feeling very happy and excited today!")
# print(text_object.raw_emotion_scores)
# print(text_object.top_emotions)





