from flask import Flask, render_template, request
import pickle
import re
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import sklearn
import os



stopwords_set = set(stopwords.words('english'))
emoticon_pattern = re.compile('(?::|;|=)(?:-)?(?:\)|\(|D|P)')

# Load the sentiment analysis model and TF-IDF vectorizer
file_path = 'C:/Users/reddy/OneDrive/Desktop/projectK/Sentiment Analysis/clf.pkl'

if os.path.exists(file_path):
    with open(file_path, 'rb') as f:
        clf = pickle.load(f)
else:
    raise FileNotFoundError(f"File not found: {file_path}")
    # Your code here

file_path = 'C:/Users/reddy/OneDrive/Desktop/projectK/Sentiment Analysis/tfidf.pkl'

if os.path.exists(file_path):
    with open(file_path, 'rb') as f:
        tfidf = pickle.load(f)
else:
    raise FileNotFoundError(f"File not found: {file_path}")


def preprocessing(text):
    text = re.sub('<[^>]*>', '', text)
    emojis = emoticon_pattern.findall(text)
    text = re.sub('[\W+]', ' ', text.lower()) + ' '.join(emojis).replace('-', '')
    prter = PorterStemmer()
    text = [prter.stem(word) for word in text.split() if word not in stopwords_set]

    return " ".join(text)


app = Flask(__name__)

@app.route('/', methods=['GET','POST'])
def analyze_sentiment():
    if request.method == 'POST':
        comment = request.form.get('comment')
        # Preprocess the comment
        preprocessed_comment = preprocessing(comment)
        # Transform the preprocessed comment into a feature vector
        comment_vector = tfidf.transform([preprocessed_comment])
        # Predict the sentiment
        sentiment = clf.predict(comment_vector)[0]
        return render_template('index.html', sentiment=sentiment)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)