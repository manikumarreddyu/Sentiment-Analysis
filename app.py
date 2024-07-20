import db
import json
from flask import Flask, request, render_template
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
file_path = 'clf.pkl'

if os.path.exists(file_path):
    with open(file_path, 'rb') as f:
        clf = pickle.load(f)
else:
    raise FileNotFoundError(f"File not found: {file_path}")
    # Your code here

file_path = 'tfidf.pkl'

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


@app.route('/', methods = ['GET', 'POST'])
def home():
    return render_template("signin.html")


@app.route('/signup', methods = ['GET', 'POST'])
def signup():
    return render_template("signup.html")



@app.route('/signin', methods = ['GET', 'POST'])
def signin():
    status, username = db.check_user()

    data = {
        "username": username,
        "status": status
    }

    return json.dumps(data)



@app.route('/register', methods = ['GET', 'POST'])
def register():
    status = db.insert_data()
    return json.dumps(status)

@app.route('/index', methods=['GET', 'POST'])
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
    app.run(debug = True)

# import db
# import json
# from flask import Flask, request, render_template, redirect, url_for

# app = Flask(__name__)

# @app.route('/', methods=['GET', 'POST'])
# def home():
#     return render_template("signin.html")

# @app.route('/signup', methods=['GET', 'POST'])
# def signup():
#     return render_template("signup.html")

# @app.route('/signin', methods=['GET', 'POST'])
# def signin():
#     if request.method == 'POST':
#         username = request.form['username']
#         password = request.form['password']
#         status, user = db.check_user(username, password)
        
#         if status:
#             return redirect(url_for('index'))
#         else:
#             return render_template("signin.html", error="Invalid credentials. Please try again.")
    
#     return render_template("signin.html")

# @app.route('/register', methods=['GET', 'POST'])
# def register():
#     if request.method == 'POST':
#         username = request.form['username']
#         password = request.form['password']
#         status = db.insert_data(username, password)
        
#         if status:
#             return redirect(url_for('signin'))
#         else:
#             return render_template("signup.html", error="Registration failed. Please try again.")
    
#     return render_template("signup.html")

# @app.route('/index', methods=['GET'])
# def index():
#     return render_template("index.html")

# if __name__ == '__main__':
#     app.run(debug=True)
