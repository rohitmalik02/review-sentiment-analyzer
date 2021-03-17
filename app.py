from flask import Flask, request, render_template
import pickle
import numpy as np
import re
import nltk
import contractions
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer


app = Flask(__name__)
print(__name__)

nltk.download('stopwords')
all_stopwords = stopwords.words('english')


model = pickle.load(open('model.pkl', 'rb'))
vector = pickle.load(open('vector.pkl', 'rb'))

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    usr_review = [x for x in request.form.values()]
    new_review = str(usr_review)
    new_review = contractions.fix(new_review)
    new_review = re.sub('[^a-zA-Z]', ' ', new_review)
    new_review = new_review.lower()
    new_review = new_review.split()
    ps = PorterStemmer()
    all_stopwords = stopwords.words('english')
    all_stopwords.remove('not')
    new_review = [ps.stem(word) for word in new_review if not word in set(all_stopwords)]
    new_review = ' '.join(new_review)
    new_corpus = [new_review]
    X_test = vector.transform(new_corpus).toarray()
    analysis = model.predict(X_test)
    sentiment = 'positive' if (analysis[0] == 1) else 'negative'

    # return render_template('index.html', prediction_text='Salary should be: {:,.2f}$'.format(output))
    return render_template('index.html', prediction_text=f"This is a {sentiment} review")


if __name__ == '__main__':
    app.run(debug=True)
