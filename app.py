from flask import request
from flask import Flask
from sentence_transformers import SentenceTransformer
from keras.models import load_model
import flask
import re
import nltk


nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

model = load_model('moviereview.h5')



def cleanText(text):
  STOPWORDS=nltk.corpus.stopwords.words('english')
  text=text.lower()
  text=re.sub(r'<[a-z]+(\s)*/*>'," ",text)
  text=re.sub(r'[,.?!]'," ",text)
  text=[word for word in text.split(' ') if word not in STOPWORDS]

  return ' '.join(text)


def processText(text):
    embeddings=SentenceTransformer('nli-roberta-base').encode(text)
    embeddings=embeddings.reshape(1,768)
    return embeddings

app = Flask(__name__)

# @app.route('/')
# def index():
#     return flask.render_template('index.html')

@app.route('/', methods=['POST','GET'])
def predict():
    if request.method=='POST':
        review = request.form['review']
        review_fin=processText(review)
        pred_prob=model.predict(review_fin)[0][0]
        prediction='Postive' if pred_prob>=0.5 else 'Negative'
        result={'prediction':prediction,'review':review,'pred_prob':pred_prob if prediction == 'Positive' else 1-pred_prob}
        return flask.render_template('index.html',result=result)
    else:
        return flask.render_template('index.html')

@app.route('/about')
def about():
    return flask.render_template('about.html')

if __name__=='__main__':
    app.run(debug=False)