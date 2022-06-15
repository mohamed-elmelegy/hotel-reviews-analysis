import flask
from flask import Flask
from textblob import TextBlob
from textblob.sentiments import NaiveBayesAnalyzer
import pandas as pd
import json


app = Flask(__name__)

@app.route("/")
def home():
    return flask.render_template('home.html')


@app.route("/sentiment")
def sentiment_analysis():
    hotel_reviews = pd.read_csv('../Storage/Data/reviews.csv')
    total = {'p_pos':0, 'p_neg':0}
    data_len = 5
    for sentce in hotel_reviews.lemmatized.values[:data_len]:
        blob = TextBlob(sentce, analyzer=NaiveBayesAnalyzer()).sentiment
        total['p_pos'] += blob.p_pos
        total['p_neg'] += blob.p_neg
    total['p_pos'] /= data_len
    total['p_neg'] /= data_len
    return flask.render_template('sentiment.html', data=total)


@app.route("/indexing")
def elastic_search_index():
    my_dictionary = {'a':'b', 'foo':'baz'}
    return flask.render_template('indexing.html', data=my_dictionary)


if __name__ == '__main__':
    app.run(debug=True)