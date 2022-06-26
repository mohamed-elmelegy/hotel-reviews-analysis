from elasticsearch import Elasticsearch
import pandas as pd
from cmath import e
import flask
from flask import Flask
from flask import request
from flask import abort, jsonify
from ElasticsearchService import ElasticsearchService
from SentimentClassifier import SentimentClassifier


app = Flask(__name__)
es = ElasticsearchService()
ELASTICSEARCH_INDEX_NAME = 'demo-hotels-1'

@app.errorhandler(404)
def resource_not_found(e):
    return jsonify(error=str(e)), 404


@app.route("/")
def home():
    try:
        hotels = pd.read_pickle('../Storage/Data/reviews_agg_sentiment.pkl')
        hotels_names = hotels.index
        return flask.render_template('home.html', hotels=hotels_names)
    except:
        abort(500)

@app.route("/sentiment")
def sentiment_analysis():
    hotel = request.args.get('hotel')
    try:
        hotels = pd.read_pickle('../Storage/Data/reviews_agg_sentiment.pkl')
        result = hotels.loc[hotel]

        if(len(result) != 2):
            abort(400, f"Can't find sentiment for this hotel({hotel})")

        result = {
            'name': hotel,
            'classification': 'Positive' if result.p_pos_mean > result.p_neg_mean else 'Negative',
            'p_pos_mean': result.p_pos_mean,
            'p_neg_mean': result.p_neg_mean,
            'model': f'TextBlob Built-in Model'
        }

        return flask.render_template('sentiment.html', data=result)
    except:
        abort(500)

@app.route('/indexing')
def elastic_search_doc():
    hotel = request.args.get('hotel')
    return jsonify(es.get_doc(ELASTICSEARCH_INDEX_NAME, hotel))


@app.route('/indexing_all')
def elastic_search_index():
    result_size = request.args.get('size')
    return jsonify(es.get_head(ELASTICSEARCH_INDEX_NAME, result_size))

@app.route('/new_sentiment')
def new_sentiment_analysis():
    model_name = request.args.get('classifier')
    hotel = request.args.get('hotel')
    try:
        hotels = pd.read_pickle('../Storage/Data/preprocessed_reviews.pkl')
        reviews = hotels.groupby('name').get_group(hotel).lemmatized
        lens = reviews.shape[0]
        if(lens == 0):
            abort(400, f"Can't find sentiment for this hotel({hotel})")
        classifier = SentimentClassifier(model_name)
        sentiments = classifier.predict_sentiment(reviews)
        result = pd.DataFrame(sentiments, columns=['Positive_prob', 'Negative_prob']).mean()
        result = {
            'name': hotel,
            'classification': 'Positive' if result.Positive_prob > result.Negative_prob else 'Negative',
            'p_pos_mean': result.Positive_prob,
            'p_neg_mean': result.Negative_prob,
            'model': f'New Trained ({model_name})'
        }
        return flask.render_template('sentiment.html', data=result)
    except Exception as e:
        print(e)


if __name__ == '__main__':
    app.run(debug=True)