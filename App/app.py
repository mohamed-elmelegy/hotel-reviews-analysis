from cmath import e
import flask
from flask import Flask
from flask import request
from flask import abort, jsonify
import pandas as pd


app = Flask(__name__)

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
            'p_neg_mean': result.p_neg_mean
        }

        return flask.render_template('sentiment.html', data=result)
    except:
        abort(500)

@app.route("/indexing")
def elastic_search_index():
    hotel = request.args.get('hotel')
    return hotel


if __name__ == '__main__':
    app.run(debug=True)