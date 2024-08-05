#app/routes.py

from flask import request, jsonify, render_template
from app import app, db, logger
from app.models import Customer, Recommendation, OptimalInterval

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/recommendations', methods=['GET'])
def recommendations():
    customer_id = request.args.get('customer_id')
    if not customer_id:
        return jsonify({"error": "No customer_id provided"}), 400

    recommendations = Recommendation.query.filter_by(customer_id=customer_id).all()
    optimal_interval = OptimalInterval.query.filter_by(customer_id=customer_id).first()

    if not recommendations:
        return jsonify({"error": "No recommendations found for customer"}), 404

    result = {
        "customer_id": customer_id,
        "recommendations": {
            "association_rules": [],
            "rnn": [],
            "collaborative_filtering": []
        },
        "probabilities": {
            "association_rules": None,
            "rnn": None,
            "collaborative_filtering": None
        },
        "optimal_days_between_purchases": optimal_interval.optimal_days_between_purchases if optimal_interval else None
    }

    for rec in recommendations:
        result["recommendations"][rec.algorithm].extend(rec.recommendations.split(","))
        result["probabilities"][rec.algorithm] = rec.probability

    return jsonify(result)
