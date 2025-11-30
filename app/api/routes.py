from flask import Blueprint, jsonify, request
from app.api.nba_api import fetch_nba_games, clean_data
from app.database import store_games, fetch_games
from app.models.xgboost_model import predict_game, train_xgb_model

bp = Blueprint('api', __name__, url_prefix='/api')

@bp.route('/fetch_games', methods=['GET'])
def fetch_and_store_games():
    season = request.args.get('season', default=2025, type=int)
    games_df = fetch_nba_games(season)
    if not games_df.empty:
        cleaned_df = clean_data(games_df)
        store_games(cleaned_df)
        return jsonify({"message": f"Games for season {season} fetched and stored successfully."}), 200
    return jsonify({"message": f"No games found for season {season}."}), 404

@bp.route('/games', methods=['GET'])
def get_games():
    games_df = fetch_games()
    if games_df.empty:
        return jsonify({"message": "No games available."}), 404
    return jsonify(games_df.to_dict(orient='records'))

@bp.route('/predict', methods=['POST'])
def predict_matchup():
    data = request.get_json()
    home_team = data.get('home_team')
    visitor_team = data.get('visitor_team')

    if not home_team or not visitor_team:
        return jsonify({"error": "home_team and visitor_team are required"}), 400

    prediction = predict_game(home_team, visitor_team)
    return jsonify({
        "home_team": home_team,
        "visitor_team": visitor_team,
        "home_team_win_probability": prediction,
        "away_team_win_probability": 1 - prediction
    }), 200

@bp.route('/retrain', methods=['POST'])
def retrain_model():
    model = train_xgb_model()
    if model:
        return jsonify({"message": "Model retrained successfully!"}), 200
    return jsonify({"message": "Model retraining failed!"}), 500

