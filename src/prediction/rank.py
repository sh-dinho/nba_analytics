import pandas as pd
import logging

# Set up logger
logger = logging.getLogger(__name__)

def generate_rankings(predictions: pd.DataFrame, bet_signal_threshold: float = 0.6, include_confidence: bool = True):
    """
    ============================================================
    Generate rankings and betting signals based on predicted win
    probabilities of NBA teams for upcoming games.

    This function ranks teams based on their predicted win
    probability and generates a betting signal (1 = bet, 0 = no bet)
    if the predicted win probability exceeds the specified threshold.
    It also optionally calculates a confidence score based on the
    predicted win probability.

    Args:
        predictions (pd.DataFrame): DataFrame containing the predicted
                                     win probabilities for each team.
                                     It must contain a 'predicted_win' column
                                     with the predicted win probability.
        bet_signal_threshold (float): The threshold above which a betting
                                      signal is generated. Defaults to 0.6.
        include_confidence (bool): If True, includes a 'confidence_score'
                                   column, which is the same as the
                                   predicted win probability. Defaults to True.

    Returns:
        pd.DataFrame: A DataFrame with the following columns:
            - 'rank': The rank of the team based on predicted win probability (ascending).
            - 'predicted_win': The predicted win probability for the home team.
            - 'bet_signal': 1 if a bet is recommended (predicted win probability > threshold), 0 otherwise.
            - 'confidence_score': (optional) The predicted win probability for confidence.
    ============================================================
    """
    if predictions.empty:
        logger.warning("No predictions available for ranking.")
        return pd.DataFrame()

    # Rank games by predicted win probability
    predictions["rank"] = predictions["predicted_win"].rank(method="min", ascending=False)

    # Generate betting signal based on the given threshold
    predictions["bet_signal"] = (predictions["predicted_win"] > bet_signal_threshold).astype(int)

    # Optionally, add a confidence score for each ranking
    if include_confidence:
        predictions["confidence_score"] = predictions["predicted_win"]

    # Log summary stats
    num_games = len(predictions)
    num_bets = predictions["bet_signal"].sum()
    logger.info(
        f"Rankings generated for {num_games} games, {num_bets} betting signals created "
        f"with bet signal threshold {bet_signal_threshold}."
    )

    # Sort by rank and reset index to get the top-ranked teams at the top
    rankings = predictions.sort_values("rank", ascending=True).reset_index(drop=True)

    return rankings

# Example usage:
if __name__ == "__main__":
    # Sample prediction DataFrame
    sample_predictions = pd.DataFrame({
        "team": ["Team A", "Team B", "Team C", "Team D"],
        "predicted_win": [0.8, 0.55, 0.9, 0.4]
    })

    rankings = generate_rankings(sample_predictions, bet_signal_threshold=0.6)
    print(rankings)
