import sys
import os
from pathlib import Path
import itertools
import pandas as pd
from typing import Dict, List, Tuple

# Add parent directory to path (more robust way)
sys.path.append(str(Path(__file__).parent.parent))

from utils import TOURNAMENT_LIST_2025, fix_names
from pga_v5 import main as pga_main

def generate_weight_combinations() -> List[Dict]:
    """
    Generate different weight combinations to test.
    Each combination must sum to 1.0
    """
    # We'll test weights in 0.1 increments
    weights = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    
    valid_combinations = []
    for w1, w2, w3, w4 in itertools.product(weights, repeat=4):
        if abs(w1 + w2 + w3 + w4 - 1.0) < 0.001:  # Sum must be 1.0
            valid_combinations.append({
                'components': {
                    'odds': w1,
                    'fit': w2,
                    'history': w3,
                    'form': w4
                }
            })
    
    # Add the default odds and form weights to each combination
    for combo in valid_combinations:
        combo['odds'] = {
            'winner': 0.35,
            'top5': 0.15,
            'top10': 0.2,
            'top20': 0.3
        }
        combo['form'] = {
            'current': 0.7,
            'long': 0.3
        }
    
    return valid_combinations

def evaluate_lineup_performance(tournament: str, lineups_df: pd.DataFrame) -> float:
    """
    Calculate the performance score of a lineup based on actual DraftKings scores.
    Returns the average DraftKings points per lineup.
    """
    # Get tournament ID from TOURNAMENT_LIST_2025
    tournament_id = TOURNAMENT_LIST_2025[tournament]['ID']
    results_file = f"past_results/2025/dk_points_id_{tournament_id}.csv"
    
    if not os.path.exists(results_file):
        print(f"Warning: No results file found for {tournament} (ID: {tournament_id})")
        return 0.0
    
    # Load actual tournament results
    results_df = pd.read_csv(results_file)
    
    # Clean player names in results
    results_df['Name'] = results_df['Name'].apply(fix_names)
    
    # Calculate score based on actual DK points
    total_score = 0
    for _, lineup in lineups_df.iterrows():
        lineup_score = 0
        for i in range(1, 7):  # For each golfer in lineup
            player = fix_names(lineup[f'G{i}'].split(' (')[0])  # Remove DK ID and clean name
            player_result = results_df[results_df['Name'] == player]
            if not player_result.empty:
                dk_score = player_result['DK Score'].iloc[0]
                if pd.notna(dk_score):  # Only add score if it's not NaN
                    lineup_score += dk_score
        total_score += lineup_score
    
    return total_score / len(lineups_df)  # Average DK points per lineup

def backtest_weights() -> Tuple[List[Dict], List[float], List[float]]:
    """
    Test different weight combinations and return the top 3 best performing ones.
    Scores are normalized within each tournament before averaging.
    """
    tournaments = ["The_Sentry", "Sony_Open_in_Hawaii", "The_American_Express", "Farmers_Insurance_Open", "AT&T_Pebble_Beach_Pro-Am", "WM_Phoenix_Open", "Mexico_Open_at_VidantaWorld"]
    weight_combinations = generate_weight_combinations()
    results_file = "backtest/pga_v5_backtest_results.csv"
    
    # Initialize or load existing results
    if os.path.exists(results_file):
        results_df = pd.read_csv(results_file)
        results_df['weights'] = results_df['weights'].apply(eval)
    else:
        results_df = pd.DataFrame(columns=['tournament', 'weights', 'score'])
    
    # Track scores by tournament and weight combination
    tournament_scores = {t: {} for t in tournaments}
    
    print(f"Testing {len(weight_combinations)} weight combinations...")
    
    for i, weights in enumerate(weight_combinations):
        print(f"\nTesting combination {i+1}/{len(weight_combinations)}")
        print(f"Weights: {weights['components']}")
        
        for tournament in tournaments:
            # Check if we already have results for this combination and tournament
            existing_result = results_df[
                (results_df['tournament'] == tournament) & 
                (results_df['weights'].apply(lambda x: x['components']) == weights['components'])
            ]
            
            if not existing_result.empty:
                score = existing_result['score'].iloc[0]
                print(f"{tournament} Score: {score:.2f} (loaded from cache)")
            else:
                # Run the main function with these weights and get lineups directly
                lineups_df = pga_main(tournament, num_lineups=20, weights=weights)
                
                # Evaluate the lineup performance
                score = evaluate_lineup_performance(tournament, lineups_df)
                
                # Save the result
                new_row = pd.DataFrame({
                    'tournament': [tournament],
                    'weights': [weights],
                    'score': [score]
                })
                results_df = pd.concat([results_df, new_row], ignore_index=True)
                results_df.to_csv(results_file, index=False)
                print(f"{tournament} Score: {score:.2f} (new)")
            
            # Store score for normalization
            tournament_scores[tournament][str(weights['components'])] = score
    
    # Calculate normalized scores and overall performance
    weight_performances = {}
    weight_raw_averages = {}  # New dictionary to track raw averages
    for weights in weight_combinations:
        weight_key = str(weights['components'])
        normalized_scores = []
        raw_scores = []  # New list to track raw scores
        
        for tournament in tournaments:
            scores = tournament_scores[tournament]
            if not scores:  # Skip if no scores for tournament
                continue
            
            # Store raw score
            raw_scores.append(scores[weight_key])
                
            # Get min and max scores for this tournament
            min_score = min(scores.values())
            max_score = max(scores.values())
            
            if max_score - min_score > 0:  # Avoid division by zero
                # Normalize score to 0-100 scale
                normalized_score = 100 * (scores[weight_key] - min_score) / (max_score - min_score)
                normalized_scores.append(normalized_score)
        
        if normalized_scores:  # Calculate averages if we have scores
            weight_performances[weight_key] = sum(normalized_scores) / len(normalized_scores)
            weight_raw_averages[weight_key] = sum(raw_scores) / len(raw_scores)  # Calculate raw average
    
    # Get top 3 performing weight combinations
    sorted_weights = sorted(weight_performances.items(), key=lambda x: x[1], reverse=True)
    top_weights = []
    top_scores = []
    top_raw_averages = []  # New list for raw averages
    
    for weight_str, score in sorted_weights[:3]:
        # Find the original weight dictionary
        for w in weight_combinations:
            if str(w['components']) == weight_str:
                top_weights.append(w)
                top_scores.append(score)
                top_raw_averages.append(weight_raw_averages[weight_str])  # Add raw average
                break
    
    return top_weights, top_scores, top_raw_averages  # Return raw averages

if __name__ == "__main__":
    best_weights, best_scores, avg_scores = backtest_weights()
    print("\nBacktesting Complete!")
    print("=" * 50)
    print("Top 3 Performing Weights:")
    for i in range(3):
        print(f"\n{i+1}. Normalized Score: {best_scores[i]:.2f}")
        print(f"   Average DK Points: {avg_scores[i]:.2f}")
        print(f"   Components: {best_weights[i]['components']}")
