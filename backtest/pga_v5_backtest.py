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
            'winner': 0.6,
            'top5': 0.5,
            'top10': 0.8,
            'top20': 0.4
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

def backtest_weights() -> Tuple[Dict, float]:
    """
    Test different weight combinations and return the best performing one
    """
    tournaments = ["The_Sentry", "Sony_Open_in_Hawaii"]
    weight_combinations = generate_weight_combinations()
    
    best_score = 0
    best_weights = None
    
    print(f"Testing {len(weight_combinations)} weight combinations...")
    
    for i, weights in enumerate(weight_combinations):
        total_score = 0
        print(f"\nTesting combination {i+1}/{len(weight_combinations)}")
        print(f"Weights: {weights['components']}")
        
        for tournament in tournaments:
            # Run the main function with these weights and get lineups directly
            lineups_df = pga_main(tournament, num_lineups=20, weights=weights)
            
            # Evaluate the lineup performance
            score = evaluate_lineup_performance(tournament, lineups_df)
            total_score += score
            print(f"{tournament} Score: {score:.2f}")
        
        avg_score = total_score / len(tournaments)
        print(f"Average Score: {avg_score:.2f}")
        
        if avg_score > best_score:
            best_score = avg_score
            best_weights = weights
            print("New best weights found!")
    
    return best_weights, best_score

if __name__ == "__main__":
    best_weights, best_score = backtest_weights()
    print("\nBacktesting Complete!")
    print("=" * 50)
    print("Best Performing Weights:")
    print(f"Components: {best_weights['components']}")
    print(f"Average Score: {best_score:.2f}")
