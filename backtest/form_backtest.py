import sys
import os
from pathlib import Path
import itertools
import pandas as pd
from typing import Dict, List, Tuple

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from utils import TOURNAMENT_LIST_2025, fix_names
from pga_v5 import main as pga_main

def generate_form_weight_combinations() -> List[Dict]:
    """
    Generate different form weight combinations to test.
    Each combination must sum to 1.0
    """
    # Test weights in 0.1 increments
    weights = [round(x/10, 1) for x in range(11)]  # [0.0, 0.1, ..., 1.0]
    
    valid_combinations = []
    for current_weight in weights:
        long_weight = round(1 - current_weight, 1)  # Ensure weights sum to 1.0
        weights_dict = {
            'odds': {  # Use default odds weights
                'winner': 0.35,
                'top5': 0.15,
                'top10': 0.2,
                'top20': 0.3
            },
            'form': {
                'current': current_weight,
                'long': long_weight
            },
            'components': {  # Use default component weights
                'odds': 0.25,
                'fit': 0.25,
                'history': 0.25,
                'form': 0.25
            }
        }
        valid_combinations.append(weights_dict)
    
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
    results_df['Name'] = results_df['Name'].apply(fix_names)
    
    # Calculate score based on actual DK points
    total_score = 0
    for _, lineup in lineups_df.iterrows():
        lineup_score = 0
        for i in range(1, 7):
            player = fix_names(lineup[f'G{i}'].split(' (')[0])
            player_result = results_df[results_df['Name'] == player]
            if not player_result.empty:
                dk_score = player_result['DK Score'].iloc[0]
                if pd.notna(dk_score):
                    lineup_score += dk_score
        total_score += lineup_score
    
    return total_score / len(lineups_df)

def backtest_form_weights() -> Tuple[List[Dict], List[float]]:
    """
    Test different form weight combinations and return the top 3 best performing ones.
    Saves progress to CSV and can resume from previous runs.
    """
    tournaments = ["The_Sentry", "Sony_Open_in_Hawaii", "The_American_Express", 
                  "Farmers_Insurance_Open", "AT&T_Pebble_Beach_Pro-Am", "WM_Phoenix_Open"]
    weight_combinations = generate_form_weight_combinations()
    results_file = "backtest/form_backtest_results.csv"
    
    # Initialize or load existing results
    if os.path.exists(results_file):
        results_df = pd.read_csv(results_file)
        results_df['weights'] = results_df['weights'].apply(eval)
    else:
        results_df = pd.DataFrame(columns=['tournament', 'weights', 'score'])
    
    top_scores = [0, 0, 0]
    top_weights = [None, None, None]
    
    print(f"Testing {len(weight_combinations)} form weight combinations...")
    
    for i, weights in enumerate(weight_combinations):
        print(f"\nTesting combination {i+1}/{len(weight_combinations)}")
        print(f"Form weights - Current: {weights['form']['current']}, Long: {weights['form']['long']}")
        
        total_score = 0
        tournaments_tested = 0
        
        for tournament in tournaments:
            # Check if we already have results for this combination and tournament
            existing_result = results_df[
                (results_df['tournament'] == tournament) & 
                (results_df['weights'].apply(lambda x: 
                    x['form']['current'] == weights['form']['current'] and 
                    x['form']['long'] == weights['form']['long'])
                )
            ]
            
            if not existing_result.empty:
                score = existing_result['score'].iloc[0]
                print(f"{tournament} Score: {score:.2f} (loaded from cache)")
            else:
                # Run the main function with these weights and get lineups
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
            
            total_score += score
            tournaments_tested += 1
        
        avg_score = total_score / tournaments_tested
        print(f"Average Score: {avg_score:.2f}")
        
        # Update top 3 if necessary
        for j in range(3):
            if avg_score > top_scores[j]:
                # Shift lower scores down
                top_scores[j+1:] = top_scores[j:-1]
                top_weights[j+1:] = top_weights[j:-1]
                # Insert new score
                top_scores[j] = avg_score
                top_weights[j] = weights
                break
    
    return top_weights, top_scores

if __name__ == "__main__":
    best_weights, best_scores = backtest_form_weights()
    print("\nBacktesting Complete!")
    print("=" * 50)
    print("Top 3 Performing Form Weight Combinations:")
    for i in range(3):
        if best_weights[i]:
            print(f"\n{i+1}. Average Score: {best_scores[i]:.2f}")
            print(f"   Current Form Weight: {best_weights[i]['form']['current']}")
            print(f"   Long-term Form Weight: {best_weights[i]['form']['long']}")
