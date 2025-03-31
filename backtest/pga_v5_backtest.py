import sys
import os
from pathlib import Path
import itertools
import pandas as pd
from typing import Dict, List, Tuple
import math

# Add parent directory to path (more robust way)
sys.path.append(str(Path(__file__).parent.parent))

from utils import TOURNAMENT_LIST_2025, fix_names
from pga_v5 import main as pga_main
from dk_find_best_finish import load_data, merge_data, optimize_lineup

def generate_weight_combinations() -> List[Dict]:
    """
    Generate different weight combinations to test.
    Each combination must sum to 1.0
    """
    # We'll test weights in 0.1 increments, including 0.0
    weights = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    
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

def get_tournament_optimal_score(tournament: str) -> float:
    """
    Calculate the optimal possible score for a tournament using perfect hindsight.
    Returns 0.0 if data isn't available.
    """
    try:
        dk_salaries, results = load_data(tournament)
        merged_data = merge_data(dk_salaries, results)
        optimal_lineup = optimize_lineup(merged_data)
        return optimal_lineup['TotalPoints']
    except (FileNotFoundError, Exception) as e:
        print(f"Warning: Couldn't calculate optimal score for {tournament}: {str(e)}")
        return 0.0

def evaluate_lineup_performance(tournament: str, lineups_df: pd.DataFrame, tournament_highlights: Dict, weights: Dict) -> Tuple[float, float, float, Dict]:
    """
    Calculate lineup performance metrics including success rate based on optimal score.
    Returns (average_dk_points, best_lineup_points, success_rate, updated_highlights)
    """
    # Get tournament ID from TOURNAMENT_LIST_2025
    tournament_id = TOURNAMENT_LIST_2025[tournament]['ID']
    results_file = f"past_results/2025/dk_points_id_{tournament_id}.csv"
    
    if not os.path.exists(results_file):
        print(f"Warning: No results file found for {tournament} (ID: {tournament_id})")
        return 0.0, 0.0, 0.0, tournament_highlights
    
    # Calculate optimal score and success threshold (70% of optimal)
    optimal_score = get_tournament_optimal_score(tournament)
    success_threshold = optimal_score * 0.70 if optimal_score > 0 else 0.0
    
    # Load actual tournament results
    results_df = pd.read_csv(results_file)
    results_df['Name'] = results_df['Name'].apply(fix_names)
    
    # Calculate scores and success rate
    total_score = 0
    best_lineup_score = 0
    success_score = 0
    
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
        best_lineup_score = max(best_lineup_score, lineup_score)
        
        # Calculate exponential success score if above threshold
        if optimal_score > 0 and lineup_score >= success_threshold:
            achievement_ratio = lineup_score / optimal_score
            # Normalize between 0.7 (70%) and 0.95 (95%) instead of 0.7 and 1.0
            normalized_ratio = min(1.0, (achievement_ratio - 0.7) / 0.25)  # 0.25 is (0.95 - 0.7)
            exponential_reward = (math.exp(normalized_ratio) - 1) / (math.e - 1)
            success_score += exponential_reward
    
    avg_score = total_score / len(lineups_df)
    
    if optimal_score > 0:
        print(f"Tournament metrics for {tournament}:")
        print(f"  Optimal score: {optimal_score:.2f}")
        print(f"  Success threshold (70%): {success_threshold:.2f}")
        print(f"  95% ceiling: {optimal_score * 0.95:.2f}")
        print(f"  Cumulative success score: {success_score:.4f}")

        '''Success Score   | Interpretation
            ----------------|------------------
            0.0 - 1.0      | Poor (few lineups above 70%)
            1.0 - 2.0      | Below Average
            2.0 - 3.0      | Average
            3.0 - 4.0      | Good
            4.0 - 5.0      | Excellent
            5.0+           | Exceptional'''
    
    # Update tournament highlights for all metrics
    if avg_score > tournament_highlights[tournament]['avg']['score']:
        tournament_highlights[tournament]['avg'] = {
            'score': avg_score,
            'weights': weights['components'].copy()
        }
    if best_lineup_score > tournament_highlights[tournament]['best']['score']:
        tournament_highlights[tournament]['best'] = {
            'score': best_lineup_score,
            'weights': weights['components'].copy()
        }
    if success_score > tournament_highlights[tournament]['success']['score']:
        tournament_highlights[tournament]['success'] = {
            'score': success_score,
            'weights': weights['components'].copy()
        }
    
    return avg_score, best_lineup_score, success_score, tournament_highlights

def backtest_weights() -> Tuple[List[Dict], List[float], List[float], List[float], Dict]:
    """
    Test different weight combinations and return the top performers based on
    multiple metrics including success rate.
    """
    tournaments = ["The_Sentry", "Sony_Open_in_Hawaii", "The_American_Express", "Farmers_Insurance_Open", "AT&T_Pebble_Beach_Pro-Am", "WM_Phoenix_Open", "Mexico_Open_at_VidantaWorld", "Cognizant_Classic_in_The_Palm_Beaches", "Arnold_Palmer_Invitational_presented_by_Mastercard", "THE_PLAYERS_Championship", "Valspar_Championship", "Texas_Children's_Houston_Open"]
    tournaments = tournaments[-5:]
    weight_combinations = generate_weight_combinations()
    results_file = "backtest/pga_v5_backtest_results.csv"
    
    # Initialize tournament highlights with proper initial values
    tournament_highlights = {
        t: {
            'avg': {'score': float('-inf'), 'weights': None},
            'best': {'score': float('-inf'), 'weights': None},
            'success': {'score': float('-inf'), 'weights': None}
        } for t in tournaments
    }
    
    # Initialize or load existing results
    if os.path.exists(results_file):
        results_df = pd.read_csv(results_file)
        results_df['weights'] = results_df['weights'].apply(eval)
    else:
        results_df = pd.DataFrame(columns=['tournament', 'weights', 'avg_score', 'best_score', 'success_score'])
    
    # Track scores and success rate
    tournament_scores = {t: {'avg': {}, 'best': {}, 'success': {}} for t in tournaments}
    
    print(f"Testing {len(weight_combinations)} weight combinations...")
    
    for i, weights in enumerate(weight_combinations):
        print(f"\nTesting combination {i+1}/{len(weight_combinations)}")
        print(f"Weights: {weights['components']}")
        
        for tournament in tournaments:
            existing_result = results_df[
                (results_df['tournament'] == tournament) & 
                (results_df['weights'].apply(lambda x: x['components']) == weights['components'])
            ]
            
            if not existing_result.empty:
                avg_score = existing_result['avg_score'].iloc[0]
                best_score = existing_result['best_score'].iloc[0]
                success_score = existing_result['success_score'].iloc[0]
                print(f"{tournament} - Avg: {avg_score:.2f}, Best: {best_score:.2f}, Success: {success_score:.2%} (cached)")
                
                # Update tournament highlights for cached results too
                if avg_score > tournament_highlights[tournament]['avg']['score']:
                    tournament_highlights[tournament]['avg'] = {
                        'score': avg_score,
                        'weights': weights['components'].copy()
                    }
                if best_score > tournament_highlights[tournament]['best']['score']:
                    tournament_highlights[tournament]['best'] = {
                        'score': best_score,
                        'weights': weights['components'].copy()
                    }
                if success_score > tournament_highlights[tournament]['success']['score']:
                    tournament_highlights[tournament]['success'] = {
                        'score': success_score,
                        'weights': weights['components'].copy()
                    }
            else:
                lineups_df = pga_main(tournament, num_lineups=20, weights=weights)
                avg_score, best_score, success_score, tournament_highlights = evaluate_lineup_performance(
                    tournament, lineups_df, tournament_highlights, weights
                )
                
                new_row = pd.DataFrame({
                    'tournament': [tournament],
                    'weights': [weights],
                    'avg_score': [avg_score],
                    'best_score': [best_score],
                    'success_score': [success_score]
                })
                results_df = pd.concat([results_df, new_row], ignore_index=True)
                results_df.to_csv(results_file, index=False)
                print(f"{tournament} - Avg: {avg_score:.2f}, Best: {best_score:.2f}, Success: {success_score:.2%} (new)")
            
            # Store scores and update highlights
            tournament_scores[tournament]['avg'][str(weights['components'])] = avg_score
            tournament_scores[tournament]['best'][str(weights['components'])] = best_score
            tournament_scores[tournament]['success'][str(weights['components'])] = success_score

    # Calculate normalized scores for both metrics
    weight_performances = {'avg': {}, 'best': {}, 'success': {}}
    weight_raw_averages = {'avg': {}, 'best': {}, 'success': {}}
    
    for weights in weight_combinations:
        weight_key = str(weights['components'])
        for metric in ['avg', 'best', 'success']:
            normalized_scores = []
            raw_scores = []
            
            for tournament in tournaments:
                scores = tournament_scores[tournament][metric]
                if not scores:
                    continue
                
                raw_scores.append(scores[weight_key])
                
                min_score = min(scores.values())
                max_score = max(scores.values())
                
                if max_score - min_score > 0:
                    normalized_score = (scores[weight_key] - min_score) / (max_score - min_score)
                    normalized_scores.append(normalized_score)
            
            if normalized_scores:
                # Calculate average instead of sum for normalized scores
                weight_performances[metric][weight_key] = sum(normalized_scores) / len(normalized_scores)
                weight_raw_averages[metric][weight_key] = sum(raw_scores) / len(raw_scores)
    
    # Get top 3 for both metrics
    top_weights = {'avg': [], 'best': [], 'success': []}
    top_scores = {'avg': [], 'best': [], 'success': []}
    top_raw_averages = {'avg': [], 'best': [], 'success': []}
    
    for metric in ['avg', 'best', 'success']:
        # Sort by raw averages for success rate, normalized scores for others
        if metric == 'success':
            sorted_weights = sorted(weight_raw_averages[metric].items(), key=lambda x: x[1], reverse=True)
        else:
            sorted_weights = sorted(weight_performances[metric].items(), key=lambda x: x[1], reverse=True)
            
        for weight_str, score in sorted_weights[:3]:
            for w in weight_combinations:
                if str(w['components']) == weight_str:
                    top_weights[metric].append(w)
                    top_scores[metric].append(score)
                    top_raw_averages[metric].append(weight_raw_averages[metric][weight_str])
                    break
    
    return (top_weights['avg'], top_scores['avg'], top_raw_averages['avg'], 
            top_weights['best'], top_scores['best'], top_raw_averages['best'],
            top_weights['success'], top_scores['success'], top_raw_averages['success'],
            tournament_highlights)

if __name__ == "__main__":
    avg_weights, avg_scores, avg_raw_scores, best_weights, best_scores, best_raw_scores, success_weights, success_scores, success_raw_scores, highlights = backtest_weights()
    print("\nBacktesting Complete!")
    print("=" * 50)
    print("Top 3 Performing Weights (By Average):")
    for i in range(3):
        print(f"\n{i+1}. Normalized Score: {avg_scores[i]:.2f}")
        print(f"   Average DK Points: {avg_raw_scores[i]:.2f}")
        print(f"   Components: {avg_weights[i]['components']}")
    
    print("\n" + "=" * 50)
    print("Top 3 Performing Weights (By Best Lineup):")
    for i in range(3):
        print(f"\n{i+1}. Normalized Score: {best_scores[i]:.2f}")
        print(f"   Best DK Points: {best_raw_scores[i]:.2f}")
        print(f"   Components: {best_weights[i]['components']}")

    print("\n" + "=" * 50)
    print("Top 3 Performing Weights (By Success Rate):")
    for i in range(3):
        # Convert cumulative score to average score per lineup (assuming 20 lineups)
        avg_success_score = success_raw_scores[i] / 20  # Divide by number of lineups
        print(f"\n{i+1}. Average Success Score: {avg_success_score:.3f}")
        print(f"   (Total Success Score: {success_raw_scores[i]:.3f})")
        print(f"   Components: {success_weights[i]['components']}")

        '''Avg Success Score | Interpretation
            -----------------|------------------
            0.00 - 0.05     | Poor
            0.05 - 0.10     | Below Average
            0.10 - 0.15     | Average
            0.15 - 0.20     | Good
            0.20 - 0.25     | Excellent
            0.25+           | Exceptional'''

    print("\n" + "=" * 50)
    print("Tournament-Specific Highlights:")
    for tournament in highlights:
        print(f"\n{tournament}:")
        print(f"  Best Average Score: {highlights[tournament]['avg']['score']:.2f}")
        print(f"  Best Average Weights: {highlights[tournament]['avg']['weights']}")
        print(f"  Best Single Lineup: {highlights[tournament]['best']['score']:.2f}")
        print(f"  Best Single Weights: {highlights[tournament]['best']['weights']}")
        print(f"  Best Success Score: {highlights[tournament]['success']['score']:.3f}")
        print(f"  Best Success Weights: {highlights[tournament]['success']['weights']}")
