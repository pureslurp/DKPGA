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

def get_prize_for_placement(placement: int) -> float:
    """Calculate prize money for a given placement."""
    if placement <= 1: return 5000.00
    elif placement <= 2: return 2500.00
    elif placement <= 3: return 1500.00
    elif placement <= 4: return 1000.00
    elif placement <= 5: return 750.00
    elif placement <= 7: return 600.00
    elif placement <= 10: return 500.00
    elif placement <= 13: return 400.00
    elif placement <= 16: return 300.00
    elif placement <= 19: return 250.00
    elif placement <= 22: return 200.00
    elif placement <= 25: return 150.00
    elif placement <= 30: return 100.00
    elif placement <= 35: return 75.00
    elif placement <= 45: return 60.00
    elif placement <= 55: return 50.00
    elif placement <= 70: return 40.00
    elif placement <= 85: return 30.00
    elif placement <= 105: return 25.00
    elif placement <= 135: return 20.00
    elif placement <= 200: return 15.00
    elif placement <= 400: return 10.00
    elif placement <= 870: return 8.00
    elif placement <= 1870: return 6.00
    elif placement <= 5028: return 5.00
    return 0.00

def estimate_placement(achievement_ratio: float, tournament: str = None) -> int:
    """
    Estimate placement using a power law distribution to create natural clustering
    at different achievement levels
    """
    PAID_ENTRIES = 5028  # Number of places that get paid
    
    if achievement_ratio >= 0.95:  # Exceptional performance
        return 1
    elif achievement_ratio < 0.65:  # Below min cash
        return PAID_ENTRIES + 1  # Out of money
        
    # Normalize ratio between min cash (0.65) and top score (0.95)
    normalized_ratio = (achievement_ratio - 0.65) / (0.95 - 0.65)
    
    # Apply power law transformation
    power = 4.4
    
    # Calculate placement using power law
    # Now using normalized_ratio^power directly instead of (1 - normalized_ratio^power)
    place = int((1 - normalized_ratio)**power * PAID_ENTRIES)
    place = max(1, min(PAID_ENTRIES, place))
    
    # Debug print for high scores
    if achievement_ratio * 524 > 425:
        print(f"\nHigh score detected:")
        print(f"  Raw score: {achievement_ratio * 524:.2f}")
        print(f"  Achievement ratio: {achievement_ratio:.3f}")
        print(f"  Estimated place: {place}")
        print(f"  Prize: ${get_prize_for_placement(place):.2f}")
    
    return place

def evaluate_lineup_performance(tournament: str, lineups_df: pd.DataFrame, tournament_highlights: Dict, weights: Dict) -> Tuple[float, float, float, Dict]:
    """
    Calculate lineup performance metrics including expected value based on optimal score.
    Returns (average_dk_points, best_lineup_points, expected_value, updated_highlights)
    """
    # Get tournament ID from TOURNAMENT_LIST_2025
    tournament_id = TOURNAMENT_LIST_2025[tournament]['ID']
    results_file = f"past_results/2025/dk_points_id_{tournament_id}.csv"
    thresholds_file = "backtest/tournament_thresholds.csv"
    
    if not os.path.exists(results_file):
        print(f"Warning: No results file found for {tournament} (ID: {tournament_id})")
        return 0.0, 0.0, 0.0, tournament_highlights
    
    # Calculate optimal score
    optimal_score = get_tournament_optimal_score(tournament)
    
    # Only calculate and save tournament thresholds if not already in file
    if optimal_score > 0:
        if os.path.exists(thresholds_file):
            existing_df = pd.read_csv(thresholds_file)
            if tournament not in existing_df['tournament'].values:
                thresholds_data = {
                    'tournament': [tournament],
                    'optimal_score': [optimal_score],
                    'min_cash_threshold': [optimal_score * 0.65],  # 65% threshold
                    'first_place_threshold': [optimal_score * 0.95],  # 95% threshold
                }
                thresholds_df = pd.DataFrame(thresholds_data)
                thresholds_df = pd.concat([existing_df, thresholds_df], ignore_index=True)
                thresholds_df.to_csv(thresholds_file, index=False)
        else:
            # Create new file if it doesn't exist
            thresholds_data = {
                'tournament': [tournament],
                'optimal_score': [optimal_score],
                'min_cash_threshold': [optimal_score * 0.65],
                'first_place_threshold': [optimal_score * 0.95],
            }
            pd.DataFrame(thresholds_data).to_csv(thresholds_file, index=False)
    
    # Calculate optimal score and success threshold (70% of optimal)
    success_threshold = optimal_score * 0.70 if optimal_score > 0 else 0.0
    
    # Load actual tournament results
    results_df = pd.read_csv(results_file)
    results_df['Name'] = results_df['Name'].apply(fix_names)
    
    # Calculate scores
    total_score = 0
    best_lineup_score = 0
    expected_value = 0.0
    
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
        
        # Calculate expected value if above min cash threshold
        if optimal_score > 0:
            achievement_ratio = lineup_score / optimal_score
            if achievement_ratio >= 0.65:
                estimated_place = estimate_placement(achievement_ratio, tournament)
                prize = get_prize_for_placement(estimated_place)
                expected_value += prize
    
    avg_score = total_score / len(lineups_df)
    
    if optimal_score > 0:
        print(f"Tournament metrics for {tournament}:")
        print(f"  Optimal score: {optimal_score:.2f}")
        print(f"  Min cash threshold (70%): {optimal_score * 0.70:.2f}")
        print(f"  First place threshold (95%): {optimal_score * 0.95:.2f}")
        print(f"  Expected value: ${expected_value:.2f}")

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
    if expected_value > tournament_highlights[tournament]['ev']['score']:
        tournament_highlights[tournament]['ev'] = {
            'score': expected_value,
            'weights': weights['components'].copy()
        }
    
    return avg_score, best_lineup_score, expected_value, tournament_highlights

def backtest_weights() -> Tuple[List[Dict], List[float], List[float], List[float], List[float], List[float], Dict]:
    """
    Test different weight combinations and return the top performers based on
    multiple metrics including success score.
    """
    tournaments = ["The_Sentry", "Sony_Open_in_Hawaii", "The_American_Express", "Farmers_Insurance_Open", "AT&T_Pebble_Beach_Pro-Am", "WM_Phoenix_Open", "Mexico_Open_at_VidantaWorld", "Cognizant_Classic_in_The_Palm_Beaches", "Arnold_Palmer_Invitational_presented_by_Mastercard", "THE_PLAYERS_Championship", "Valspar_Championship", "Texas_Children's_Houston_Open", "Valero_Texas_Open"]
    # tournaments = tournaments[-5:]
    weight_combinations = generate_weight_combinations()
    results_file = "backtest/pga_v5_backtest_results.csv"
    
    # Initialize tournament highlights with proper initial values
    tournament_highlights = {
        t: {
            'avg': {'score': float('-inf'), 'weights': None},
            'best': {'score': float('-inf'), 'weights': None},
            'ev': {'score': float('-inf'), 'weights': None}
        } for t in tournaments
    }
    
    # Initialize or load existing results
    if os.path.exists(results_file):
        results_df = pd.read_csv(results_file)
        results_df['weights'] = results_df['weights'].apply(eval)
    else:
        results_df = pd.DataFrame(columns=['tournament', 'weights', 'avg_score', 'best_score', 'ev_score'])
    
    # Track scores
    tournament_scores = {t: {'avg': {}, 'best': {}, 'ev': {}} for t in tournaments}
    
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
                ev_score = existing_result['ev_score'].iloc[0]
                print(f"{tournament} - Avg: {avg_score:.2f}, Best: {best_score:.2f}, EV: ${ev_score:.2f} (cached)")
                
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
                if ev_score > tournament_highlights[tournament]['ev']['score']:
                    tournament_highlights[tournament]['ev'] = {
                        'score': ev_score,
                        'weights': weights['components'].copy()
                    }
            else:
                lineups_df = pga_main(tournament, num_lineups=20, weights=weights)
                avg_score, best_score, ev_score, tournament_highlights = evaluate_lineup_performance(
                    tournament, lineups_df, tournament_highlights, weights
                )
                
                new_row = pd.DataFrame({
                    'tournament': [tournament],
                    'weights': [weights],
                    'avg_score': [avg_score],
                    'best_score': [best_score],
                    'ev_score': [ev_score]
                })
                results_df = pd.concat([results_df, new_row], ignore_index=True)
                results_df.to_csv(results_file, index=False)
                print(f"{tournament} - Avg: {avg_score:.2f}, Best: {best_score:.2f}, EV: ${ev_score:.2f} (new)")
            
            # Store scores
            tournament_scores[tournament]['avg'][str(weights['components'])] = avg_score
            tournament_scores[tournament]['best'][str(weights['components'])] = best_score
            tournament_scores[tournament]['ev'][str(weights['components'])] = ev_score

    # Calculate normalized scores for both metrics
    weight_performances = {'avg': {}, 'best': {}, 'ev': {}}
    weight_raw_averages = {'avg': {}, 'best': {}, 'ev': {}}
    
    for weights in weight_combinations:
        weight_key = str(weights['components'])
        for metric in ['avg', 'best', 'ev']:
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
                if metric == 'ev':
                    # Sum EV scores across tournaments
                    weight_raw_averages[metric][weight_key] = sum(raw_scores)
                else:
                    # Calculate average for other metrics
                    weight_raw_averages[metric][weight_key] = sum(raw_scores) / len(raw_scores)
                weight_performances[metric][weight_key] = sum(normalized_scores) / len(normalized_scores)
    
    # Get top 3 for both metrics
    top_weights = {'avg': [], 'best': [], 'ev': []}
    top_scores = {'avg': [], 'best': [], 'ev': []}
    top_raw_averages = {'avg': [], 'best': [], 'ev': []}
    
    for metric in ['avg', 'best', 'ev']:
        # Sort by raw averages for all metrics
        sorted_weights = sorted(weight_raw_averages[metric].items(), key=lambda x: x[1], reverse=True)
            
        for weight_str, score in sorted_weights[:3]:
            for w in weight_combinations:
                if str(w['components']) == weight_str:
                    top_weights[metric].append(w)
                    top_scores[metric].append(score)
                    top_raw_averages[metric].append(weight_raw_averages[metric][weight_str])
                    break
    
    # Create DataFrame with weight components and EV scores
    component_analysis = pd.DataFrame([{
        'odds_weight': eval(weight_key)['odds'],
        'fit_weight': eval(weight_key)['fit'],
        'history_weight': eval(weight_key)['history'],
        'form_weight': eval(weight_key)['form'],
        'ev_score': score
    } for weight_key, score in weight_raw_averages['ev'].items()])
    
    # Calculate correlations with EV
    weight_correlations = component_analysis.corr()['ev_score'].drop('ev_score')
    print("\nWeight Component Correlations with EV:")
    print(weight_correlations.sort_values(ascending=False))
    
    # Optional: Add scatter plots for visualization
    try:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        for idx, component in enumerate(['odds_weight', 'fit_weight', 'history_weight', 'form_weight']):
            ax = axes[idx // 2, idx % 2]
            ax.scatter(component_analysis[component], component_analysis['ev_score'])
            ax.set_xlabel(component.replace('_', ' ').title())
            ax.set_ylabel('Expected Value')
            ax.set_title(f'EV vs {component.replace("_", " ").title()}')
        plt.tight_layout()
        plt.savefig('backtest/weight_correlations.png')
        plt.close()
    except ImportError:
        print("Matplotlib not installed - skipping plots")
    
    return (top_weights['avg'], top_scores['avg'], top_raw_averages['avg'], 
            top_weights['best'], top_scores['best'], top_raw_averages['best'],
            top_weights['ev'], top_scores['ev'], top_raw_averages['ev'],
            tournament_highlights)

if __name__ == "__main__":
    avg_weights, avg_scores, avg_raw_scores, best_weights, best_scores, best_raw_scores, ev_weights, ev_scores, ev_raw_scores, highlights = backtest_weights()
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
    print("Top 3 Performing Weights (By Expected Value):")
    for i in range(3):
        print(f"\n{i+1}. Total Expected Value: ${ev_raw_scores[i]:.2f}")
        print(f"   Per Lineup EV: ${ev_raw_scores[i]/20:.2f}")
        print(f"   Components: {ev_weights[i]['components']}")

    print("\n" + "=" * 50)
    print("Tournament-Specific Highlights:")
    for tournament in highlights:
        print(f"\n{tournament}:")
        print(f"  Best Average Score: {highlights[tournament]['avg']['score']:.2f}")
        print(f"  Best Average Weights: {highlights[tournament]['avg']['weights']}")
        print(f"  Best Single Lineup: {highlights[tournament]['best']['score']:.2f}")
        print(f"  Best Single Weights: {highlights[tournament]['best']['weights']}")
        print(f"  Best Expected Value: ${highlights[tournament]['ev']['score']:.2f}")
        print(f"  Best EV Weights: {highlights[tournament]['ev']['weights']}")
