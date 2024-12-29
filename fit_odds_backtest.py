from typing import Dict, List, Tuple
import pandas as pd
import numpy as np
from course_fit_backtest import load_backtest_data
from pga_v5 import CoursePlayerFit, fix_names, Golfer, odds_to_score
import os

'''
Script that optimizes weights for course-to-player fit AND odds

Output: 
- optimization_results/fit_odds_final_weights.csv: the final weights for each stat
- optimization_results/fit_odds_optimization_iterations.csv: the optimization iterations
- optimization_results/fit_odds_tournament_correlations.csv: the correlations by tournament
'''

class FitOddsBacktest:
    def __init__(self):
        self.results_cache = {}

    def _calculate_performance_correlation(self,
                                        tournament_name: str,
                                        odds_weight: float,
                                        fit_weight: float,
                                        golfers: List['Golfer'],
                                        actual_results: pd.DataFrame) -> float:
        """
        Calculate correlation between combined scores and actual performance
        for a given tournament using specific weights
        """
        # First collect all scores
        fit_scores = []
        odds_scores = []
        valid_golfers = []
        
        for golfer in golfers:
            if golfer.fit_score is not None and golfer._odds_total is not None:
                fit_scores.append(golfer.fit_score)
                odds_scores.append(golfer._odds_total)
                valid_golfers.append(golfer)
        
        if not valid_golfers:
            return 0.0
            
        # Normalize scores using min-max scaling
        fit_scores = np.array(fit_scores)
        odds_scores = np.array(odds_scores)
        
        normalized_fit = (fit_scores - np.min(fit_scores)) / (np.max(fit_scores) - np.min(fit_scores))
        normalized_odds = (odds_scores - np.min(odds_scores)) / (np.max(odds_scores) - np.min(odds_scores))
        
        # Calculate total scores using normalized values
        scores_data = []
        for i, golfer in enumerate(valid_golfers):
            total = (odds_weight * normalized_odds[i]) + (fit_weight * normalized_fit[i])
            scores_data.append({
                'Name': golfer.get_clean_name,
                'Total': total
            })
        
        scores_df = pd.DataFrame(scores_data)
        
        # Calculate correlation
        merged = pd.merge(
            scores_df,
            actual_results[['Name', 'DK Score']],
            on='Name',
            how='inner'
        )
        
        if len(merged) == 0:
            return 0.0
            
        correlation = np.corrcoef(merged['Total'], merged['DK Score'])[0, 1]
        
        # Handle NaN correlations
        if np.isnan(correlation):
            return 0.0
            
        # Convert correlation to positive range (0 to 1)
        correlation = (correlation + 1) / 2
        
        return correlation

    def optimize_weights(self,
                        tournaments: List[str],
                        dk_data: Dict[str, pd.DataFrame],
                        results: Dict[str, pd.DataFrame]) -> Tuple[float, float, float]:
        """
        Optimize weights across multiple tournaments to maximize correlation
        """
        # Add debugging at the start
        print(f"Total tournaments to process: {len(tournaments)}")
        print("Available tournaments:", tournaments)
        
        optimization_data = []
        
        # Define weight ranges to try (must sum to 1)
        weight_range = np.arange(0, 1.1, 0.1)
        best_correlation = float('-inf')
        best_odds_weight = 0
        best_fit_weight = 0
        
        for odds_weight in weight_range:
            for fit_weight in weight_range:
                # Skip invalid weight combinations
                if abs(odds_weight + fit_weight - 1.0) > 0.001:
                    continue
                    
                correlations = []
                iteration_data = {
                    'odds_weight': odds_weight,
                    'fit_weight': fit_weight,
                    'correlations': {}
                }
                
                print(f"\nTesting weights - Odds: {odds_weight:.2f}, Fit: {fit_weight:.2f}")
                
                # Add tournament validation
                for i, tournament in enumerate(tournaments, 1):
                    if tournament not in dk_data or tournament not in results:
                        print(f"Warning: Missing data for tournament {tournament}")
                        continue
                        
                    print(f"  Processing tournament {i}/{len(tournaments)}: {tournament}")  # Removed \r
                    correlation = self._calculate_performance_correlation(
                        tournament,
                        odds_weight,
                        fit_weight,
                        dk_data[tournament],
                        results[tournament]
                    )
                    correlations.append(correlation)
                    iteration_data['correlations'][tournament] = correlation
                
                # Add debugging for correlations
                print(f"\nProcessed {len(correlations)} tournaments")
                valid_correlations = [c for c in correlations if not np.isnan(c)]
                print(f"Valid correlations: {len(valid_correlations)}/{len(correlations)}")
                
                mean_corr = np.mean(valid_correlations)
                print(f"Mean correlation: {mean_corr:.4f}" + " " * 50)
                
                iteration_data['mean_correlation'] = mean_corr
                optimization_data.append(iteration_data)
                
                if mean_corr > best_correlation:
                    best_correlation = mean_corr
                    best_odds_weight = odds_weight
                    best_fit_weight = fit_weight
        
        # Save optimization results
        pd.DataFrame(optimization_data).to_csv(
            'optimization_results/fit_odds_optimization_iterations.csv',
            index=False
        )
        
        return best_odds_weight, best_fit_weight, best_correlation

def main():
    """Main function to run the optimization process"""
    # Create optimization_results directory if it doesn't exist
    os.makedirs('optimization_results', exist_ok=True)
    
    # Load backtest data using existing function - now includes odds data
    tournaments, all_golfers, course_stats, results = load_backtest_data()
    
    # Create dk_data dictionary for each tournament
    dk_data = {}
    for tournament in tournaments:
        # Create fit scores for tournament
        analyzer = CoursePlayerFit(course_stats[tournament], all_golfers[tournament])
        
        # Calculate fit scores for all golfers
        for golfer in all_golfers[tournament]:
            fit_score = analyzer.calculate_fit_score(golfer)
            golfer.set_fit_score(fit_score['overall_fit'])  # Set the fit score
        
        dk_data[tournament] = all_golfers[tournament]  # Store golfer objects directly
    
    # Run optimization
    backtest = FitOddsBacktest()
    odds_weight, fit_weight, mean_correlation = backtest.optimize_weights(
        tournaments,
        dk_data,
        results
    )
    
    print("\nOptimal Weights Found:")
    print(f"Odds Weight: {odds_weight:.2f}")
    print(f"Fit Weight: {fit_weight:.2f}")
    print(f"Mean Correlation: {mean_correlation:.4f}")
    
    # Save final weights
    weights_df = pd.DataFrame([{
        'odds_weight': odds_weight,
        'fit_weight': fit_weight,
        'correlation': mean_correlation
    }])
    weights_df.to_csv('optimization_results/fit_odds_final_weights.csv', index=False)

if __name__ == "__main__":
    main()