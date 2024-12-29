from typing import Dict, List, Tuple
import pandas as pd
import numpy as np
from scipy.optimize import minimize
from concurrent.futures import ProcessPoolExecutor
import itertools
from pga_v5 import CourseStats, fix_names, Golfer, CoursePlayerFit, PGATourStatsScraper, StatCorrelation, DistanceMapping, AccuracyMapping, StrokesGainedMapping, odds_to_score
from course_info import load_course_stats
from utils import TOURNAMENT_LIST_2024
import os
import pdb

'''
Script that optimizes weights for course-to-player fit only

Output: 
- optimization_results/course_to_player_fit_final_weights.csv: the final weights for each stat
- optimization_results/course_to_player_fit_optimization_iterations.csv: the optimization iterations
- optimization_results/course_to_player_fit_tournament_correlations.csv: the correlations by tournament
'''


class CourseFitBacktest:
    def __init__(self):
        self.results_cache = {}

    def _calculate_performance_correlation(self, 
                                        tournament_name: str,
                                        weights: List[float],
                                        golfers: List['Golfer'],
                                        course_stats: CourseStats,
                                        actual_results: pd.DataFrame) -> float:
        """
        Calculate correlation between course fit scores and actual performance
        for a given tournament using specific weights
        """
        # Create CoursePlayerFit with custom weights
        custom_mappings = [
            DistanceMapping('adj_driving_distance', 'driving_distance', weights[0]),
            AccuracyMapping('adj_driving_accuracy', 'driving_accuracy', weights[1]),
            AccuracyMapping('fw_width', 'driving_accuracy', weights[2], is_inverse=True),
            StrokesGainedMapping('ott_sg', 'sg_off_tee', weights[3]),
            StrokesGainedMapping('app_sg', 'sg_approach', weights[4]),
            StrokesGainedMapping('arg_sg', 'sg_around_green', weights[5]),
            StrokesGainedMapping('putt_sg', 'sg_putting', weights[6]),
            AccuracyMapping('arg_bunker_sg', 'scrambling_sand', weights[7])
        ]
        
        course_fit = CoursePlayerFit(course_stats, golfers, custom_mappings)
        course_fit.STAT_MAPPINGS = custom_mappings
        
        # Calculate fit scores for all golfers
        fit_scores = []
        actual_scores = []
        
        for golfer in golfers:
            try:
                # Get course fit score
                fit_score = course_fit.calculate_fit_score(golfer)['overall_fit']
                
                # Get actual performance
                golfer_name = golfer.get_clean_name
                actual_score = actual_results[
                    actual_results['Name'] == golfer_name
                ]['DK Score'].iloc[0]
                
                fit_scores.append(fit_score)
                actual_scores.append(actual_score)
                
            except Exception:
                continue
        
        # Calculate correlation
        if len(fit_scores) > 0:
            # Convert to numpy arrays for calculations
            fit_scores = np.array(fit_scores)
            actual_scores = np.array(actual_scores)
            
            # Normalize both arrays to 0-1 range
            # Use z-score normalization to preserve relative differences
            fit_scores = (fit_scores - np.mean(fit_scores)) / np.std(fit_scores)
            actual_scores = (actual_scores - np.mean(actual_scores)) / np.std(actual_scores)
            # Calculate correlation
            correlation = np.corrcoef(fit_scores, actual_scores)[0, 1]
            
            # Handle NaN correlations (can happen with constant values)
            if np.isnan(correlation):
                return 0.0
            
            # Convert correlation to positive range (0 to 1)
            # This maps -1 to 0, 0 to 0.5, and 1 to 1
            correlation = (correlation + 1) / 2

            # print(f"Variance in fit_scores: {np.var(fit_scores)}, actual_scores: {np.var(actual_scores)}")

            
            return correlation
        return 0.0

    def optimize_weights(self, 
                    tournaments: List[str],
                    all_golfers: Dict[str, List['Golfer']],
                    course_stats: Dict[str, CourseStats],
                    results: Dict[str, pd.DataFrame]) -> Tuple[List[float], float]:
        """
        Optimize weights across multiple tournaments to maximize correlation
        """
        optimization_data = []
        
        bounds = [(0, 2)] * 8  # Allow wider range
        constraints = [
            {'type': 'ineq', 'fun': lambda x: 2 - np.sum(x)},  # More flexible sum
            {'type': 'ineq', 'fun': lambda x: np.sum(x) - 0.5}
        ]
        def objective(weights):
            correlations = []
            iteration_data = {
                'weights': weights.tolist(),
                'correlations': {}
            }
            
            print("\nTesting weights:", [f"{w:.3f}" for w in weights])
            for i, tournament in enumerate(tournaments, 1):
                print(f"  Processing tournament {i}/{len(tournaments)}: {tournament}", end='\r')
                correlation = self._calculate_performance_correlation(
                    tournament,
                    weights,
                    all_golfers[tournament],
                    course_stats[tournament],
                    results[tournament]
                )
                correlations.append(correlation)
                iteration_data['correlations'][tournament] = correlation
            
            mean_corr = np.mean([c for c in correlations if not np.isnan(c)])
            print(f"\nMean correlation: {mean_corr:.4f}" + " " * 50)  # Extra spaces to clear previous line
            iteration_data['mean_correlation'] = mean_corr
            optimization_data.append(iteration_data)
            
            return -mean_corr

        # Try multiple starting points to avoid local minima
        best_result = None
        best_score = float('-inf')
        
        # Reduce the number of starting points
        starting_points = [
            np.ones(8) / 8,  # Equal weights
            np.array([0.5, 0.3, 0.05, 0.05, 0.05, 0.025, 0.015, 0.01]),  # Driving bias
            np.array([0.05, 0.05, 0.05, 0.5, 0.2, 0.1, 0.03, 0.02]),     # Approach bias
        ]
        
        for initial_weights in starting_points:
            result = minimize(
                objective,
                initial_weights,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'ftol': 1e-6, 'maxiter': 500}  # Reduced precision and iterations
            )
            
            if result.success and -result.fun > best_score:
                best_score = -result.fun
                best_result = result
        
        if best_result is None:
            raise ValueError("Optimization failed to find a solution")
        
        final_weights = best_result.x / np.sum(best_result.x)
        
        # Save optimization results
        stat_names = [
            "Driving Distance",
            "Driving Accuracy",
            "Fairway Width",
            "Off the Tee SG",
            "Approach SG",
            "Around Green SG",
            "Putting SG",
            "Sand Save"
        ]
        
        # Save final weights with more precision
        weights_df = pd.DataFrame({
            'Stat': stat_names,
            'Weight': final_weights
        })
        weights_df.to_csv('optimization_results/course_to_player_fit_final_weights.csv', index=False, float_format='%.6f')
        
        # Print optimization results
        print("\nOptimization Results:")
        print("--------------------")
        for stat, weight in zip(stat_names, final_weights):
            print(f"{stat:<20}: {weight:.6f}")
        print(f"\nMean Correlation: {-best_result.fun:.6f}")
        
        return final_weights, -best_result.fun

    def test_weights(self,
                    weights: List[float],
                    tournaments: List[str],
                    all_golfers: Dict[str, List['Golfer']],
                    course_stats: Dict[str, CourseStats],
                    results: Dict[str, pd.DataFrame]) -> Dict[str, float]:
        """
        Test a set of weights across multiple tournaments and return correlations
        """
        correlations = {}
        print("\nTesting final weights across tournaments:")
        for i, tournament in enumerate(tournaments, 1):
            print(f"Processing tournament {i}/{len(tournaments)}: {tournament}", end='\r')
            correlation = self._calculate_performance_correlation(
                tournament,
                weights,
                all_golfers[tournament],
                course_stats[tournament],
                results[tournament]
            )
            correlations[tournament] = correlation
        
        print("\nTesting complete!" + " " * 50)  # Extra spaces to clear previous line
        return correlations

def load_backtest_data():
    """Load all necessary data for backtesting"""
    
    # Initialize scraper
    scraper = PGATourStatsScraper()
    print("Initialized PGA Tour Stats Scraper")
    
    # Load tournaments from TOURNAMENT_LIST_2024
    tournaments = list(TOURNAMENT_LIST_2024.keys())
    
    # Initialize dictionaries
    all_golfers = {}
    course_stats = {}
    results = {}
    
    for tournament in tournaments:
        try:
            print(f"\nProcessing {tournament}...")
            
            # Load and merge DK salaries and odds data
            dk_df = pd.read_csv(f'2024/{tournament}/DKSalaries.csv')
            odds_df = pd.read_csv(f'2024/{tournament}/odds.csv')
            
            # Fix names in both dataframes
            dk_df['Name'] = dk_df['Name'].apply(fix_names)
            odds_df['Name'] = odds_df['Name'].apply(fix_names)
            
            # Merge dataframes
            golfer_df = pd.merge(dk_df, odds_df, on='Name', how='inner')
            golfers = [Golfer(row) for _, row in golfer_df.iterrows()]
            
            # Update golfers with their stats
            print(f"Fetching stats for {len(golfers)} golfers...")
            scraper.update_golfer_list(golfers)
            
            # Calculate fit scores
            course_stat = load_course_stats(tournament)
            analyzer = CoursePlayerFit(course_stat, golfers)
            
            # Load odds data
            odds_df = pd.read_csv(f'2024/{tournament}/odds.csv')
            odds_df['Name'] = odds_df['Name'].apply(fix_names)
            
            # Calculate odds total
            odds_df['Tournament Winner'] = odds_df['Tournament Winner'].apply(
                lambda x: odds_to_score(x, "Tournament Winner", w=0.6))
            odds_df['Top 5 Finish'] = odds_df['Top 5 Finish'].apply(
                lambda x: odds_to_score(x, "Top 5 Finish", t5=0.5))
            odds_df['Top 10 Finish'] = odds_df['Top 10 Finish'].apply(
                lambda x: odds_to_score(x, "Top 10 Finish", t10=0.8))
            odds_df['Top 20 Finish'] = odds_df['Top 20 Finish'].apply(
                lambda x: odds_to_score(x, "Top 20 Finish", t20=0.4))
            
            odds_df['Odds Total'] = (
                odds_df['Tournament Winner'] +
                odds_df['Top 5 Finish'] +
                odds_df['Top 10 Finish'] +
                odds_df['Top 20 Finish']
            )
            
            # Create name to odds mapping
            name_to_odds = dict(zip(odds_df['Name'], odds_df['Odds Total']))
            
            # Update golfers with fit scores and odds
            for golfer in golfers:
                # Set fit score
                fit_score = analyzer.calculate_fit_score(golfer)['overall_fit']
                golfer.set_fit_score(fit_score)
                
                # Set odds total
                odds_total = name_to_odds.get(golfer.get_clean_name, 0.0)
                golfer.set_odds_total(odds_total)
            
            all_golfers[tournament] = golfers
            course_stats[tournament] = course_stat
            
            # Load DK scoring results
            tournament_id = TOURNAMENT_LIST_2024[tournament]['ID']
            results_df = pd.read_csv(f'past_results/2024/dk_points_id_{tournament_id}.csv')
            
            # Ensure results DataFrame has required columns
            if 'Name' not in results_df.columns or 'DK Score' not in results_df.columns:
                print(f"Warning: Results for {tournament} missing required columns")
                continue
            
            # Clean names in results DataFrame to match golfer format
            results_df['Name'] = results_df['Name'].apply(fix_names)
            
            # Print name matching diagnostics
            golfer_names = set(g.get_clean_name for g in golfers)
            result_names = set(results_df['Name'])
            
            print("\nName matching diagnostics:")
            print(f"Golfers in DK data: {len(golfer_names)}")
            print(f"Players in results: {len(result_names)}")
            
            missing_results = golfer_names - result_names
            if missing_results:
                print("\nGolfers missing from results:")
                for name in missing_results:
                    print(f"- {name}")
                    
                print("\nAvailable result names:")
                for name in sorted(result_names)[:10]:  # Show first 10 names
                    print(f"- {name}")
                
            results[tournament] = results_df
            
            print(f"Successfully loaded complete data for {tournament}")
                
        except FileNotFoundError as e:
            print(f"Warning: Missing files for {tournament}: {str(e)}")
            continue
        except Exception as e:
            print(f"Error processing {tournament}: {str(e)}")
            continue
            
    # Clean up scraper resources
    try:
        scraper._quit_driver()
    except:
        pass
    
    
    # Filter tournaments to only those with complete data
    complete_tournaments = [t for t in tournaments 
                          if t in all_golfers 
                          and t in course_stats 
                          and t in results]
    
    if not complete_tournaments:
        raise ValueError("No tournaments with complete data found")
    
    print(f"\nLoaded complete data for {len(complete_tournaments)} tournaments")
    
    # Validate stats are populated
    for tournament in complete_tournaments:
        golfers_with_stats = sum(1 for g in all_golfers[tournament] 
                                if g.stats['current']['strokes_gained'].total != 0)
        print(f"\n{tournament}:")
        print(f"  Total golfers: {len(all_golfers[tournament])}")
        print(f"  Golfers with stats: {golfers_with_stats}")
        if golfers_with_stats == 0:
            print("  Warning: No golfers have stats populated!")
    
    return (complete_tournaments,
            {t: all_golfers[t] for t in complete_tournaments},
            {t: course_stats[t] for t in complete_tournaments},
            {t: results[t] for t in complete_tournaments})

def main():
    """Example usage"""
    # try:
    tournaments, all_golfers, course_stats, results = load_backtest_data()
    
    # Print summary of loaded data
    print("\nData Summary:")
    for tournament in tournaments:
        print(f"\n{tournament}:")
        print(f"  Golfers: {len(all_golfers[tournament])}")
        print(f"  Course: {course_stats[tournament].name}")
        print(f"  Results: {len(results[tournament])} players")
        
        # Check data alignment
        golfer_names = set(g.get_clean_name for g in all_golfers[tournament])
        result_names = set(results[tournament]['Name'])
        missing_results = golfer_names - result_names
        if missing_results:
            print(f"  Warning: {len(missing_results)} golfers missing results")
    
    backtest = CourseFitBacktest()
    optimal_weights, mean_correlation = backtest.optimize_weights(
        tournaments,
        all_golfers,
        course_stats,
        results
    )
    
    print("\nOptimal Weights:")
    stat_names = [
        "Driving Distance",
        "Driving Accuracy",
        "Fairway Width",
        "Off the Tee SG",
        "Approach SG",
        "Around Green SG",
        "Putting SG",
        "Sand Save",
        "Penalties"
    ]
    
    for name, weight in zip(stat_names, optimal_weights):
        print(f"{name}: {weight:.3f}")
    
    print(f"\nMean Correlation: {mean_correlation:.3f}")
    
    # Test optimal weights by tournament
    correlations = backtest.test_weights(
        optimal_weights,
        tournaments,
        all_golfers,
        course_stats,
        results
    )
    
    print("\nCorrelations by Tournament:")
    for tournament, correlation in correlations.items():
        print(f"{tournament}: {correlation:.3f}")
        
    # except Exception as e:
    #     print(f"Error during backtesting: {str(e)}")

if __name__ == "__main__":
    main()