# Standard library imports
import os
import warnings
from concurrent.futures import ProcessPoolExecutor
from itertools import combinations

# Third-party imports
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from tqdm import tqdm

# Local imports
from utils import TOURNAMENT_LIST_2025
# update imports from pga_v4
from pga_v5 import fix_names, odds_to_score, DKLineupOptimizer, optimize_dk_lineups
from Legacy.pga_dk_scoring import dk_points_df

'''
Script that optimizes weights for the DK scoring system based on odds

Output: 
- weight_optimization_results.csv: the optimization results
'''

# Suppress warnings
warnings.filterwarnings('ignore')


class WeightOptimizer:
    def __init__(self, tournaments=None):
        """Initialize the optimizer with a list of tournaments to analyze"""
        self.tournaments = tournaments or list(TOURNAMENT_LIST_2025.keys())
        self.results_cache = {}
        
    def validate_data(self, df):
        """Validate and clean data before optimization"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        # Check for NaN values in numeric columns
        nan_cols = df[numeric_cols].columns[df[numeric_cols].isna().any()].tolist()
        if nan_cols:
            print(f"Found NaN values in numeric columns: {nan_cols}")
            print("NaN counts:")
            print(df[nan_cols].isna().sum())
            
            for col in nan_cols:
                if col == 'Salary':
                    print(f"Critical column {col} has NaN values")
                    print(df[df[col].isna()])
                    raise ValueError(f"Cannot have NaN values in {col}")
                else:
                    df[col] = df[col].fillna(0)
                    
        # Check for inf values only in numeric columns
        inf_cols = df[numeric_cols].columns[np.isinf(df[numeric_cols]).any()].tolist()
        if inf_cols:
            print(f"Found inf values in columns: {inf_cols}")
            for col in inf_cols:
                df[col] = df[col].replace([np.inf, -np.inf], df[col].replace([np.inf, -np.inf], np.nan).median())
                
        # Verify required columns exist
        required_cols = ['Name + ID', 'Salary', 'Total', 'Value']
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")
        
        return df
        
    def load_tournament_data(self, tournament):
        """Load and prepare data for a single tournament"""
        try:
            sal_df = pd.read_csv(f'2025/{tournament}/DKSalaries.csv')
            odds_df = pd.read_csv(f'2025/{tournament}/odds.csv')
            
            sal_df = sal_df[['Name', 'Name + ID', 'Salary']]
            odds_cols = ["Name", "Tournament Winner", "Top 5 Finish", "Top 10 Finish", "Top 20 Finish"]
            odds_df = odds_df[odds_cols]
            
            try:
                results_df = pd.read_csv(f'past_results/2025/dk_points_id_{TOURNAMENT_LIST_2025[tournament]["ID"]}.csv')
                results_df = results_df[['Name', 'DK Score']]
            except FileNotFoundError:
                print(f"Results not found for {tournament}, generating them...")
                dk_points_df(TOURNAMENT_LIST_2025[tournament]["ID"])
                results_df = pd.read_csv(f'past_results/2025/dk_points_id_{TOURNAMENT_LIST_2025[tournament]["ID"]}.csv')
                results_df = results_df[['Name', 'DK Score']]
            
            sal_df["Name"] = sal_df["Name"].apply(fix_names)
            odds_df["Name"] = odds_df["Name"].apply(fix_names)
            
            df = pd.merge(results_df, odds_df, on="Name", how="inner")
            df = pd.merge(df, sal_df, on="Name", how="inner")
            
            required_cols = ['Name', 'Name + ID', 'Salary', 'DK Score', 
                           'Tournament Winner', 'Top 5 Finish', 'Top 10 Finish', 'Top 20 Finish']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
                
            return df
            
        except Exception as e:
            print(f"Error loading data for {tournament}: {e}")
            return None
            
    def calculate_lineup_score(self, weights, tournament_data):
        """Calculate lineup score for a given set of weights using optimization flow"""
        tournament_data = tournament_data.copy()
        odds_cols = ["Tournament Winner", "Top 5 Finish", "Top 10 Finish", "Top 20 Finish"]
        
        print("\nTesting weights:", weights)
        
        # Apply each weight to its corresponding odds column
        for i, col in enumerate(odds_cols):
            if col == "Tournament Winner":
                tournament_data[col] = tournament_data[col].apply(
                    lambda x: odds_to_score(x, col, w=weights[i], t5=0, t10=0, t20=0))
            elif col == "Top 5 Finish":
                tournament_data[col] = tournament_data[col].apply(
                    lambda x: odds_to_score(x, col, w=0, t5=weights[i], t10=0, t20=0))
            elif col == "Top 10 Finish":
                tournament_data[col] = tournament_data[col].apply(
                    lambda x: odds_to_score(x, col, w=0, t5=0, t10=weights[i], t20=0))
            else:  # Top 20 Finish
                tournament_data[col] = tournament_data[col].apply(
                    lambda x: odds_to_score(x, col, w=0, t5=0, t10=0, t20=weights[i]))
            
        tournament_data["Total"] = tournament_data[odds_cols].sum(axis=1)
        tournament_data["Value"] = tournament_data["Total"] / tournament_data["Salary"] * 1000
        tournament_data = self.validate_data(tournament_data)
        
        optimizer = DKLineupOptimizer()
        lineups = optimizer.generate_lineups(tournament_data, num_lineups=20)
        optimized_lineups = optimize_dk_lineups(tournament_data, num_lineups=20)
        
        actual_scores = []
        for _, lineup in optimized_lineups.iterrows():
            lineup_score = 0
            for i in range(1, 7):
                player_name = lineup[f'G{i}']
                player_score = tournament_data[tournament_data['Name + ID'] == player_name]['DK Score'].iloc[0]
                lineup_score += player_score
            actual_scores.append(lineup_score)
            
        mean_score = np.mean(actual_scores)
        print(f"Mean lineup score with weights {weights}: {mean_score}")
        return -mean_score
            
    def optimize_single_tournament(self, tournament):
        """Optimize weights for a single tournament"""
        print(f"Optimizing weights for {tournament}")
        tournament_data = self.load_tournament_data(tournament)
        if tournament_data is None:
            return None
            
        n_starts = 5
        best_score = float('-inf')
        best_result = None
        tested_weights = set()  # Keep track of tested weight combinations
        
        # Progress bar for optimization rounds
        with tqdm(total=n_starts, desc=f"Optimization rounds for {tournament}") as pbar:
            attempts = 0
            while attempts < n_starts:
                # Generate initial weights ensuring they're sufficiently different
                initial_weights = np.random.uniform(0.1, 2, 4)
                initial_weights = initial_weights / sum(initial_weights) * 2
                
                # Convert weights to tuple for hashable comparison
                weights_key = tuple(np.round(initial_weights, 2))
                
                # Skip if we've already tested very similar weights
                if weights_key in tested_weights:
                    continue
                    
                tested_weights.add(weights_key)
                attempts += 1
                
                bounds = [(0, 2)] * 4
                constraints = [
                    {'type': 'ineq', 'fun': lambda x: 8 - sum(x)},
                    {'type': 'ineq', 'fun': lambda x: sum(x)},
                    {'type': 'ineq', 'fun': lambda x: min(x) + 0.1}
                ]
                
                # Add random perturbation to avoid local minima
                def perturbed_objective(x):
                    noise = np.random.normal(0, 0.01, size=len(x))  # Small random noise
                    return self.calculate_lineup_score(x + noise, tournament_data)
                
                result = minimize(
                    perturbed_objective,
                    initial_weights,
                    method='SLSQP',
                    bounds=bounds,
                    constraints=constraints,
                    options={'ftol': 1e-2, 'eps': 0.1, 'maxiter': 20}  # Increased step size
                )
                
                score = -result.fun if not np.isnan(result.fun) else float('-inf')
                
                if score > best_score:
                    best_score = score
                    best_result = result
                    pbar.set_postfix({'best_score': best_score})
                
                pbar.update(1)
        
        optimal_weights = best_result.x
        final_score = best_score
        
        print(f"\nOptimization results for {tournament}:")
        print("Final score:", final_score)
        print("Optimal weights:", optimal_weights)
        
        return {
            'tournament': tournament,
            'optimal_weights': optimal_weights,
            'score': final_score,
            'success': best_result.success,
            'message': best_result.message
        }
        
    def optimize_weights(self):
        """Optimize weights across all tournaments"""
        # Check for existing results
        try:
            existing_results = pd.read_csv('weight_optimization_results.csv')
            tested_tournaments = set(existing_results['tournament'])
            tournaments_to_test = [t for t in self.tournaments if t not in tested_tournaments]
            
            if existing_results.empty:
                print("No existing results found. Testing all tournaments.")
                results = []
            else:
                print(f"Found {len(tested_tournaments)} previously tested tournaments.")
                print("Already tested:", tested_tournaments)
                print(f"\nTesting {len(tournaments_to_test)} new tournaments.")
                # Convert existing results to the same format as new results will have
                results = [
                    {
                        'tournament': row['tournament'],
                        'optimal_weights': [row['win_weight'], row['top5_weight'], 
                                        row['top10_weight'], row['top20_weight']],
                        'score': row['score'],
                        'success': True,  # Assuming previous results were successful
                        'message': 'Loaded from existing results'
                    }
                    for _, row in existing_results.iterrows()
                ]
        except FileNotFoundError:
            print("No existing results file found. Testing all tournaments.")
            tournaments_to_test = self.tournaments
            results = []
            
        if not tournaments_to_test:
            print("All tournaments have already been tested!")
            return results
        
        # Process one tournament at a time
        for tournament in tqdm(tournaments_to_test, desc="Processing tournaments"):
            result = self.optimize_single_tournament(tournament)
            if result is not None:
                results.append(result)
                # Convert to DataFrame and save after each tournament in the desired format
                print(results)
                results_df = pd.DataFrame({
                    'tournament': [r['tournament'] for r in results],
                    'course': [TOURNAMENT_LIST_2025[r['tournament']]['Course'] for r in results],
                    'score': [r['score'] for r in results],
                    'win_weight': [r['optimal_weights'][0] for r in results],
                    'top5_weight': [r['optimal_weights'][1] for r in results],
                    'top10_weight': [r['optimal_weights'][2] for r in results],
                    'top20_weight': [r['optimal_weights'][3] for r in results]
                })
                results_df.to_csv('optimization_results/weight_optimization_results.csv', index=False)
                
        self.analyze_results(results)
        return results
    
    def analyze_results(self, results):
        """Analyze optimization results"""
        df_results = pd.DataFrame(results)
        self.analyzed_results = df_results
        
        avg_weights = np.mean([r['optimal_weights'] for r in results], axis=0)
        
        print("\nWeight Analysis:")
        print("Average Optimal Weights:")
        categories = ["Win", "Top 5", "Top 10", "Top 20"]
        for i, category in enumerate(categories):
            print(f"{category}: {avg_weights[i].round(3)}")
            
        var_weights = np.var([r['optimal_weights'] for r in results], axis=0)
        print("\nWeight Variance:")
        for i, category in enumerate(categories):
            print(f"{category}: {var_weights[i].round(3)}")
            
        scores = [r['score'] for r in results]
        print(f"\nScore Statistics:")
        print(f"Mean Score: {np.mean(scores):.2f}")
        print(f"Std Dev: {np.std(scores):.2f}")
        print(f"Min Score: {np.min(scores):.2f}")
        print(f"Max Score: {np.max(scores):.2f}")
        
    def get_best_performing_weights(self):
        """Returns the weights from the tournament that produced the highest score"""
        if not hasattr(self, 'analyzed_results'):
            raise ValueError("Must run optimize_weights first")
            
        best_tournament = self.analyzed_results.loc[self.analyzed_results['score'].idxmax()]
        return {
            'tournament': best_tournament['tournament'],
            'course': best_tournament['course'],
            'score': best_tournament['score'],
            'weights': {
                'win': best_tournament['win_weight'],
                'top5': best_tournament['top5_weight'],
                'top10': best_tournament['top10_weight'],
                'top20': best_tournament['top20_weight']
            }
        }
        
    def get_tournament_specific_weights(self, tournament_name):
        """Returns the optimal weights for a specific tournament"""
        if not hasattr(self, 'analyzed_results'):
            raise ValueError("Must run optimize_weights first")
            
        tournament_data = self.analyzed_results[
            self.analyzed_results['tournament'] == tournament_name
        ].iloc[0]
        
        return {
            'tournament': tournament_name,
            'course': tournament_data['course'],
            'score': tournament_data['score'],
            'weights': {
                'win': tournament_data['win_weight'],
                'top5': tournament_data['top5_weight'],
                'top10': tournament_data['top10_weight'],
                'top20': tournament_data['top20_weight']
            }
        }
        
    def get_weight_correlations(self):
        """Analyzes correlations between different weights and scores"""
        if not hasattr(self, 'analyzed_results'):
            raise ValueError("Must run optimize_weights first")
            
        weight_cols = ['win_weight', 'top5_weight', 'top10_weight', 'top20_weight', 'score']
        return self.analyzed_results[weight_cols].corr()


def main():
    optimizer = WeightOptimizer()
    results = optimizer.optimize_weights()
    print("\nOptimization complete. Results saved to weight_optimization_results.csv")


if __name__ == "__main__":
    main()