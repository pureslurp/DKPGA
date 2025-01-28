import pandas as pd
import pulp
from typing import List, Dict
from dataclasses import dataclass
from typing import Dict, List, Optional
from datetime import datetime
import pandas as pd
import numpy as np
import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from pga_stats import create_pga_stats
from models import Golfer
from utils import fix_names
import os
import validate_player_data

'''
This is the main script that runs the PGA model for DraftKings

Looking for this version to be more systematic:

The model will take into considereation the following:
- Tournament History (tournament_history.csv) -- DONE
   - PGATour.com has a new site that has past 5 (same) event finishes
- Course Fit (course_fit.csv) -- DONE
- Form (Current and Long) (current_form.csv and pga_stats.csv) -- DONE
- Odds (odds.csv) -- DONE
   - Win, Top 5, Top 10, Top 20
   - Starting weights:
        Tournament Winner: 0.6
        Top 5: 0.5
        Top 10: 0.8
        Top 20: 0.4
- Robust Optimization (DKLineupOptimizer) to csv -- DONE
'''

TOURNEY = "Farmers_Insurance_Open"

def odds_to_score(col, header, w=1, t5=1, t10=1, t20=1):
    '''
    win:  30 pts
    top 5: 14
    top 10: 7
    top 20: 5
    '''
    if col < 0:
        final_line = (col/-110)
    else:
        final_line = (100/col)
    match header:
        case "Tournament Winner":
            return round(final_line * 30 * w, 3)
        case "Top 5 Finish":
            return round(final_line * 14 * t5, 3)
        case "Top 10 Finish":
            return round(final_line * 7 * t10, 3)
        case "Top 20 Finish":
            return round(final_line * 5 * t20, 3)

class DKLineupOptimizer:
    def __init__(self, salary_cap: int = 50000, lineup_size: int = 6):
        self.salary_cap = salary_cap
        self.lineup_size = lineup_size
        
    def generate_lineups(self, df: pd.DataFrame, num_lineups: int = 20, 
                        min_salary: int = 45000, exposure_limit: float = 0.66,
                        overlap_limit: int = 4) -> List[Dict]:
        """
        Generate optimal lineups using Integer Linear Programming with exposure and overlap limits
        
        Args:
            df: DataFrame with columns ['Name + ID', 'Salary', 'Total']
            num_lineups: Number of lineups to generate
            min_salary: Minimum total salary to use
            exposure_limit: Maximum percentage of lineups a player can appear in (0.0 to 1.0)
            overlap_limit: Maximum number of players that can overlap between lineups
        """
        lineups = []
        players = df.to_dict('records')
        
        # First generate all lineups with overlap constraint
        for i in range(num_lineups):
            # Create the model
            prob = pulp.LpProblem(f"DraftKings_Lineup_{i}", pulp.LpMaximize)
            
            # Decision variables - whether to include each player
            decisions = pulp.LpVariable.dicts("players",
                                           ((p['Name + ID']) for p in players),
                                           cat='Binary')
            
            # Objective: Maximize total projected points
            prob += pulp.lpSum([decisions[p['Name + ID']] * p['Total'] for p in players])
            
            # Constraint 1: Must select exactly 6 players
            prob += pulp.lpSum([decisions[p['Name + ID']] for p in players]) == self.lineup_size
            
            # Constraint 2: Must not exceed salary cap
            prob += pulp.lpSum([decisions[p['Name + ID']] * p['Salary'] for p in players]) <= self.salary_cap
            
            # Constraint 3: Must meet minimum salary
            prob += pulp.lpSum([decisions[p['Name + ID']] * p['Salary'] for p in players]) >= min_salary
            
            # Constraint 4: Maximum one player over $10000
            expensive_players = [p for p in players if p['Salary'] >= 10000]
            if expensive_players:
                prob += pulp.lpSum([decisions[p['Name + ID']] for p in expensive_players]) <= 1
            
            # Constraint 5: Maximum one player in lowest salary tier
            cheap_threshold = 6400 if min_salary >= 6000 else 6000
            cheap_players = [p for p in players if p['Salary'] <= cheap_threshold]
            if cheap_players:
                prob += pulp.lpSum([decisions[p['Name + ID']] for p in cheap_players]) <= 1
            
            # Constraint 6: Overlap constraint with previous lineups
            if i > 0:
                for prev_lineup in lineups:
                    prev_players = [p['Name + ID'] for p in [prev_lineup[f'G{j}'] for j in range(1, 7)]]
                    prob += pulp.lpSum([decisions[p] for p in prev_players]) <= overlap_limit
            
            # Solve the optimization problem
            prob.solve(pulp.PULP_CBC_CMD(msg=False))
            
            if pulp.LpStatus[prob.status] == 'Optimal':
                # Extract the lineup
                lineup = {
                    'G1': None, 'G2': None, 'G3': None, 
                    'G4': None, 'G5': None, 'G6': None,
                    'Salary': 0,
                    'TotalPoints': 0
                }
                
                idx = 0
                for p in players:
                    if decisions[p['Name + ID']].value() == 1:
                        lineup[f'G{idx+1}'] = p
                        lineup['Salary'] += p['Salary']
                        lineup['TotalPoints'] += p['Total']
                        idx += 1
                
                lineups.append(lineup)
            else:
                print(f"Could not find optimal solution for lineup {i+1}")
                break
        
        # Calculate player exposures
        player_counts = {}
        for lineup in lineups:
            for pos in ['G1', 'G2', 'G3', 'G4', 'G5', 'G6']:
                player = lineup[pos]['Name + ID']
                player_counts[player] = player_counts.get(player, 0) + 1
        
        # Sort lineups by total points (ascending) to modify worst lineups first
        lineups.sort(key=lambda x: x['TotalPoints'])
        
        # Identify overexposed players
        max_appearances = int(num_lineups * exposure_limit)
        overexposed = {player: count for player, count in player_counts.items() 
                      if count > max_appearances}
        
        if overexposed:
            print("\nAdjusting lineups for exposure limits...")
            print(f"Players over {exposure_limit*100}% exposure:")
            for player, count in overexposed.items():
                print(f"- {player}: {count}/{num_lineups} lineups ({count/num_lineups*100:.1f}%)")
            
            # Create pool of replacement players (sorted by Total, excluding overexposed)
            replacement_pool = sorted(
                [p for p in players if p['Name + ID'] not in overexposed],
                key=lambda x: x['Total'],
                reverse=True
            )
            
            # Adjust lineups to meet exposure limits
            for lineup in lineups:
                changes_made = False
                used_players = set()  # Track players already in this lineup
                
                # First, add all valid players to used_players set
                for pos in ['G1', 'G2', 'G3', 'G4', 'G5', 'G6']:
                    player = lineup[pos]['Name + ID']
                    if player not in overexposed or player_counts[player] <= max_appearances:
                        used_players.add(player)
                
                # Then make replacements
                for pos in ['G1', 'G2', 'G3', 'G4', 'G5', 'G6']:
                    player = lineup[pos]['Name + ID']
                    if player in overexposed and player_counts[player] > max_appearances:
                        # Find best replacement that fits salary constraints and isn't already in lineup
                        current_salary = lineup['Salary'] - lineup[pos]['Salary']
                        for replacement in replacement_pool:
                            replacement_id = replacement['Name + ID']
                            if (current_salary + replacement['Salary'] <= self.salary_cap and 
                                player_counts.get(replacement_id, 0) < max_appearances and
                                replacement_id not in used_players):
                                # Make the swap
                                player_counts[player] -= 1
                                player_counts[replacement_id] = player_counts.get(replacement_id, 0) + 1
                                lineup[pos] = replacement
                                lineup['Salary'] = current_salary + replacement['Salary']
                                lineup['TotalPoints'] = sum(lineup[f'G{i+1}']['Total'] for i in range(6))
                                used_players.add(replacement_id)  # Add to used players
                                changes_made = True
                                break
                
                if changes_made:
                    # Re-sort lineups after modifications
                    lineups.sort(key=lambda x: x['TotalPoints'])
        
        # Final sort by total points (descending)
        lineups.sort(key=lambda x: x['TotalPoints'], reverse=True)
        return lineups

def optimize_dk_lineups(dk_merge: pd.DataFrame, num_lineups: int = 20) -> pd.DataFrame:
    """
    Main function to generate optimized DraftKings lineups
    
    Args:
        dk_merge: DataFrame with merged odds and salary data
        num_lineups: Number of lineups to generate
    """
    # First, check for and handle NaN values
    critical_columns = ['Name + ID', 'Salary', 'Total']
    
    # Print initial stats
    print(f"\nInitial dataset size: {len(dk_merge)}")
    
    # Check for NaN values in each column
    for col in critical_columns:
        nan_count = dk_merge[col].isna().sum()
        if nan_count > 0:
            print(f"Found {nan_count} NaN values in {col}")
            if col == 'Total':
                # Fill NaN totals with 0
                dk_merge[col] = dk_merge[col].fillna(0)
                print(f"Filled NaN values in {col} with 0")
            else:
                # For other critical columns, we need to drop these rows
                dk_merge = dk_merge.dropna(subset=[col])
                print(f"Dropped rows with NaN in {col}")
    
    print(f"Final dataset size after handling NaN values: {len(dk_merge)}\n")
    
    # Create the optimizer and generate lineups
    optimizer = DKLineupOptimizer()
    lineups = optimizer.generate_lineups(dk_merge, num_lineups)
    
    # Convert lineups to DataFrame format
    lineup_rows = []
    for lineup in lineups:
        row = {}
        for pos in ['G1', 'G2', 'G3', 'G4', 'G5', 'G6']:
            row[pos] = lineup[pos]['Name + ID']
        row['Salary'] = lineup['Salary']
        row['TotalPoints'] = lineup['TotalPoints']
        lineup_rows.append(row)
        
    return pd.DataFrame(lineup_rows)

def calculate_tournament_history_score(name: str, history_df: pd.DataFrame) -> float:
    """
    Calculate a tournament history score based on past performance.
    Players with only old history (>3 years ago) are treated similar to new players.
    """
    # Get player's history
    player_history = history_df[history_df['Name'].apply(fix_names) == fix_names(name)]
    
    # Get median score from players with recent history
    def has_recent_history(player):
        recent_years = ['24', '2022-23', '2021-22']  # Last 3 years
        for year in recent_years:
            if year in history_df.columns and pd.notna(player[year]):
                return True
        return False
    
    # Get players with recent history for median calculation
    players_with_recent = history_df[history_df.apply(has_recent_history, axis=1)]
    median_score = 0.0
    if len(players_with_recent) > 0:
        history_scores = []
        for _, player in players_with_recent.iterrows():
            temp_df = pd.DataFrame([player])
            score = calculate_tournament_history_score_internal(temp_df.iloc[0], history_df)
            history_scores.append(score)
        median_score = np.median(history_scores)
    
    # If player has no history or only old history, return median
    if len(player_history) == 0 or player_history['measured_years'].iloc[0] == 0:
        return median_score
        
    # Check if player has only old history
    has_only_old_history = True
    recent_years = ['24', '2022-23', '2021-22']  # Last 3 years
    for year in recent_years:
        if year in history_df.columns and pd.notna(player_history[year].iloc[0]):
            has_only_old_history = False
            break
    
    if has_only_old_history:
        return median_score
    
    return calculate_tournament_history_score_internal(player_history.iloc[0], history_df)

def calculate_tournament_history_score_internal(player_history: pd.Series, history_df: pd.DataFrame) -> float:
    """Internal function to calculate tournament history score for a player with history"""
    years = ['24', '2022-23', '2021-22', '2020-21', '2019-20']
    weights = [1.0, 0.8, 0.6, 0.4, 0.2]  # More recent years weighted higher
    
    finishes = []
    weighted_points = 0.0
    max_possible_points = 0.0
    appearances = 0
    
    # First pass: collect valid finishes and calculate base points
    for year, weight in zip(years, weights):
        if year in history_df.columns:
            finish = player_history[year]
            if pd.notna(finish):
                # Convert 'CUT' to 65
                if isinstance(finish, str) and finish.upper() == 'CUT':
                    finish = 65
                else:
                    try:
                        finish = float(finish)
                    except (ValueError, TypeError):
                        continue
                
                finishes.append(finish)
                appearances += 1
                
                # Convert finish position to points (1st = 100, 60th = 0)
                points = max(0, 100 - ((finish - 1) * (100/60)))
                weighted_points += points * weight
                max_possible_points += 100 * weight
    
    if appearances == 0:
        return 0.0
    
    # Calculate base score (40% of total)
    base_score = (weighted_points / max_possible_points) * 40 if max_possible_points > 0 else 0
    
    # Calculate consistency bonus (30% of total)
    consistency_bonus = 0.0
    if appearances > 1:
        # Calculate percentage of finishes in top positions
        top_10_finishes = sum(1 for finish in finishes if finish <= 10)
        top_20_finishes = sum(1 for finish in finishes if finish <= 20)
        
        top_10_pct = top_10_finishes / appearances
        top_20_pct = top_20_finishes / appearances
        
        # Bonus points for consistent good finishes
        consistency_bonus = (top_10_pct * 20) + (top_20_pct * 10)
        
        # Additional bonus for multiple appearances
        appearance_multiplier = min(1.5, 1 + (appearances - 1) * 0.15)  # Max 50% bonus for 4+ appearances
        consistency_bonus *= appearance_multiplier
    
    # Cap consistency bonus at 30 points
    consistency_bonus = min(30, consistency_bonus)
    
    # Calculate recency bonus (30% of total)
    recency_bonus = 0.0
    if len(finishes) > 0:
        recent_finishes = finishes[:2]  # Look at most recent 2 appearances
        if recent_finishes:
            avg_recent_finish = sum(recent_finishes) / len(recent_finishes)
            # Convert to points (1st = 30, 60th = 0)
            recency_bonus = max(0, 30 - ((avg_recent_finish - 1) * (30/60)))
    
    total_score = base_score + consistency_bonus + recency_bonus
    
    return total_score

def get_current_tuesday() -> datetime:
    """Get the date of the current week's Tuesday"""
    today = datetime.now()
    days_since_tuesday = (today.weekday() - 1) % 7  # Tuesday is 1
    tuesday = today - pd.Timedelta(days=days_since_tuesday)
    return tuesday

def calculate_fit_score_from_csv(names: pd.Series, course_fit_df: pd.DataFrame) -> pd.Series:
    """Calculate fit score from course_fit.csv data for all players at once"""
    # Find players in dataframe using vectorized operations
    max_rank = course_fit_df['projected_course_fit'].max()
    
    # Merge the data to get projected_course_fit for each name
    result = pd.merge(
        pd.DataFrame({'Name': names}),
        course_fit_df[['Name', 'projected_course_fit']],
        on='Name',
        how='left'
    )
    
    # Convert to 0-100 scale where 100 is best (lowest rank)
    # Handle NaN values by returning 0
    return result['projected_course_fit'].apply(
        lambda x: 100 * (1 - (x / max_rank)) if pd.notna(x) else 0
    )

def calculate_form_score(tourney: str, weights: dict) -> pd.DataFrame:
    """
    Calculate form score by merging current and long-term form data with weights
    
    Args:
        tourney: Tournament name
        weights: Dictionary containing form weights
    
    Returns:
        DataFrame with merged form data and final form score
    """
    # Load PGA stats (long-term form)
    pga_stats = pd.read_csv(f'2025/{tourney}/pga_stats.csv')
    
    # Initialize form DataFrame with long-term stats
    form_df = pga_stats[['Name', 'sg_total']].copy()
    form_df = form_df.rename(columns={'sg_total': 'long_term_form'})
    
    # Try to load current form data
    current_form_path = f'2025/{tourney}/current_form.csv'
    if os.path.exists(current_form_path):
        print("Loading and merging current form data...")
        current_form = pd.read_csv(current_form_path)
        
        # Calculate current form total
        sg_columns = ['sg_off_tee', 'sg_approach', 'sg_around_green', 'sg_putting']
        current_form['current_form'] = current_form[sg_columns].sum(axis=1)
        
        # Merge with form_df, using outer join to keep all players from both sources
        form_df = pd.merge(
            form_df, 
            current_form[['Name', 'current_form']], 
            on='Name', 
            how='outer'  # Changed from 'left' to 'outer'
        )
        
        # Fill NaN values with the other source if available
        form_df['current_form'] = form_df['current_form'].fillna(form_df['long_term_form'])
        form_df['long_term_form'] = form_df['long_term_form'].fillna(form_df['current_form'])
        
        # Calculate weighted form score
        # For players with only current form, use that exclusively
        form_df['Form Score'] = np.where(
            form_df['long_term_form'].isna(),
            form_df['current_form'],  # Use only current form if no long-term data
            form_df['current_form'] * weights['form']['current'] + 
            form_df['long_term_form'] * weights['form']['long']
        )
    else:
        print("Current form data not found, using PGA stats only")
        form_df['Form Score'] = form_df['long_term_form']
    
    # Clean up intermediate columns
    form_df = form_df[['Name', 'Form Score']]
    
    return form_df

def normalize_with_outlier_handling(series: pd.Series) -> pd.Series:
    """
    Normalize a series to 0-1 range while handling outliers using IQR method.
    
    Args:
        series: Series of values to normalize
    
    Returns:
        Normalized series with outliers handled
    """
    
    # Calculate Q1, Q3 and IQR
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    
    # Define bounds for outliers (using 1.5 * IQR rule)
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # Create a copy to avoid modifying original data
    clean_series = series.copy()
    
    # Replace outliers with bounds
    clean_series = clean_series.clip(lower=lower_bound, upper=upper_bound)
    
    # Now normalize the cleaned data
    min_val = clean_series.min()
    max_val = clean_series.max()
    
    if max_val == min_val:
        print("\nAll values are the same, returning neutral score")
        return pd.Series([0.832621864455015] * len(series))
    
    normalized = (clean_series - min_val) / (max_val - min_val)
    
    return normalized

def main(tourney: str, num_lineups: int = 20, weights: dict = None):
    """
    Main function for PGA optimization
    
    Args:
        tourney: Tournament name
        num_lineups: Number of lineups to generate
        weights: Dictionary containing all weights
    """
    # Default weights if none provided
    default_weights = {
        'odds': {
            'winner': 0.35,
            'top5': 0.15,
            'top10': 0.2,
            'top20': 0.3
        },
        'form': {
            'current': 0.7,
            'long': 0.3
        },
        'components': {
            'odds': 0.1,
            'fit': 0.3,
            'history': 0.4,
            'form': 0.2
        }
    }
    
    weights = weights or default_weights
    
    global TOURNEY
    TOURNEY = tourney
    print(f"\n{'='*50}")
    print(f"Running optimization for {TOURNEY}")
    print(f"{'='*50}\n")

    create_pga_stats(TOURNEY)

    # Read all required data
    odds_df = pd.read_csv(f'2025/{TOURNEY}/odds.csv')
    dk_salaries = pd.read_csv(f'2025/{TOURNEY}/DKSalaries.csv')
    tourney_history = pd.read_csv(f'2025/{TOURNEY}/tournament_history.csv')
    course_fit_df = pd.read_csv(f'2025/{TOURNEY}/course_fit.csv')
    
    print(f"Loaded {len(odds_df)} players from odds data")
    print(f"Loaded {len(dk_salaries)} players from DraftKings data\n")
    
    # Clean up names in all dataframes
    odds_df['Name'] = odds_df['Name'].apply(fix_names)
    dk_salaries['Name'] = dk_salaries['Name'].apply(fix_names)
    tourney_history['Name'] = tourney_history['Name'].apply(fix_names)
    course_fit_df['Name'] = course_fit_df['Name'].apply(fix_names)
    
    # Merge all data together
    dk_data = pd.merge(dk_salaries, odds_df, on='Name', how='left')
    dk_data = pd.merge(
        dk_data,
        tourney_history[['Name', 'history_score']],
        on='Name',
        how='left'
    )
    dk_data = pd.merge(
        dk_data,
        course_fit_df[['Name', 'projected_course_fit']],
        on='Name',
        how='left'
    )
    
    # Fill NaN values with 0 for history score and convert course fit to 0-100 scale
    dk_data['history_score'] = dk_data['history_score'].fillna(0)
    dk_data['Fit Score'] = calculate_fit_score_from_csv(dk_data['Name'], course_fit_df)
    
    print(f"After merging: {len(dk_data)} players\n")
    
    # Calculate odds total using provided weights
    dk_data['Tournament Winner'] = dk_data['Tournament Winner'].apply(
        lambda x: odds_to_score(x, "Tournament Winner", w=weights['odds']['winner']))
    dk_data['Top 5 Finish'] = dk_data['Top 5 Finish'].apply(
        lambda x: odds_to_score(x, "Top 5 Finish", t5=weights['odds']['top5']))
    dk_data['Top 10 Finish'] = dk_data['Top 10 Finish'].apply(
        lambda x: odds_to_score(x, "Top 10 Finish", t10=weights['odds']['top10']))
    dk_data['Top 20 Finish'] = dk_data['Top 20 Finish'].apply(
        lambda x: odds_to_score(x, "Top 20 Finish", t20=weights['odds']['top20']))
    
    # Fill NaN values with 0 for each odds column
    odds_columns = ['Tournament Winner', 'Top 5 Finish', 'Top 10 Finish', 'Top 20 Finish']
    for col in odds_columns:
        dk_data[col] = dk_data[col].fillna(0)
    
    dk_data['Odds Total'] = (
        dk_data['Tournament Winner'] +
        dk_data['Top 5 Finish'] + 
        dk_data['Top 10 Finish'] +
        dk_data['Top 20 Finish']
    )

    print("Sample of odds calculations:")
    print(dk_data[['Name', 'Tournament Winner', 'Top 5 Finish', 'Top 10 Finish', 'Top 20 Finish', 'Odds Total']]
          .sort_values('Odds Total', ascending=False)
          .head(5)
          .to_string())
    print("\n")

    # Create golfers from DraftKings data
    golfers = [Golfer(row) for _, row in dk_data.iterrows()]

    # Calculate form score
    form_df = calculate_form_score(tourney, weights)
    dk_data = pd.merge(dk_data, form_df, on='Name', how='left')
    dk_data['Form Score'] = dk_data['Form Score'].fillna(0)

    # Normalize all components
    dk_data['Normalized Odds'] = (dk_data['Odds Total'] - dk_data['Odds Total'].min()) / \
        (dk_data['Odds Total'].max() - dk_data['Odds Total'].min())
    dk_data['Normalized Fit'] = (dk_data['Fit Score'] - dk_data['Fit Score'].min()) / \
        (dk_data['Fit Score'].max() - dk_data['Fit Score'].min())
    dk_data['Normalized History'] = (dk_data['history_score'] - dk_data['history_score'].min()) / \
        (dk_data['history_score'].max() - dk_data['history_score'].min())
    dk_data['Normalized Form'] = normalize_with_outlier_handling(dk_data['Form Score'])

    # Calculate Total using all components
    dk_data['Total'] = (
        dk_data['Normalized Odds'] * weights['components']['odds'] +
        dk_data['Normalized Fit'] * weights['components']['fit'] +
        dk_data['Normalized History'] * weights['components']['history'] +
        dk_data['Normalized Form'] * weights['components']['form']
    )

    dk_data['Value'] = dk_data['Total'] / dk_data['Salary'] * 100000

    print("Top 5 Players by Value:")
    print("-" * 80)
    print(dk_data[['Name', 'Salary', 'Odds Total', 'Fit Score', 'history_score', 'Total', 'Value']]
          .sort_values(by='Value', ascending=False)
          .head(5)
          .to_string())
    print("\n")

    print("Top 5 Players by Total Points:")
    print("-" * 80) 
    print(dk_data[['Name', 'Salary', 'Odds Total', 'Fit Score', 'history_score', 'Total', 'Value']]
          .sort_values(by='Total', ascending=False)
          .head(5)
          .to_string())
    print("\n")
    # Generate optimized lineups
    print(f"Generating {num_lineups} optimized lineups...")
    optimized_lineups = optimize_dk_lineups(dk_data, num_lineups)
    
    # Save the lineups
    output_path = f"2025/{TOURNEY}/dk_lineups_optimized.csv"
    optimized_lineups.to_csv(output_path, index=False)
    print(f"Generated {len(optimized_lineups)} optimized lineups")
    print(f"Saved to: {output_path}\n")
    
    # Print sample lineup
    print("Top Lineup:")
    print("-" * 80)
    sample_lineup = optimized_lineups.iloc[0]
    total_salary = sum(dk_data[dk_data['Name + ID'] == player]['Salary'].iloc[0] for player in sample_lineup[:6])
    total_points = sum(dk_data[dk_data['Name + ID'] == player]['Total'].iloc[0] for player in sample_lineup[:6])
    
    print("Players:")
    for i, player in enumerate(sample_lineup[:6], 1):
        salary = dk_data[dk_data['Name + ID'] == player]['Salary'].iloc[0]
        points = dk_data[dk_data['Name + ID'] == player]['Total'].iloc[0]
        print(f"{i}. {player:<30} ${salary:,}  {points:.2f}pts")
    print(f"\nTotal Salary: ${total_salary:,}")
    print(f"Total Points: {total_points:.2f}")
    
    # Save detailed player data
    output_data_path = f"2025/{TOURNEY}/player_data.csv"
    columns_to_save = [
        'Name', 'Salary', 'Odds Total', 'Normalized Odds',
        'Fit Score', 'Normalized Fit',
        'history_score', 'Normalized History',
        'Form Score', 'Normalized Form',
        'Total', 'Value'
    ]
    dk_data[columns_to_save].sort_values('Total', ascending=False).to_csv(output_data_path, index=False)
    print(f"Saved detailed player data to: {output_data_path}")
    return optimized_lineups


if __name__ == "__main__":
    main(TOURNEY)