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

TOURNEY = "The_Genesis_Invitational"

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
    weights = [1.0, 0.8, 0.6, 0.4, 0.2]
    
    # Calculate perfect score benchmark
    perfect_score = 95
    
    finish_points = 0.0
    total_weight = 0.0
    appearances = 0
    good_finishes = 0
    
    for year, weight in zip(years, weights):
        if year in history_df.columns:
            finish = player_history[year]
            if pd.notna(finish):
                if isinstance(finish, str) and finish.upper() == 'CUT':
                    finish = 65
                else:
                    try:
                        finish = float(finish)
                        if finish <= 15:
                            good_finishes += 1
                    except (ValueError, TypeError):
                        continue
                
                appearances += 1
                
                if finish <= 60:
                    points = max(0, 100 - ((finish - 1) * (100/60)))
                else:
                    points = 0
                
                finish_points += points * weight
                total_weight += weight
    
    # Calculate base finish score normalized to our perfect score benchmark
    base_finish_score = ((finish_points / total_weight) / perfect_score * 50) if total_weight > 0 else 0
    
    # After calculating base_finish_score
    if appearances == 1:
        base_finish_score = base_finish_score * 0.95  # 5% reduction for single appearance
    
    # Apply bonus for multiple good finishes
    if good_finishes >= 2:
        bonus = min(good_finishes - 1, 3) * 0.1
        finish_score = base_finish_score * (1 + bonus)
    else:
        finish_score = base_finish_score
    
    # Calculate SG score (max 30 points)
    sg_stats = ['sg_ott', 'sg_app', 'sg_atg', 'sg_putting']
    sg_weights = [0.25, 0.35, 0.25, 0.15]
    
    try:
        sg_score = 0.0
        valid_sg_stats = 0
        for stat, weight in zip(sg_stats, sg_weights):
            if stat in history_df.columns:
                sg_val = player_history[stat]
                if pd.notna(sg_val):
                        normalized_sg = min(100, max(0, (sg_val + 2) * 25))
                        sg_score += normalized_sg * weight
                        valid_sg_stats += 1
    except:
        sg_score = 0.0
        valid_sg_stats = 0
    
    sg_score = sg_score * 0.3 if valid_sg_stats > 0 else 0
    
    # Calculate consistency score (max 20 points)
    consistency_score = 0
    if appearances > 0:
        made_cuts_pct = player_history['made_cuts_pct'] if 'made_cuts_pct' in history_df.columns else 0
        max_consistency_points = 15 if appearances == 1 else 20
        consistency_score = made_cuts_pct * max_consistency_points
    
    total_score = finish_score + sg_score + consistency_score
    
    return min(100, total_score)

def get_current_tuesday() -> datetime:
    """Get the date of the current week's Tuesday"""
    today = datetime.now()
    days_since_tuesday = (today.weekday() - 1) % 7  # Tuesday is 1
    tuesday = today - pd.Timedelta(days=days_since_tuesday)
    return tuesday

def calculate_fit_score_from_csv(name: str, course_fit_df: pd.DataFrame) -> float:
    """
    Calculate fit score from course_fit.csv data.
    Uses 'fit_score' column if available, otherwise calculates from 'projected_course_fit'.
    """
    # Find player in dataframe
    player_data = course_fit_df[course_fit_df['Name'].apply(fix_names) == fix_names(name)]
    
    if len(player_data) == 0:
        return 0.0  # Return 0 if player not found
    
    # If fit_score column exists, use it directly
    if 'fit score' in course_fit_df.columns:
        return player_data['fit score'].iloc[0]
        
    # Otherwise, calculate from projected course fit (lower is better)
    fit_score = player_data['projected_course_fit'].iloc[0]
    
    # Convert to 0-100 scale where 100 is best (lowest rank)
    max_rank = course_fit_df['projected_course_fit'].max()
    return 100 * (1 - (fit_score / max_rank))

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
        tournament_history: Whether to include tournament history
        weights: Dictionary containing all weights with the following structure:
            {
                'odds': {
                    'winner': 0.6,
                    'top5': 0.5,
                    'top10': 0.8,
                    'top20': 0.4
                },
                'form': {
                    'current': 0.7,  # Short-term form weight
                    'long': 0.3      # Long-term form weight
                },
                'components': {
                    'odds': 0.4,
                    'fit': 0.3,
                    'history': 0.2,
                    'form': 0.1
                }
            }
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
            'fit': 0.4,
            'history': 0.4,
            'form': 0.1
        }
    }
    
    weights = weights or default_weights
    
    global TOURNEY
    TOURNEY = tourney
    print(f"\n{'='*50}")
    print(f"Running optimization for {TOURNEY}")
    print(f"{'='*50}\n")

    # TODO: make pga_v5 all encompasing for getting csv files
    # Replace stats file creation logic with:
    create_pga_stats(TOURNEY)

    # Read odds data and DraftKings salaries
    odds_df = pd.read_csv(f'2025/{TOURNEY}/odds.csv')
    dk_salaries = pd.read_csv(f'2025/{TOURNEY}/DKSalaries.csv')
    
    # Try to read tournament history, fall back to course history if not available
    history_file = f'2025/{TOURNEY}/tournament_history.csv'
    if not os.path.exists(history_file):
        history_file = f'2025/{TOURNEY}/course_history.csv'
        print(f"Tournament history not found, using course history data instead")
    
    try:
        tourney_history = pd.read_csv(history_file)
    except FileNotFoundError:
        print(f"No history data found. Setting history scores to 0.")
        # Create empty history DataFrame with required columns
        tourney_history = pd.DataFrame(columns=['Name', 'measured_years', 'made_cuts_pct'])
        for year in ['24', '2022-23', '2021-22', '2020-21', '2019-20']:
            tourney_history[year] = None
    
    print(f"Loaded {len(odds_df)} players from odds data")
    print(f"Loaded {len(dk_salaries)} players from DraftKings data\n")
    
    # Clean up names in both dataframes
    odds_df['Name'] = odds_df['Name'].apply(fix_names)
    dk_salaries['Name'] = dk_salaries['Name'].apply(fix_names)
    
    # Merge odds with DraftKings data
    dk_data = pd.merge(dk_salaries, odds_df, on='Name', how='left')
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

    # Load course fit data
    print(f"Loading course fit data for {TOURNEY}...")
    course_fit_df = pd.read_csv(f'2025/{TOURNEY}/course_fit.csv')
    print("Course fit data loaded successfully\n")

    # Calculate fit scores and history scores for all golfers
    fit_scores_data = []
    history_scores = []
    for golfer in golfers:
        history_score = calculate_tournament_history_score(golfer.get_clean_name, tourney_history)
            
        # Calculate fit score from CSV data
        fit_score = calculate_fit_score_from_csv(golfer.get_clean_name, course_fit_df)
        golfer.set_fit_score(fit_score)
        
        fit_scores_data.append({
            'Name': golfer.get_clean_name,
            'Fit Score': fit_score
        })
        history_scores.append({
            'Name': golfer.get_clean_name,
            'History Score': history_score
        })
    
    history_scores_df = pd.DataFrame(history_scores)
    fit_scores_df = pd.DataFrame(fit_scores_data)
    form_df = calculate_form_score(tourney, weights)


    # Merge fit scores and history scores with dk_data
    dk_data = pd.merge(dk_data, fit_scores_df, on='Name', how='left')
    dk_data = pd.merge(dk_data, history_scores_df, on='Name', how='left')
    dk_data = pd.merge(dk_data, form_df, on='Name', how='left')
    dk_data['Form Score'] = dk_data['Form Score'].fillna(0)
    

    # Normalize all components
    dk_data['Normalized Odds'] = (dk_data['Odds Total'] - dk_data['Odds Total'].min()) / \
        (dk_data['Odds Total'].max() - dk_data['Odds Total'].min())
    dk_data['Normalized Fit'] = (dk_data['Fit Score'] - dk_data['Fit Score'].min()) / \
        (dk_data['Fit Score'].max() - dk_data['Fit Score'].min())
    dk_data['Normalized History'] = (dk_data['History Score'] - dk_data['History Score'].min()) / \
        (dk_data['History Score'].max() - dk_data['History Score'].min())
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
    print(dk_data[['Name', 'Salary', 'Odds Total', 'Fit Score', 'History Score', 'Total', 'Value']]
          .sort_values(by='Value', ascending=False)
          .head(5)
          .to_string())
    print("\n")

    print("Top 5 Players by Total Points:")
    print("-" * 80) 
    print(dk_data[['Name', 'Salary', 'Odds Total', 'Fit Score', 'History Score', 'Total', 'Value']]
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
        'History Score', 'Normalized History',
        'Form Score', 'Normalized Form',
        'Total', 'Value'
    ]
    dk_data[columns_to_save].sort_values('Total', ascending=False).to_csv(output_data_path, index=False)
    print(f"Saved detailed player data to: {output_data_path}")
    return optimized_lineups


if __name__ == "__main__":
    main(TOURNEY)