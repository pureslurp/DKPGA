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
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial

# Optimize pandas settings
pd.options.mode.chained_assignment = None  # default='warn'
pd.options.compute.use_bottleneck = True
pd.options.compute.use_numexpr = True

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

TOURNEY = "RBC_Heritage"

def odds_to_score(col, header, w=1, t5=1, t10=1, t20=1):
    '''
    win:  30 pts
    top 5: 14
    top 10: 7
    top 20: 5
    
    Returns 0 if column is None, NaN, or otherwise invalid
    '''
    # Return 0 if column is None, NaN, or invalid
    if pd.isna(col) or col is None:
        return 0
    
    try:
        if col < 0:
            final_line = (col/-110)
        else:
            final_line = (100/col)
    except (TypeError, ValueError):
        return 0
        
    # Only calculate scores for columns that exist in the data
    match header:
        case "Tournament Winner":
            return round(final_line * 30 * w, 3)
        case "Top 5 Finish":
            return round(final_line * 14 * t5, 3)
        case "Top 10 Finish":
            return round(final_line * 7 * t10, 3)
        case "Top 20 Finish":
            return round(final_line * 5 * t20, 3)
        case _:
            return 0

class DKLineupOptimizer:
    def __init__(self, salary_cap: int = 50000, lineup_size: int = 6):
        self.salary_cap = salary_cap
        self.lineup_size = lineup_size
        
    def generate_lineups(self, df: pd.DataFrame, num_lineups: int = 20, 
                        min_salary: int = 49000, exposure_limit: float = 0.66,
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
            
            # Constraint 4: Must have at least one player over $9800 and no more than 1 player over $10000
            expensive_players = [p for p in players if p['Salary'] >= 9800]
            very_expensive_players = [p for p in players if p['Salary'] >= 10000]
            if expensive_players:
                prob += pulp.lpSum([decisions[p['Name + ID']] for p in expensive_players]) >= 1
            if very_expensive_players:
                prob += pulp.lpSum([decisions[p['Name + ID']] for p in very_expensive_players]) <= 1
            
            # Constraint 5: No players below lowest salary tier
            cheap_threshold = 6400 if min_salary >= 6000 else 6000
            cheap_players = [p for p in players if p['Salary'] <= cheap_threshold]
            if cheap_players:
                prob += pulp.lpSum([decisions[p['Name + ID']] for p in cheap_players]) <= 1
            
            # # Constraint: Maximum two players above $9000
            # expensive_players = [p for p in players if p['Salary'] >= 9000]
            # if expensive_players:
            #     prob += pulp.lpSum([decisions[p['Name + ID']] for p in expensive_players]) <= 2
            
            # # Constraint: Minimum two players between $7000-$8900
            # mid_tier_players = [p for p in players if 7000 <= p['Salary'] <= 8900]
            # if mid_tier_players:
            #     prob += pulp.lpSum([decisions[p['Name + ID']] for p in mid_tier_players]) >= 2
            
            # # Constraint: Maximum three players under $7000
            # value_players = [p for p in players if p['Salary'] < 7000]
            # if value_players:
            #     prob += pulp.lpSum([decisions[p['Name + ID']] for p in value_players]) <= 3
            
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
                        original_salary = lineup[pos]['Salary']
                        
                        # Check if this is the highest-salary player and if single swap is impossible
                        is_highest_salary = all(lineup[p]['Salary'] <= original_salary 
                                             for p in ['G1', 'G2', 'G3', 'G4', 'G5', 'G6'])
                        next_highest_salary = max(r['Salary'] for r in replacement_pool 
                                               if r['Name + ID'] not in used_players)
                        salary_gap = original_salary - next_highest_salary
                        
                        # If this is highest salary player and gap makes single swap impossible
                        if is_highest_salary and salary_gap > 500:
                            # Find the lowest-salary player in the lineup
                            lowest_pos = min(
                                [p for p in ['G1', 'G2', 'G3', 'G4', 'G5', 'G6'] if p != pos],
                                key=lambda p: lineup[p]['Salary']
                            )
                            lowest_salary = lineup[lowest_pos]['Salary']
                            
                            # Try to find two replacements that work
                            valid_pairs = []
                            for r1 in replacement_pool:
                                if (player_counts.get(r1['Name + ID'], 0) >= max_appearances or 
                                    r1['Name + ID'] in used_players):
                                    continue
                                    
                                for r2 in replacement_pool:
                                    if (r2['Name + ID'] == r1['Name + ID'] or 
                                        player_counts.get(r2['Name + ID'], 0) >= max_appearances or 
                                        r2['Name + ID'] in used_players):
                                        continue
                                        
                                    new_salary = (lineup['Salary'] - original_salary - lowest_salary + 
                                                r1['Salary'] + r2['Salary'])
                                    
                                    if 49500 <= new_salary <= 50000:
                                        valid_pairs.append((r1, r2))
                            
                            if valid_pairs:
                                # Sort pairs by combined total points
                                valid_pairs.sort(key=lambda x: x[0]['Total'] + x[1]['Total'], reverse=True)
                                r1, r2 = valid_pairs[0]
                                
                                # Make the swaps
                                player_counts[player] -= 1
                                player_counts[lineup[lowest_pos]['Name + ID']] -= 1
                                player_counts[r1['Name + ID']] = player_counts.get(r1['Name + ID'], 0) + 1
                                player_counts[r2['Name + ID']] = player_counts.get(r2['Name + ID'], 0) + 1
                                
                                lineup[pos] = r1
                                lineup[lowest_pos] = r2
                                lineup['Salary'] = (lineup['Salary'] - original_salary - lowest_salary + 
                                                  r1['Salary'] + r2['Salary'])
                                lineup['TotalPoints'] = sum(lineup[f'G{i+1}']['Total'] for i in range(6))
                                used_players.add(r1['Name + ID'])
                                used_players.add(r2['Name + ID'])
                                changes_made = True
                                continue
                        
                        # If double-swap not needed or failed, try regular single replacement
                        current_salary = lineup['Salary'] - original_salary
                        valid_replacements = [
                            r for r in replacement_pool
                            if (49500 <= (current_salary + r['Salary']) <= 50000 and
                                player_counts.get(r['Name + ID'], 0) < max_appearances and
                                r['Name + ID'] not in used_players)
                        ]
                        
                        if valid_replacements:
                            replacement = max(valid_replacements, key=lambda x: x['Total'])
                            replacement_id = replacement['Name + ID']
                            
                            player_counts[player] -= 1
                            player_counts[replacement_id] = player_counts.get(replacement_id, 0) + 1
                            lineup[pos] = replacement
                            lineup['Salary'] = current_salary + replacement['Salary']
                            lineup['TotalPoints'] = sum(lineup[f'G{i+1}']['Total'] for i in range(6))
                            used_players.add(replacement_id)
                            changes_made = True
                        else:
                            print(f"Warning: Could not find valid replacement for {player} in lineup")
                
                if changes_made:
                    # Re-sort lineups after modifications
                    lineups.sort(key=lambda x: x['TotalPoints'])
        
        # Final sort by total points (descending)
        lineups.sort(key=lambda x: x['TotalPoints'], reverse=True)
        return lineups

def optimize_dk_lineups(dk_merge: pd.DataFrame, num_lineups: int = 20) -> pd.DataFrame:
    """
    Main function to generate optimized DraftKings lineups
    """
    # Convert DataFrame to dictionary once, outside the loop
    players = dk_merge.to_dict('records')
    
    # Create the optimizer and generate lineups
    optimizer = DKLineupOptimizer()
    lineups = optimizer.generate_lineups(dk_merge, num_lineups)
    
    # Use list comprehension instead of appending in a loop
    lineup_rows = [{
        'G1': lineup['G1']['Name + ID'],
        'G2': lineup['G2']['Name + ID'],
        'G3': lineup['G3']['Name + ID'],
        'G4': lineup['G4']['Name + ID'],
        'G5': lineup['G5']['Name + ID'],
        'G6': lineup['G6']['Name + ID'],
        'Salary': lineup['Salary'],
        'TotalPoints': lineup['TotalPoints']
    } for lineup in lineups]
    
    return pd.DataFrame(lineup_rows)

def calculate_tournament_history_score(name: str, history_df: pd.DataFrame) -> float:
    """Calculate tournament history score with caching"""
    # Use class-level cache for fixed names
    if not hasattr(calculate_tournament_history_score, 'name_cache'):
        calculate_tournament_history_score.name_cache = {}
    
    # Cache the player history lookup
    if name not in calculate_tournament_history_score.name_cache:
        fixed_name = fix_names(name)
        calculate_tournament_history_score.name_cache[name] = fixed_name
    else:
        fixed_name = calculate_tournament_history_score.name_cache[name]
    
    player_history = history_df[
        history_df['Name'].map(calculate_tournament_history_score.name_cache) == fixed_name
    ]
    
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
    recent_years = ['24', '2022-23', '2021-22']
    old_years = ['2020-21', '2019-20']
    weights = [1.0, 0.8, 0.6, 0.4, 0.2]
    
    
    finish_points = 0.0
    appearances = 0
    total_possible = 0.0
    good_finishes = 0
    
    for year, weight in zip(years, weights):
        if year in history_df.columns:
            finish = player_history[year]
            if pd.notna(finish):
                # Count appearance
                appearances += 1
                
                # Convert finish to points and count good finishes
                if isinstance(finish, str) and finish.upper() == 'CUT':
                    points = 0
                else:
                    try:
                        finish = float(finish)
                        # Count good finishes (top 15)
                        if finish <= 15:
                            good_finishes += 1
                            
                        if finish <= 50:
                            points = max(0, 100 - ((finish - 1) * (100/50)))
                        else:
                            points = 0
                    except (ValueError, TypeError):
                        continue
                
                finish_points += points * weight
                total_possible += 100 * weight  # Add full weight for played years
            else:
                # Add penalty weight to denominator for missing years
                total_possible += 100 * (weight * 0.15)
    
    # Calculate base finish score
    base_finish_score = (finish_points / total_possible * 50) if total_possible > 0 else 0
    
    # # After calculating base_finish_score
    # if appearances == 1:
    #     base_finish_score = base_finish_score * 0.95  # 5% reduction for single appearance
    
    # Apply bonus for multiple good finishes
    if good_finishes >= 2:
        bonus = min(good_finishes, 3) * 0.1
        finish_score = base_finish_score * (1 + bonus)
    else:
        finish_score = base_finish_score
    
    # Calculate SG score (max 30 points)
    try:
        sg_val = player_history['sg_total']
        rounds_played = player_history['rounds']
        
        if pd.notna(sg_val) and pd.notna(rounds_played):
            # Calculate base SG score
            base_sg_score = min(100, max(0, (sg_val / 3) * 100)) * 0.3
            
            # Apply rounds played multiplier
            if rounds_played <= 4:
                rounds_multiplier = 0.75  # 75% for 4 rounds or less
            elif rounds_played <= 8:
                # Linear interpolation between 75% and 100% for 4-8 rounds
                rounds_multiplier = 0.75 + (0.25 * ((rounds_played - 4) / 4))
            else:
                rounds_multiplier = 1.0  # Full score for 8+ rounds
            
            sg_score = base_sg_score * rounds_multiplier
        else:
            sg_score = 0.0
    except:
        sg_score = 0.0
    
    # Calculate consistency score (max 20 points)
    consistency_score = 0
    if appearances > 0:
        # Calculate made cuts percentage with reduced penalty for older missed cuts
        recent_cuts = 0
        recent_total = 0
        old_cuts = 0
        old_total = 0
        
        for year in recent_years:
            if year in history_df.columns and pd.notna(player_history[year]):
                recent_total += 1
                if not (isinstance(player_history[year], str) and player_history[year].upper() == 'CUT'):
                    recent_cuts += 1
        
        for year in old_years:
            if year in history_df.columns and pd.notna(player_history[year]):
                old_total += 1
                if not (isinstance(player_history[year], str) and player_history[year].upper() == 'CUT'):
                    old_cuts += 1
        
        # Calculate made cuts percentage with old cuts counting as 0.5 cuts made
        total_weighted_cuts = recent_cuts + (old_cuts + (old_total - old_cuts) * 0.5)
        total_rounds = recent_total + old_total
        
        if total_rounds > 0:
            made_cuts_pct = total_weighted_cuts / total_rounds
            max_consistency_points = 15 if appearances == 1 else (17.5 if appearances == 2 else 20)
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

# def calculate_form_score(tourney: str, weights: dict) -> pd.DataFrame:
#     """
#     Calculate form score by merging current and long-term form data with weights
    
#     Args:
#         tourney: Tournament name
#         weights: Dictionary containing form weights
    
#     Returns:
#         DataFrame with merged form data and final form score
#     """
#     # Load PGA stats (long-term form)
#     pga_stats = pd.read_csv(f'2025/{tourney}/pga_stats.csv')
    
#     # Initialize form DataFrame with long-term stats
#     form_df = pga_stats[['Name', 'sg_total']].copy()
#     form_df = form_df.rename(columns={'sg_total': 'long_term_form'})
    
#     # Try to load current form data
#     current_form_path = f'2025/{tourney}/current_form.csv'
#     if os.path.exists(current_form_path):
#         print("Loading and merging current form data...")
#         current_form = pd.read_csv(current_form_path)
        
#         # Calculate current form total
#         sg_columns = ['sg_off_tee', 'sg_approach', 'sg_around_green', 'sg_putting']
#         finishes = current_form['recent_finishes'].apply(_parse_recent_finishes)
#         current_form['current_form'] = current_form[sg_columns].sum(axis=1)

        
#         # Merge with form_df, using outer join to keep all players from both sources
#         form_df = pd.merge(
#             form_df, 
#             current_form[['Name', 'current_form']], 
#             on='Name', 
#             how='outer'  # Changed from 'left' to 'outer'
#         )
        
#         # Fill NaN values with the other source if available
#         form_df['current_form'] = form_df['current_form'].fillna(form_df['long_term_form'])
#         form_df['long_term_form'] = form_df['long_term_form'].fillna(form_df['current_form'])
        
#         # Calculate weighted form score
#         # For players with only current form, use that exclusively
#         form_df['Form Score'] = np.where(
#             form_df['long_term_form'].isna(),
#             form_df['current_form'],  # Use only current form if no long-term data
#             form_df['current_form'] * weights['form']['current'] + 
#             form_df['long_term_form'] * weights['form']['long']
#         )
#     else:
#         print("Current form data not found, using PGA stats only")
#         form_df['Form Score'] = form_df['long_term_form']
    
#     # Clean up intermediate columns
#     form_df = form_df[['Name', 'Form Score']]
    
#     return form_df

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

def calculate_scores_parallel(golfers, tourney_history, course_fit_df, tourney: str, weights: dict):
    """Calculate fit, history, and form scores in parallel"""
    # Load data once before parallel processing
    pga_stats = pd.read_csv(f'2025/{tourney}/pga_stats.csv')
    current_form_df = None
    current_form_path = f'2025/{tourney}/current_form.csv'
    if os.path.exists(current_form_path):
        current_form_df = pd.read_csv(current_form_path)

    def process_golfer(golfer, pga_stats, current_form_df):  # Add parameters here
        history_score = calculate_tournament_history_score(golfer.get_clean_name, tourney_history)
        fit_score = calculate_fit_score_from_csv(golfer.get_clean_name, course_fit_df)
        
        # Calculate form score for this golfer
        player_stats = pga_stats[pga_stats['Name'].apply(fix_names) == golfer.get_clean_name]
        long_term_form = player_stats['sg_total'].iloc[0] if not player_stats.empty else 0

        def _parse_recent_finishes(recent_finishes: str) -> float:
            '''Convert recent finishes (example: [20, 15, 25, 12, 2]) to points'''
            # Remove brackets and split by comma
            finishes_str = recent_finishes.strip('[]').split(',')
            
            # Convert each finish to points
            total_points = 0
            valid_finishes = 0
            cuts = 0
            
            for finish in finishes_str:
                finish = finish.strip().strip("'").strip('"')  # Remove quotes and whitespace
                if finish == 'CUT':
                    total_points += 0
                    valid_finishes += 1
                elif finish != 'None':
                    try:
                        position = int(finish)
                        # Points system: 100 for 1st, scaling down to 0 for 50th or worse
                        points = max(0, 100 - ((position - 1) * (100/65)))
                        total_points += points
                        valid_finishes += 1
                    except ValueError:
                        continue
            try:
                consistency_score = 1 - (cuts / valid_finishes)
            except:
                consistency_score = 0
            # Return average points if there are valid finishes, otherwise 0
            return total_points / valid_finishes if valid_finishes > 0 else 0, consistency_score
        
        current_form = 0
        if current_form_df is not None:
            player_data = current_form_df[current_form_df['Name'].apply(fix_names) == golfer.get_clean_name]
            if len(player_data) > 0:
                sg_columns = ['sg_off_tee', 'sg_approach', 'sg_around_green', 'sg_putting']
                # Get all three raw scores
                sg_total = player_data[sg_columns].sum(axis=1).iloc[0]
                recent_finishes_score, consistency_score = player_data['recent_finishes'].apply(_parse_recent_finishes).iloc[0]
                
                # Get min/max values from current dataset for sg_total
                min_sg = current_form_df[sg_columns].sum(axis=1).min()
                max_sg = current_form_df[sg_columns].sum(axis=1).max()
                normalized_sg = (sg_total - min_sg) / (max_sg - min_sg) if max_sg != min_sg else 0.5
                
                # Get min/max for recent finishes scores
                all_finishes = current_form_df['recent_finishes'].apply(_parse_recent_finishes)
                min_recent = min(score for score, _ in all_finishes)
                max_recent = max(score for score, _ in all_finishes)
                normalized_recent = (recent_finishes_score - min_recent) / (max_recent - min_recent) if max_recent != min_recent else 0.5
                
                # Combine all three scores with weights
                current_form = (normalized_sg * 0.4 + 
                              normalized_recent * 0.4 + 
                              consistency_score * 0.2)

        # Normalize long-term form
        min_long_term = pga_stats['sg_total'].min()
        max_long_term = pga_stats['sg_total'].max()
        normalized_long_term = ((long_term_form - min_long_term) / 
                              (max_long_term - min_long_term)) if max_long_term != min_long_term else 0.5

        # Calculate weighted form score using normalized values
        form_score = (current_form * weights['form']['current'] + 
                     normalized_long_term * weights['form']['long'])
        
        return {
            'Name': golfer.get_clean_name,
            'History Score': history_score,
            'Fit Score': fit_score,
            'Form Score': form_score
        }
    
    # Create a partial function with the loaded data
    process_golfer_with_data = partial(process_golfer, 
                                     pga_stats=pga_stats, 
                                     current_form_df=current_form_df)
    
    with ThreadPoolExecutor(max_workers=min(32, len(golfers))) as executor:
        futures = [executor.submit(process_golfer_with_data, golfer) for golfer in golfers]
        results = [f.result() for f in as_completed(futures)]
    
    return pd.DataFrame(results)

def main(tourney: str, num_lineups: int = 20, weights: dict = None, exclude_golfers: List[str] = None):
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
            'odds': 0.2,
            'fit': 0.5,
            'history': 0.3,
            'form': 0.0
        }
    }
    
    weights = weights or default_weights
    exclude_golfers = [fix_names(name) for name in (exclude_golfers or [])]
    
    if exclude_golfers:
        print(f"\nExcluding golfers: {', '.join(exclude_golfers)}")
    
    global TOURNEY
    TOURNEY = tourney
    print(f"\n{'='*50}")
    print(f"Running optimization for {TOURNEY}")
    print(f"{'='*50}\n")

    # TODO: make pga_v5 all encompasing for getting csv files
    # Replace stats file creation logic with:
    create_pga_stats(TOURNEY)

    # Load all data at once
    data_files = {
        'odds': f'2025/{tourney}/odds.csv',
        'dk_salaries': f'2025/{tourney}/DKSalaries.csv',
        'course_fit': f'2025/{tourney}/course_fit.csv'
    }
    
    # Use dictionary comprehension for parallel loading
    dfs = {
        name: pd.read_csv(path) 
        for name, path in data_files.items()
    }
    
    # Pre-process names once
    for df in dfs.values():
        if 'Name' in df.columns:
            df['Name'] = df['Name'].apply(fix_names)
    
    # Read odds data and DraftKings salaries
    odds_df = dfs['odds']
    dk_salaries = dfs['dk_salaries']
    
    try:
        # Try tournament history first
        tourney_history = pd.read_csv(f'2025/{TOURNEY}/tournament_history.csv')
    except FileNotFoundError:
        try:
            # Fall back to course history
            tourney_history = pd.read_csv(f'2025/{TOURNEY}/course_history.csv')
            print(f"Tournament history not found, using course history data instead")
        except FileNotFoundError:
            print(f"No history data found. Setting history scores to 0.")
            # Create empty history DataFrame with required columns
            tourney_history = pd.DataFrame(columns=['Name', 'measured_years', 'made_cuts_pct'])
            for year in ['24', '2022-23', '2021-22', '2020-21', '2019-20']:
                tourney_history[year] = None
    
    print(f"Loaded {len(odds_df)} players from odds data")
    print(f"Loaded {len(dk_salaries)} players from DraftKings data\n")
    
    # Merge odds with DraftKings data
    dk_data = pd.merge(dk_salaries, odds_df, on='Name', how='left')

    # Calculate odds total using provided weights
    odds_columns = {
        'Tournament Winner': weights['odds']['winner'],
        'Top 5 Finish': weights['odds']['top5'],
        'Top 10 Finish': weights['odds']['top10'],
        'Top 20 Finish': weights['odds']['top20']
    }
    
    # Initialize odds columns with 0
    for col in odds_columns:
        if col not in dk_data.columns:
            dk_data[col] = 0
        else:
            dk_data[col] = dk_data[col].apply(
                lambda x: odds_to_score(x, col, w=odds_columns[col]))
    
    # Calculate Odds Total
    dk_data['Odds Total'] = (
        dk_data['Tournament Winner'] +
        dk_data['Top 5 Finish'] + 
        dk_data['Top 10 Finish'] +
        dk_data['Top 20 Finish']
    )

    # Apply logarithmic transformation before normalization to compress the range
    dk_data['Normalized Odds'] = np.log1p(dk_data['Odds Total'])
    dk_data['Normalized Odds'] = (dk_data['Normalized Odds'] - dk_data['Normalized Odds'].min()) / \
        (dk_data['Normalized Odds'].max() - dk_data['Normalized Odds'].min())

    # Alternative approach using softmax-inspired normalization
    # temperature = 0.3  # Adjust this value to control the spread (lower = more compressed)
    # dk_data['Normalized Odds'] = np.exp(dk_data['Odds Total'] / temperature)
    # dk_data['Normalized Odds'] = dk_data['Normalized Odds'] / dk_data['Normalized Odds'].max()

    # Create golfers from DraftKings data
    golfers = [Golfer(row) for _, row in dk_data.iterrows()]

    # Load course fit data
    print(f"Loading course fit data for {TOURNEY}...")
    course_fit_df = dfs['course_fit']
    print("Course fit data loaded successfully\n")

    # Calculate scores in parallel
    print(f"Calculating scores in parallel for {len(golfers)} golfers...")
    scores_df = calculate_scores_parallel(golfers, tourney_history, course_fit_df, TOURNEY, weights)
    
    # Merge scores with dk_data
    dk_data = pd.merge(dk_data, scores_df, on='Name', how='left')
    dk_data['Form Score'] = dk_data['Form Score'].fillna(0)
    
    # Normalize all components
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

    # Save the full player data before exclusions
    columns_to_save = [
        'Name', 'Salary', 'Odds Total', 'Normalized Odds',
        'Fit Score', 'Normalized Fit',
        'History Score', 'Normalized History',
        'Form Score', 'Normalized Form',
        'Total', 'Value'
    ]
    dk_data[columns_to_save].sort_values('Total', ascending=False).to_csv(f"2025/{tourney}/player_data.csv", index=False)
    print(f"Saved detailed player data to: 2025/{tourney}/player_data.csv")

    # Apply exclusions only for lineup optimization
    if exclude_golfers:
        print(f"\nExcluding golfers from lineups: {', '.join(exclude_golfers)}")
        dk_data = dk_data[~dk_data['Name'].isin(exclude_golfers)]

    print(f"Players available for lineups: {len(dk_data)}\n")

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
    
    # # Save detailed player data
    # output_data_path = f"2025/{TOURNEY}/player_data.csv"
    # columns_to_save = [
    #     'Name', 'Salary', 'Odds Total', 'Normalized Odds',
    #     'Fit Score', 'Normalized Fit',
    #     'History Score', 'Normalized History',
    #     'Form Score', 'Normalized Form',
    #     'Total', 'Value'
    # ]
    # dk_data[columns_to_save].sort_values('Total', ascending=False).to_csv(output_data_path, index=False)
    # print(f"Saved detailed player data to: {output_data_path}")
    return optimized_lineups


if __name__ == "__main__":
    main(TOURNEY)