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
from io import StringIO
from course_info import CourseStats
import os
from utils import TOURNAMENT_LIST_2024
from course_info import load_course_stats

'''
This is the main script that runs the PGA model for DraftKings

Looking for this version to be more systematic:

The model will take into considereation the following:
- Tournament History ({tournament_name}/tournament_history.csv) -- TODO
   - PGATour.com has a new site that has past 5 (same) event finishes
- Course Attributes (dg_course_table.cvs) against Player Attributes -- DONE
   - Player Attributes
      - Strokes Gained (golfers/pga_stats_{date}.csv or golfers/current_form_{date}.csv)
         - Off the tee, Approach, Around the green, Putting
      - Driving Accuracy (golfers/pga_stats_{date}.csv)
      - Green in Regulation (golfers/pga_stats_{date}.csv)
      - Scrambling from Sand (golfers/pga_stats_{date}.csv)
   - Need to create a mapping table for course attributes to player attributes -- DONE
- Weighted Optimizion (weight_optimization_results.csv) -- DONE
   - Win, Top 5, Top 10, Top 20
   - Starting weights:
        Tournament Winner: 0.6
        Top 5: 0.5
        Top 10: 0.8
        Top 20: 0.4
- Robust Optimization (DKLineupOptimizer) to csv -- DONE
'''

def fix_names(name):
    if name == "Si Woo":
        return "si woo kim"
    elif name == "Byeong Hun":
        return "byeong hun an"
    elif name == "Erik Van":
        return "erik van rooyen"
    elif name == "Adrien Dumont":
        return "adrien dumont de chassart"
    elif name == "Matthias Schmid":
        return "matti schmid"
    elif name == "Samuel Stevens":
        return "sam stevens"
    elif name == "Benjamin Silverman":
        return "ben silverman"
    elif name =="Min Woo":
        return "min woo lee"
    elif name == "Santiago De":
        return "santiago de la fuente"
    elif name == "Jose Maria":
        return "jose maria olazabal"
    elif name == "Niklas Norgaard Moller":
        return "niklas moller"
    elif name == "Jordan L. Smith":
        return "jordan l."
    elif name == "daniel bradbury":
        return "dan bradbury"
    elif name == "Ludvig Åberg":
        return "ludvig aberg"
    elif name == "Cam Davis":
        return "cameron davis"
    elif name == "Nicolai Højgaard":
        return "nicolai hojgaard"
    else:
        return name.lower()

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

@dataclass
class StrokesGained:
    """Class to hold strokes gained statistics"""
    off_tee: float = 0.0
    approach: float = 0.0
    around_green: float = 0.0
    putting: float = 0.0

    @property
    def tee_to_green(self) -> float:
        """Calculate strokes gained tee to green"""
        return self.off_tee + self.approach + self.around_green
    
    @property
    def total(self) -> float:
        """Calculate total strokes gained"""
        return self.off_tee + self.approach + self.around_green + self.putting

class Golfer:
    def __init__(self, golfer: pd.DataFrame):
        # Original DraftKings attributes
        try:
            self.name = golfer["Name + ID"].iloc[0]
            self.salary = golfer["Salary"].iloc[0]
        except:
            try:
                self.name = golfer["Name + ID"]
                self.salary = golfer["Salary"]
            except:
                raise Exception("Unable to assign ", golfer)
        
        # Try to get total from dk_final.csv, default to 0 if not found
        try:
            self._total = float(golfer["Total"].iloc[0] if isinstance(golfer, pd.DataFrame) else golfer["Total"])
        except (KeyError, AttributeError, ValueError):
            self._total = 0.0

        # Try to get odds total, default to 0 if not found
        try:
            self._odds_total = float(golfer["Odds Total"].iloc[0] if isinstance(golfer, pd.DataFrame) else golfer["Odds Total"])
        except (KeyError, AttributeError, ValueError):
            self._odds_total = 0.0
                
        # New statistics attributes
        self.stats = {
            'current': self._initialize_stats(),
            'historical': {}  # Will hold stats by date
        }
        self.fit_score = None
        

        
    @property
    def total(self) -> float:
        """Get total projected score"""
        return self._total
        
    @total.setter
    def total(self, value: float):
        """Set total projected score"""
        self._total = value
        
    @property
    def value(self) -> float:
        """Calculate value (total points per salary)"""
        if self.salary == 0:
            return 0.0
        return self.total / self.salary * 1000
        
    @property
    def get_clean_name(self) -> str:
        """Returns cleaned name without DK ID and standardized format"""
        # Remove DK ID if present and clean the name
        return fix_names(self.name.split('(')[0].strip())
    
    def get_stats_summary(self) -> Dict:
        """Returns a clean summary of golfer's current stats"""
        stats = self.stats['current']
        sg = stats['strokes_gained']
        
        return {
            'name': self.get_clean_name,
            'driving_distance': stats['driving_distance'],
            'driving_accuracy': stats['driving_accuracy'],
            'gir': stats['gir'],
            'scrambling_sand': stats['scrambling_sand'],
            'strokes_gained': {
                'off_the_tee': sg.off_tee,
                'approach': sg.approach,
                'around_green': sg.around_green,
                'putting': sg.putting,
                'total': sg.total
            },
            'last_updated': stats['last_updated']
        }

    def print_stats(self):
        """Prints a formatted summary of golfer's current stats"""
        stats = self.get_stats_summary()
        print(f"\nStats for {stats['name']}:")
        print(f"Driving Distance: {stats['driving_distance']:.1f}")
        print(f"Driving Accuracy: {stats['driving_accuracy']:.1%}")
        print(f"GIR: {stats['gir']:.1%}")
        print(f"Sand Save: {stats['scrambling_sand']:.1%}")
        print("\nStrokes Gained:")
        for key, value in stats['strokes_gained'].items():
            print(f"  {key.replace('_', ' ').title()}: {value:.3f}")
        
    def _initialize_stats(self) -> Dict:
        """Initialize the stats dictionary with default values"""
        return {
            'strokes_gained': StrokesGained(),
            'driving_distance': 0.0,
            'driving_accuracy': 0.0,
            'gir': 0.0,
            'scrambling_sand': 0.0,
            'scoring_average': 0.0,
            'last_updated': None
        }

    def update_stats(self, 
                    strokes_gained: Optional[StrokesGained] = None,
                    driving_distance: Optional[float] = None,
                    driving_accuracy: Optional[float] = None,
                    gir: Optional[float] = None,
                    scrambling_sand: Optional[float] = None,
                    scoring_average: Optional[float] = None,
                    date: Optional[datetime] = None) -> None:
        """
        Update golfer statistics. If date is provided, stores in historical data.
        Otherwise updates current stats.
        """
        stats_dict = self.stats['historical'].setdefault(date.strftime('%Y-%m-%d'), 
                                                        self._initialize_stats()) if date else self.stats['current']
        
        if strokes_gained:
            stats_dict['strokes_gained'] = strokes_gained
        if driving_distance is not None:
            stats_dict['driving_distance'] = driving_distance
        if driving_accuracy is not None:
            stats_dict['driving_accuracy'] = driving_accuracy / 100  # Convert to decimal
        if gir is not None:
            stats_dict['gir'] = gir / 100
        if scrambling_sand is not None:
            stats_dict['scrambling_sand'] = scrambling_sand / 100
        if scoring_average is not None:
            stats_dict['scoring_average'] = scoring_average
        
        stats_dict['last_updated'] = datetime.now()

    def get_stats_trend(self, stat_name: str, weeks: int = 12) -> List[float]:
        """Get trend data for a specific statistic over the last n weeks"""
        if not self.stats['historical']:
            return []

        dates = sorted(self.stats['historical'].keys())[-weeks:]
        
        if '.' in stat_name:
            main_stat, sub_stat = stat_name.split('.')
            return [self.stats['historical'][date][main_stat].__dict__[sub_stat] 
                   for date in dates]
        
        return [self.stats['historical'][date][stat_name] for date in dates]

    def get_stats_average(self, stat_name: str, weeks: int = 12) -> float:
        """Calculate average for a specific statistic over the last n weeks"""
        trend_data = self.get_stats_trend(stat_name, weeks)
        return np.mean(trend_data) if trend_data else 0.0

    def get_current_form(self) -> float:
        """Calculate golfer's current form based on recent performance"""
        sg_trend = self.get_stats_trend('strokes_gained.total', weeks=4)
        if not sg_trend:
            return 0.0
        
        weights = np.array([0.1, 0.2, 0.3, 0.4])
        weights = weights[-len(sg_trend):]
        weights = weights / weights.sum()
        
        return max(0.0, min(1.0, np.average(sg_trend, weights=weights) / 2.0 + 0.5))

    def adjust_total_for_stats(self) -> float:
        """
        Adjust the total projected points based on current statistics.
        This method can be customized based on how you want to incorporate
        stats into your projections.
        """
        base_total = self.total
        
        # Example adjustment based on current form and recent strokes gained
        form_factor = self.get_current_form()
        recent_sg = self.get_stats_average('strokes_gained.total', weeks=4)
        
        # Simple adjustment formula - can be modified based on your needs
        adjustment = (form_factor - 0.5) * 5  # +/- 2.5 points based on form
        adjustment += recent_sg  # Add recent strokes gained directly
        
        return base_total + adjustment

    def __str__(self):
        return self.name
    
    def __repr__(self):
        return f"{self.name}"
    
    def __eq__(self, other):
        return self.name == other.name

    def set_fit_score(self, score: float) -> None:
        """Set the course fit score for this golfer"""
        self.fit_score = score
        
    def set_odds_total(self, total: float) -> None:
        """Set the odds total score for this golfer"""
        self._odds_total = total
        
    def calculate_total(self, odds_weight: float, fit_weight: float) -> None:
        """Calculate total score using provided weights"""
        if self.odds_total is not None and self.fit_score is not None:
            self._total = (
                odds_weight * self._odds_total +
                fit_weight * self.fit_score
            )
        else:
            self._total = 0.0

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

@dataclass
class StatConfig:
    """Configuration for each statistic"""
    url: str
    name_col: str = 'PLAYER NAME'
    value_col: str = 'AVG.'
    drop_cols: list = None
    table_index: int = 1

class PGATourStatsScraper:
    """Scraper for PGA Tour statistics using Selenium"""
    
    STAT_URLS = {
        'sg_off_tee': 'https://www.pgatour.com/stats/detail/02567',
        'sg_approach': 'https://www.pgatour.com/stats/detail/02568',
        'sg_around_green': 'https://www.pgatour.com/stats/detail/02569',
        'sg_putting': 'https://www.pgatour.com/stats/detail/02564',
        'driving_distance': 'https://www.pgatour.com/stats/detail/101',
        'driving_accuracy': 'https://www.pgatour.com/stats/detail/102',
        'gir': 'https://www.pgatour.com/stats/detail/103',
        'sand_saves': 'https://www.pgatour.com/stats/detail/362'
    }

    def __init__(self):
        self.stats_cache = {}
        self.last_update = {}
        self.driver = None

    def _init_driver(self):
        """Initialize the Selenium webdriver if not already initialized"""
        if self.driver is None:
            options = webdriver.FirefoxOptions()
            options.add_argument('--headless')
            self.driver = webdriver.Firefox(options=options)

    def _quit_driver(self):
        """Quit the Selenium webdriver"""
        if self.driver is not None:
            self.driver.quit()
            self.driver = None

    @staticmethod
    def _clean_name(name: str) -> str:
        """Clean player names to match Golfer class format"""
        # Handle NaN and non-string values
        if pd.isna(name) or not isinstance(name, str):
            return ""
            
        try:
            # Remove any ID portion if present
            name = name.split('(')[0].strip()
            # Convert to lowercase and strip whitespace
            return fix_names(name)
        except Exception as e:
            print(f"Error cleaning name '{name}': {str(e)}")
            return ""

    def _fetch_stat(self, stat_name: str, force_refresh: bool = False) -> pd.DataFrame:
        """Fetch a single statistic from PGA Tour website using Selenium"""
        if not force_refresh and stat_name in self.stats_cache:
            last_update = self.last_update.get(stat_name)
            if last_update and (datetime.now() - last_update).total_seconds() < 86400:
                return self.stats_cache[stat_name]

        url = self.STAT_URLS[stat_name]
        
        try:
            self._init_driver()
            self.driver.get(url)
            
            # Wait for the table to load
            wait = WebDriverWait(self.driver, 10)
            table = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "table")))
            
            # Get the table HTML and wrap in StringIO
            table_html = table.get_attribute('outerHTML')
            
            # Parse the table
            df = pd.read_html(StringIO(table_html))[0]
            
            # Print column names for debugging
            print(f"\nColumns found for {stat_name}:")
            print(df.columns.tolist())
            
            # Handle column names based on the stat type
            if 'Player' in df.columns:
                df = df.rename(columns={'Player': 'Name'})
            
            # Drop unnecessary columns and handle value column
            cols_to_keep = ['Name']
            
            # Map stat names to their column names in the table
            stat_col_map = {
                'sg_off_tee': 'Avg',  # Updated based on your output
                'sg_approach': 'Avg',
                'sg_around_green': 'Avg',
                'sg_putting': 'Avg',
                'driving_distance': 'Avg',
                'driving_accuracy': '%',
                'gir': '%',
                'sand_saves': '%'
            }
            
            # Find the value column
            value_col = stat_col_map.get(stat_name)
            if value_col in df.columns:
                cols_to_keep.append(value_col)
                df = df[cols_to_keep]
                df = df.rename(columns={value_col: stat_name})
            else:
                print(f"Warning: Could not find value column '{value_col}' for {stat_name}")
                print("Available columns:", df.columns.tolist())
                return pd.DataFrame(columns=['Name', stat_name])
            
            # Clean names and remove empty rows
            df['Name'] = df['Name'].apply(self._clean_name)
            df = df[df['Name'] != ""]  # Remove rows with empty names
            
            # Convert stat values to float
            df[stat_name] = pd.to_numeric(df[stat_name].astype(str).str.strip('%'), errors='coerce').fillna(0)
            
            # Update cache
            self.stats_cache[stat_name] = df
            self.last_update[stat_name] = datetime.now()
            
            time.sleep(1)
            
            return df
            
        except Exception as e:
            print(f"Error fetching {stat_name}: {str(e)}")
            print("Full error details:", e.__class__.__name__)
            import traceback
            print(traceback.format_exc())
            return pd.DataFrame(columns=['Name', stat_name])

    def get_all_stats(self, force_refresh: bool = False, use_current_form: bool = False) -> pd.DataFrame:
        """
        Fetch all statistics and combine them into a single DataFrame
        
        Args:
            force_refresh: Whether to force refresh the stats even if they exist
            use_current_form: Whether to use current form data for strokes gained stats
        """
        try:
            timestamp = datetime.now().strftime('%Y%m%d')
            
            # First load or fetch PGA stats as base data
            pga_stats_path = f'golfers/pga_stats_{timestamp}.csv'
            
            if not force_refresh and not should_refresh_stats(pga_stats_path):
                print("Loading existing PGA stats from this week...")
                base_df = pd.read_csv(pga_stats_path)
            else:
                print("Fetching new PGA stats (last update was before this week's Tuesday's...)...")
                base_df = None
                for stat_name in self.STAT_URLS.keys():
                    print(f"Fetching {stat_name}...")
                    df = self._fetch_stat(stat_name, force_refresh)
                    
                    if base_df is None:
                        base_df = df
                    else:
                        base_df = pd.merge(base_df, df, on='Name', how='outer')
            
                # Save PGA stats
                base_df.to_csv(pga_stats_path, index=False)
                
            if use_current_form:
                # Load current form data
                form_path = f'golfers/current_form_20241228.csv' # TODO: Make this dynamic once season starts
                if os.path.exists(form_path):
                    print("Loading and merging current form data with 70/30 weighting...")
                    form_df = pd.read_csv(form_path)
                    
                    # Keep only the strokes gained columns from current form
                    sg_columns = ['Name', 'sg_off_tee', 'sg_approach', 'sg_around_green', 'sg_putting']
                    form_df = form_df[sg_columns]
                    
                    # Merge with base_df
                    final_df = pd.merge(base_df, form_df, on='Name', how='left', suffixes=('_pga', '_form'))
                    
                    # For each SG stat, apply weighted average (70% current form, 30% PGA stats)
                    for sg_stat in ['sg_off_tee', 'sg_approach', 'sg_around_green', 'sg_putting']:
                        pga_col = f'{sg_stat}_pga'
                        form_col = f'{sg_stat}_form'
                        
                        # Fill NaN values with the other source if available
                        final_df[pga_col] = final_df[pga_col].fillna(final_df[form_col])
                        final_df[form_col] = final_df[form_col].fillna(final_df[pga_col])
                        
                        # Calculate weighted average
                        final_df[sg_stat] = (
                            final_df[form_col] * 0.7 + 
                            final_df[pga_col] * 0.3
                        )
                        
                        # Drop the temporary columns
                        final_df = final_df.drop(columns=[pga_col, form_col])
                else:
                    print("Current form data not found, using PGA stats only")
                    final_df = base_df
            else:
                final_df = base_df
            
            # Fill any remaining NaN values with 0
            final_df = final_df.fillna(0)
            
            return final_df
            
        finally:
            self._quit_driver()

    def update_golfer(self, golfer: 'Golfer', stats_df: Optional[pd.DataFrame] = None) -> None:
        """Update a Golfer instance with the latest statistics"""
        if stats_df is None:
            stats_df = self.get_all_stats()
        
        golfer_name = self._clean_name(golfer.name)
        player_stats = stats_df[stats_df['Name'] == golfer_name]
        
        if len(player_stats) == 0:
            print(f"Warning: No stats found for {golfer_name}")
            return
            
        player_stats = player_stats.iloc[0]
        
        # Create StrokesGained object
        sg = StrokesGained(
            off_tee=float(player_stats.get('sg_off_tee', 0)),
            approach=float(player_stats.get('sg_approach', 0)),
            around_green=float(player_stats.get('sg_around_green', 0)),
            putting=float(player_stats.get('sg_putting', 0))
        )
        
        # Update golfer's stats
        golfer.update_stats(
            strokes_gained=sg,
            driving_distance=float(player_stats.get('driving_distance', 0)),
            driving_accuracy=float(player_stats.get('driving_accuracy', 0)),
            gir=float(player_stats.get('gir', 0)),
            scrambling_sand=float(player_stats.get('sand_saves', 0))
        )

    def update_golfer_list(self, golfers: list['Golfer'], use_current_form: bool = False) -> None:
        """Update a list of golfers efficiently by fetching stats once"""
        stats_df = self.get_all_stats(use_current_form=use_current_form)
        for golfer in golfers:
            self.update_golfer(golfer, stats_df)

@dataclass
class StatCorrelation:
    """Defines how a player stat correlates with a course stat"""
    course_stat: str  # Name of the course stat
    player_stat: str  # Name of the player stat
    weight: float     # How important this correlation is
    is_inverse: bool = False  # True if higher course stat means lower player stat is better

@dataclass
class StatMapping:
    """Defines how a player stat maps to a course stat"""
    course_stat: str
    player_stat: str
    weight: float
    is_inverse: bool = False
    
    def calculate_fit(self, course_val: float, player_val: float) -> float:
        """Abstract method to be implemented by subclasses"""
        raise NotImplementedError

class DistanceMapping(StatMapping):
    def __init__(self, course_stat: str, player_stat: str, weight: float, 
                 course_range: tuple = (270, 325), player_range: tuple = (270, 325)):
        super().__init__(course_stat, player_stat, weight)
        self.course_range = course_range
        self.player_range = player_range  
        
    def calculate_fit(self, course_val: float, player_val: float) -> float:
        """
        For distance:
        - If player >= course requirement:
          - Get 0.9 base points for meeting requirement
          - Get up to 0.1 additional points based on how much they exceed it
        - If player < course requirement:
          - Get partial points based on how close they are (max 50% penalty)
        """
        if self.is_inverse:
            player_val = 1 - player_val  # Invert for negative correlations
            
        if player_val >= course_val:
            # Base score for meeting requirement
            base_score = 0.9
            
            # Calculate bonus (max 0.1) based on how much they exceed requirement
            # Cap the bonus calculation at 30 yards over requirement
            excess_yards = min(player_val - course_val, 30)
            bonus = 0.1 * (excess_yards / 30)
            
            return base_score + bonus
        else:
            # Calculate deficit percentage (max penalty of 50%)
            deficit = (course_val - player_val) / course_val
            return max(0.3, 1.0 - deficit)

class AccuracyMapping(StatMapping):
    # Default ranges for different accuracy types
    DEFAULT_RANGES = {
        'adj_driving_accuracy': {
            'course': (0.44, 0.87),
            'player': (0.49, 0.73)
        },
        'fw_width': {
            'course': (23.5, 71.9),
            'player': (0.49, 0.73)
        }
    }
    
    def __init__(self, course_stat: str, player_stat: str, weight: float, 
                 course_range: tuple = None, player_range: tuple = None,
                 is_inverse: bool = False):
        """
        Initialize with ranges based on the accuracy stat type.
        Falls back to provided ranges or generic defaults if stat type not found.
        """
        super().__init__(course_stat, player_stat, weight, is_inverse)
        
        # Get default ranges for this stat type
        defaults = self.DEFAULT_RANGES.get(course_stat, {
            'course': (40, 80),  # Generic fallback ranges as percentages
            'player': (40, 80)
        })
        
        self.course_range = course_range or defaults['course']
        self.player_range = player_range or defaults['player']

    def calculate_fit(self, course_val: float, player_val: float) -> float:
        """
        For accuracy:
        - All scores start at 1.0 (100%)
        - Apply penalties based on course difficulty and player accuracy
        """
        if self.is_inverse and self.course_stat == 'fw_width':
            # Normalize fairway width (lower is harder)
            min_course, max_course = self.course_range
            norm_width = (course_val - min_course) / (max_course - min_course)
            
            # Normalize player accuracy (higher is better)
            min_player, max_player = self.player_range
            norm_accuracy = (player_val - min_player) / (max_player - min_player)
            
            # For narrow fairways (low norm_width), we want high accuracy
            # Calculate penalty: higher when fairways are narrow AND accuracy is low
            difficulty = 1 - norm_width  # Convert width to difficulty (0=wide, 1=narrow)
            accuracy_deficit = 1 - norm_accuracy  # How inaccurate the player is
            
            # Penalty increases with both difficulty and inaccuracy
            penalty = difficulty * accuracy_deficit
            
            return max(0.3, 1.0 - penalty)  # Cap maximum penalty at 60%
    
        # Original logic for other accuracy mappings
        if self.is_inverse:
            course_val = 1 - course_val
            
        if player_val >= course_val:
            return 1.0  # No penalty for meeting/exceeding requirement
        else:
            deficit = (course_val - player_val) / course_val
            return max(0.4, 1.0 - deficit)  # Up to 60% penalty

class StrokesGainedMapping(StatMapping):
    # Default ranges for different SG types
    DEFAULT_RANGES = {
        'ott_sg': {
            'course': (-0.120, 0.130),
            'player': (-1.4, 0.9)
        },
        'app_sg': {
            'course': (-0.06, 0.9),
            'player': (-1.1, 1.3)
        },
        'arg_sg': {
            'course': (-0.120, 0.08),
            'player': (-1.0, 0.6)
        },
        'putt_sg': {
            'course': (-0.035, 0.035),
            'player': (-1.0, 1.0)
        }
    }
    
    def __init__(self, course_stat: str, player_stat: str, weight: float):
        super().__init__(course_stat, player_stat, weight)
        self.original_weight = weight
        self.course_range = self.DEFAULT_RANGES[course_stat]['course']
        self.player_range = self.DEFAULT_RANGES[course_stat]['player']
        # Remove default ranges, will calculate per tournament

    def calculate_tournament_ranges(self, golfers: List['Golfer']) -> tuple:
        """Calculate min/max ranges based on the tournament field"""
        values = []
        for golfer in golfers:
            try:
                # Use self.player_stat instead of golfer.player_stat
                if self.player_stat.startswith('sg_'):
                    val = getattr(golfer.stats['current']['strokes_gained'], 
                                self.player_stat.replace('sg_', ''))
                else:
                    val = golfer.stats['current'][self.player_stat]
                values.append(val)
            except (AttributeError, KeyError) as e:
                continue  # Skip golfers with missing stats
        
        if not values:
            return (-1, 1)  # Default range if no valid values found
        
        # Use percentiles instead of min/max to handle outliers
        min_val = np.percentile(values, 5)  # 5th percentile
        max_val = np.percentile(values, 95)  # 95th percentile
        return (min_val, max_val)

    def calculate_fit(self, course_val: float, player_val: float, tournament_range: tuple = None) -> float:
        """
        For Strokes Gained:
        - Score directly correlates to player's percentile in tournament field
        - Apply weight adjustments based on course importance
        """
        if tournament_range:
            min_player, max_player = tournament_range
        else:
            min_player, max_player = self.DEFAULT_RANGES[self.course_stat]['player']
        
        # Normalize course_val to -1 to 1 range based on its specific course_range
        min_course, max_course = self.course_range
        normalized_course_val = 2 * (course_val - min_course) / (max_course - min_course) - 1

        # Adjust weight based on both raw and normalized values
        if course_val < 0:
            # Stronger penalty for actually negative values
            penalty = min(0.7, abs(normalized_course_val))  # Up to 70% penalty
            self.weight = self.original_weight * (1.0 - penalty)
        elif normalized_course_val < 0:
            # Milder penalty for positive values that normalize to negative
            penalty = min(0.2, abs(normalized_course_val))  # Up to 20% penalty
            self.weight = self.original_weight * (1.0 - penalty)
        else:
            self.weight = self.original_weight

        # Calculate percentile directly (0 to 1 range)
        if player_val <= min_player:
            return 0.0
        elif player_val >= max_player:
            return 1.0
        else:
            return (player_val - min_player) / (max_player - min_player)

class CoursePlayerFit:
    """Analyzes how well a player's stats fit a course's characteristics"""
    
    # The mapping weights are in the optimization_results/final_weights.csv file
    # Load weights from CSV
    weights_df = pd.read_csv('optimization_results/final_weights.csv')
    weights = {row['Stat']: row['Weight'] * 4 for _, row in weights_df.iterrows()}
        
    STAT_MAPPINGS = [
        # Distance correlations
        DistanceMapping('adj_driving_distance', 'driving_distance', weights['Driving Distance']),
        # Accuracy correlations
        AccuracyMapping('adj_driving_accuracy', 'driving_accuracy', weights['Driving Accuracy']),
        AccuracyMapping('fw_width', 'driving_accuracy', weights['Fairway Width'], is_inverse=True),
        
        # Strokes Gained correlations
        StrokesGainedMapping('ott_sg', 'sg_off_tee', weights['Off the Tee SG']),
        StrokesGainedMapping('app_sg', 'sg_approach', weights['Approach SG']),
        StrokesGainedMapping('arg_sg', 'sg_around_green', weights['Around Green SG']),
        StrokesGainedMapping('putt_sg', 'sg_putting', weights['Putting SG']),
        
        # Specific situation correlations
        AccuracyMapping('arg_bunker_sg', 'scrambling_sand', weights['Sand Save'])
    ]

    def __init__(self, course: CourseStats, golfers: List['Golfer'], mappings: List[StatMapping] = None, verbose: bool = False):
        self.course = course
        self.verbose = verbose
        self.STAT_MAPPINGS = mappings if mappings is not None else self.STAT_MAPPINGS
        
        # Calculate tournament-specific ranges
        self.tournament_ranges = {}
        if self.verbose:
            print("\nCalculating Tournament-Specific Ranges:")
            print("----------------------------------------")
            print("Stat             Default Range          Tournament Range")
            print("----------------------------------------")
            
        for mapping in self.STAT_MAPPINGS:
            if isinstance(mapping, StrokesGainedMapping):
                tournament_range = mapping.calculate_tournament_ranges(golfers)
                self.tournament_ranges[mapping.course_stat] = tournament_range
                
                if self.verbose:
                    # Access DEFAULT_RANGES from StrokesGainedMapping class
                    default_range = StrokesGainedMapping.DEFAULT_RANGES[mapping.course_stat]['player']
                    print(f"{mapping.course_stat:<15} ({default_range[0]:6.3f}, {default_range[1]:6.3f})    "
                          f"({tournament_range[0]:6.3f}, {tournament_range[1]:6.3f})")

    def calculate_fit_score(self, golfer: 'Golfer', verbose: bool = None) -> Dict[str, float]:
        """Calculate how well a golfer's stats fit the course."""
        verbose = self.verbose if verbose is None else verbose
        
        if verbose:
            print(f"\nCalculating fit score for {golfer.name}")
            print("\nComponent Scores:")
            print("----------------------------------------")
            print("Stat             Raw Values        Normalized      Base   Adj    Score")
            print("                Course  Player    Course  Player   Wgt    Wgt")
            print("----------------------------------------")
        
        scores = {}
        weights_sum = 0
        total_score = 0
        
        for mapping in self.STAT_MAPPINGS:
            try:
                # Get course value
                course_val = getattr(self.course, mapping.course_stat)
                
                # Get player value
                if mapping.player_stat.startswith('sg_'):
                    player_val = getattr(golfer.stats['current']['strokes_gained'], 
                                       mapping.player_stat.replace('sg_', ''))
                else:
                    player_val = golfer.stats['current'][mapping.player_stat]
                
                # Store original weight for SG stats
                original_weight = mapping.weight if isinstance(mapping, StrokesGainedMapping) else None
                
                # Calculate fit score using appropriate mapping
                fit_score = mapping.calculate_fit(course_val, player_val) * mapping.weight
                
                scores[f"{mapping.course_stat}-{mapping.player_stat}"] = fit_score
                total_score += fit_score
                weights_sum += mapping.weight
                
                # Move print statements inside verbose check
                if verbose:
                    if isinstance(mapping, StrokesGainedMapping):
                        min_course, max_course = mapping.course_range
                        min_player, max_player = mapping.player_range
                        norm_course = 2 * (course_val - min_course) / (max_course - min_course) - 1
                        norm_player = 2 * (player_val - min_player) / (max_player - min_player) - 1
                        print(f"{mapping.course_stat:<15} {course_val:6.3f}  {player_val:6.3f}    "
                              f"{norm_course:6.3f}  {norm_player:6.3f}  {original_weight:6.3f}  "
                              f"{mapping.weight:6.3f}  {fit_score:6.3f}")
                    else:
                        print(f"{mapping.course_stat:<15} {course_val:6.3f}  {player_val:6.3f}            "
                              f"---  ---      {mapping.weight:6.3f}  {mapping.weight:6.3f}  {fit_score:6.3f}")
                    
                    if mapping.is_inverse:
                        print("  (Inverse correlation)")
                
            except (AttributeError, KeyError, TypeError) as e:
                if verbose:
                    print(f"Warning: Couldn't calculate {mapping.course_stat}-{mapping.player_stat} "
                          f"correlation: {str(e)}")
        
        overall_score = (total_score / weights_sum * 100) if weights_sum > 0 else 0
        if verbose:
            print("----------------------------------------")
            # Calculate overall fit score (0-100 scale)
            print(f"\nOverall Fit: {overall_score:.1f}%")
        
        return {
            'overall_fit': overall_score,
            'component_scores': scores
        }

    def get_key_stats(self, golfer: 'Golfer', verbose: bool = None) -> List[str]:
        """Returns list of key stats for this course based on the golfer's profile."""
        verbose = self.verbose if verbose is None else verbose
        
        if verbose:
            print(f"\nAnalyzing key stats for {golfer.name} at {self.course.name}")
        
        key_stats = []
        fit_scores = self.calculate_fit_score(golfer, verbose=False)['component_scores']
        
        course_emphasis_threshold = 0.7
        player_weakness_threshold = 0.5
        
        if verbose:
            print("\nKey Stats Analysis:")
            
        for corr in self.STAT_CORRELATIONS:
            if corr.course_stat not in self.normalized_stats:
                continue
                
            course_val = self.normalized_stats[corr.course_stat]
            fit_score = fit_scores.get(corr.course_stat, 0)
            
            if course_val > course_emphasis_threshold:
                if fit_score < player_weakness_threshold:
                    msg = f"Improve {corr.player_stat} for this course"
                    key_stats.append(msg)
                    if verbose:
                        print(f"- {msg} (course: {course_val:.3f}, fit: {fit_score:.3f})")
                elif fit_score > 0.8:
                    msg = f"Leverage strong {corr.player_stat}"
                    key_stats.append(msg)
                    if verbose:
                        print(f"+ {msg} (course: {course_val:.3f}, fit: {fit_score:.3f})")
        
        return key_stats

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
    
    finish_points = 0.0
    max_possible_points = 0.0
    appearances = 0
    avg_finish = 0.0
    
    for year, weight in zip(years, weights):
        if year in history_df.columns:
            finish = player_history[year]
            if pd.notna(finish):
                # Track number of appearances and average finish
                appearances += 1
                avg_finish += finish
                
                # Convert finish position to points (1st = 100, 60th = 0)
                points = max(0, 100 - ((finish - 1) * (100/60)))
                finish_points += points * weight
                max_possible_points += 100 * weight
    
    # Calculate average finish position
    avg_finish = avg_finish / appearances if appearances > 0 else 0
    
    # Apply consistency penalty for multiple poor performances
    if appearances > 1:
        # Penalty increases with number of appearances and average finish position
        consistency_penalty = (avg_finish / 60) * (appearances / 5)  # 5 is max years tracked
        finish_points *= (1 - consistency_penalty)
    
    # Normalize finish points (50% of total score)
    finish_score = (finish_points / max_possible_points) * 50 if max_possible_points > 0 else 0
    
    # Calculate strokes gained score (30% of total)
    sg_stats = ['sg_ott', 'sg_app', 'sg_atg', 'sg_putting']
    sg_weights = [0.25, 0.35, 0.25, 0.15]  # Must sum to 1
    
    sg_score = 0.0
    for stat, weight in zip(sg_stats, sg_weights):
        if stat in history_df.columns:
            sg_val = player_history[stat]
            if pd.notna(sg_val):
                # Convert SG to 0-100 scale (-2 to +2 range)
                normalized_sg = min(100, max(0, (sg_val + 2) * 25))
                sg_score += normalized_sg * weight
    
    # Scale SG score to 30% of total
    sg_score *= 0.3
    
    # Calculate consistency score (20% of total)
    consistency_score = player_history['made_cuts_pct'] * 20 if 'made_cuts_pct' in history_df.columns else 0
    # Combine all components
    total_score = finish_score + sg_score + consistency_score
    
    return total_score

def get_current_tuesday() -> datetime:
    """Get the date of the current week's Tuesday"""
    today = datetime.now()
    days_since_tuesday = (today.weekday() - 1) % 7  # Tuesday is 1
    return today - pd.Timedelta(days=days_since_tuesday)

def should_refresh_stats(stats_path: str) -> bool:
    """
    Check if stats should be refreshed based on file existence and date
    Returns True if:
    - File doesn't exist
    - File is from before current week's Tuesday
    """
    if not os.path.exists(stats_path):
        return True
        
    file_timestamp = datetime.fromtimestamp(os.path.getmtime(stats_path))
    current_tuesday = get_current_tuesday()
    
    return file_timestamp < current_tuesday

def main(tourney: str, num_lineups: int = 20, tournament_history: bool = False):
    print(f"\n{'='*50}")
    print(f"Running optimization for {tourney}")
    print(f"{'='*50}\n")

    # Read odds data and DraftKings salaries
    odds_df = pd.read_csv(f'2025/{tourney}/odds.csv')
    dk_salaries = pd.read_csv(f'2025/{tourney}/DKSalaries.csv')
    if tournament_history:
        tourney_history = pd.read_csv(f'tournaments/{tourney}/tournament_history.csv')
    else:
        tourney_history = pd.DataFrame()
    
    print(f"Loaded {len(odds_df)} players from odds data")
    print(f"Loaded {len(dk_salaries)} players from DraftKings data\n")
    
    # Clean up names in both dataframes
    odds_df['Name'] = odds_df['Name'].apply(fix_names)
    dk_salaries['Name'] = dk_salaries['Name'].apply(fix_names)
    
    # Merge odds with DraftKings data
    dk_data = pd.merge(dk_salaries, odds_df, on='Name', how='left')
    print(f"After merging: {len(dk_data)} players\n")
    
    # Calculate odds total using odds_to_score function with optimized weights
    dk_data['Tournament Winner'] = dk_data['Tournament Winner'].apply(lambda x: odds_to_score(x, "Tournament Winner", w=0.6))
    dk_data['Top 5 Finish'] = dk_data['Top 5 Finish'].apply(lambda x: odds_to_score(x, "Top 5 Finish", t5=0.5))
    dk_data['Top 10 Finish'] = dk_data['Top 10 Finish'].apply(lambda x: odds_to_score(x, "Top 10 Finish", t10=0.8))
    dk_data['Top 20 Finish'] = dk_data['Top 20 Finish'].apply(lambda x: odds_to_score(x, "Top 20 Finish", t20=0.4))
    
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

    # Create scraper
    scraper = PGATourStatsScraper()

    # Update all golfers at once
    scraper.update_golfer_list(golfers, use_current_form=True)

    print(f"Loading course stats for {tourney}...")
    course_stats = load_course_stats(tourney)
    analyzer = CoursePlayerFit(course_stats, golfers)
    print("Course stats loaded successfully\n")

    # Calculate fit scores for all golfers and create DataFrame
    fit_scores_data = []
    history_scores = []
    for golfer in golfers:
        if tournament_history:
            history_score = calculate_tournament_history_score(golfer.get_clean_name, tourney_history)
        else:
            history_score = 0
        fit_score = analyzer.calculate_fit_score(golfer)
        golfer.set_fit_score(fit_score['overall_fit'])
        fit_scores_data.append({
            'Name': golfer.get_clean_name,
            'Fit Score': fit_score['overall_fit']
        })
        history_scores.append({
            'Name': golfer.get_clean_name,
            'History Score': history_score
        })
    history_scores_df = pd.DataFrame(history_scores)
    fit_scores_df = pd.DataFrame(fit_scores_data)
    

    # Merge fit scores and history scores with dk_data
    dk_data = pd.merge(dk_data, fit_scores_df, on='Name', how='left')
    dk_data = pd.merge(dk_data, history_scores_df, on='Name', how='left')


    # Load weights from optimization results
    
    if len(tourney_history) > 0:
        odds_weight = 0.5
        fit_weight = 0.3
        history_weight = 0.2
    else:
        weights_df = pd.read_csv('optimization_results/fit_odds_final_weights.csv')
        odds_weight = weights_df['odds_weight'].iloc[0]
        fit_weight = weights_df['fit_weight'].iloc[0]
        history_weight = 0.0

    
    
    print("Optimization Weights:")
    print(f"Odds Weight: {odds_weight:.3f}")
    print(f"Fit Weight:  {fit_weight:.3f}\n")
    print(f"History Weight: {history_weight:.3f}\n")

    # Normalize Odds Total and Fit Score using min-max scaling
    dk_data['Normalized Odds'] = (dk_data['Odds Total'] - dk_data['Odds Total'].min()) / (dk_data['Odds Total'].max() - dk_data['Odds Total'].min())
    dk_data['Normalized Fit'] = (dk_data['Fit Score'] - dk_data['Fit Score'].min()) / (dk_data['Fit Score'].max() - dk_data['Fit Score'].min())
    if tournament_history:
        dk_data['Normalized History'] = (dk_data['History Score'] - dk_data['History Score'].min()) / \
        (dk_data['History Score'].max() - dk_data['History Score'].min())
    else:
        dk_data['Normalized History'] = 0
    
    # Calculate Total using normalized values
    dk_data['Total'] = dk_data['Normalized Odds'] * odds_weight + dk_data['Normalized Fit'] * fit_weight + dk_data['Normalized History'] * history_weight
    
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
    output_path = f"2025/{tourney}/dk_lineups_optimized.csv"
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
    output_data_path = f"2025/{tourney}/player_data.csv"
    columns_to_save = [
        'Name', 'Salary', 'Odds Total', 'Normalized Odds',
        'Fit Score', 'Normalized Fit',
        'History Score', 'Normalized History',
        'Total', 'Value'
    ]
    dk_data[columns_to_save].sort_values('Total', ascending=False).to_csv(output_data_path, index=False)
    print(f"Saved detailed player data to: {output_data_path}")

if __name__ == "__main__":
    main("The_Sentry", num_lineups=10,tournament_history=True)