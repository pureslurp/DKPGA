from datetime import datetime
import pandas as pd
import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from io import StringIO
import os
from utils import fix_names
from models import StrokesGained, Golfer
from typing import Optional


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

    def get_all_stats(self,tourney: str, force_refresh: bool = False, use_current_form: bool = False, weights: dict = None) -> pd.DataFrame:
        """
        Fetch all statistics and combine them into a single DataFrame
        
        Args:
            force_refresh: Whether to force refresh the stats even if they exist
            use_current_form: Whether to use current form data for strokes gained stats
            weights: Weights for the different components
        """
        try:
            pga_stats_path = get_stats_filename(tourney)
            
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
                form_path = f'2025/{tourney}/current_form.csv'
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
                            final_df[form_col] * weights['form']['current'] + 
                            final_df[pga_col] * weights['form']['long']
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
            
            # Add sg_total column as sum of all other sg columns
            sg_columns = ['sg_off_tee', 'sg_approach', 'sg_around_green', 'sg_putting']
            final_df['sg_total'] = final_df[sg_columns].sum(axis=1)
            
            return final_df
            
        finally:
            self._quit_driver()

    def update_golfer(self, golfer: 'Golfer', tourney: str, stats_df: Optional[pd.DataFrame] = None) -> None:
        """Update a Golfer instance with the latest statistics"""
        if stats_df is None:
            stats_df = self.get_all_stats(tourney)
        
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

    def update_golfer_list(self, golfers: list['Golfer'], tourney: str, use_current_form: bool = False) -> None:
        """Update a list of golfers efficiently by fetching stats once"""
        stats_df = self.get_all_stats(tourney, use_current_form=use_current_form)
        for golfer in golfers:
            self.update_golfer(golfer, tourney, stats_df)

def get_stats_filename(tourney: str) -> str:
    """Get the standardized filename for the tournament's stats file"""
    return f'2025/{tourney}/pga_stats.csv'

def should_refresh_stats(stats_path: str) -> bool:
    """Check if stats should be refreshed"""
    return not os.path.exists(stats_path)

def create_pga_stats(tourney: str, weights: dict = None):
    weights = weights or {'form': {'current': 0.7, 'long': 0.3}}
    
    stats_path = get_stats_filename(tourney)
    
    if should_refresh_stats(stats_path):
        print("No existing stats file found. Fetching fresh stats...")
        scraper = PGATourStatsScraper()
        stats_df = scraper.get_all_stats(tourney)
        
        if not stats_df.empty:
            # Ensure directory exists
            os.makedirs(os.path.dirname(stats_path), exist_ok=True)
            # Save to tournament directory
            stats_df.to_csv(stats_path, index=False)
            print(f"Created new stats file at: {stats_path}")
        else:
            print("Error: Could not fetch PGA Tour stats")
    else:
        print(f"Stats file already exists at: {stats_path}")

if __name__ == "__main__":
    tourney = "Rocket_Classic"
    create_pga_stats(tourney)