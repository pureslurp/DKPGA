# Standard library imports
import time
from typing import Optional, Dict, Any

# Third-party imports
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import os

from utils import TOURNAMENT_LIST_2025

'''
Script to scrape tournament history data from the PGA Tour website for a specific tournament.
This spans back 5 years.

Output:
    - tournament/{tournament_name}/tournament_history.csv: the tournament history data
'''

from pga_v5 import fix_names
def parse_finishes(row_data: Dict[str, str]) -> Dict[str, float]:
    """Parse finishing positions for last 5 years"""
    years = ["24", "2022-23", "2021-22", "2020-21", "2019-20"]
    finishes = {}
    
    for year in years:
        finish = row_data.get(year, '')
        if pd.isna(finish) or finish == '':
            finishes[year] = np.nan
        else:
            # Handle special cases
            if finish == 'P1':
                finishes[year] = 1
            elif finish in ['PT2', 'P2']:
                finishes[year] = 2
            elif finish in ['CUT', 'CU', 'DQ', 'WD', 'MDF']:  # Add more special cases as needed
                finishes[year] = "CUT"  # or could use a specific value like 999 if you want to track cuts
            else:
                # Remove T and convert to int
                finishes[year] = int(finish.replace('T', ''))
    
    return finishes

def parse_strokes_gained(row_data: Dict[str, str]) -> Dict[str, float]:
    """Parse strokes gained statistics"""
    stats = {}
    
    sg_fields = {
        'SG: OTT': 'sg_ott',
        'SG: APP': 'sg_app', 
        'SG: ATG': 'sg_atg',
        'SG: P': 'sg_putting',
        'SG: TOT': 'sg_total'
    }
    
    for field, key in sg_fields.items():
        value = row_data.get(field, '')
        if pd.isna(value) or value == '' or value == '-':
            stats[key] = np.nan
        else:
            stats[key] = float(value)
            
    # Get number of measured rounds
    rounds = row_data.get('Rounds', 0)
    if pd.isna(rounds) or rounds == '' or rounds == '-':
        stats['rounds'] = 0
    else:
        stats['rounds'] = int(rounds)
        
    return stats

def extract_player_data(html_table: BeautifulSoup) -> pd.DataFrame:
    """Extract player tournament history data from HTML table"""
    players = []
    
    for row in html_table.find_all('tr')[1:]:  # Skip header row
        cells = row.find_all('td')
        if len(cells) < 12:  # Skip incomplete rows
            continue
            
        # Get player name and format it
        name_cell = cells[0].find('span', class_='chakra-text')
        if not name_cell:
            continue
        
        # Convert "Last, First" to "first last" format
        name = name_cell.text.strip()
        if ',' in name:
            last_name, first_name = name.split(',', 1)
            name = f"{first_name.strip()} {last_name.strip()}"
        name = fix_names(name)  # Use existing fix_names function
        
        # Get recent finishes
        finish_data = {}
        for i, year in enumerate(['24', '2022-23', '2021-22', '2020-21', '2019-20']):
            finish = cells[i+1].find('span', class_='chakra-text')
            if finish:
                finish_data[year] = finish.text.strip()
            
        # Get strokes gained data
        sg_data = {}
        for i, field in enumerate(['Rounds', 'SG: OTT', 'SG: APP', 'SG: ATG', 'SG: P', 'SG: TOT']):
            sg = cells[i+7].find('span', class_='chakra-text')
            if sg:
                sg_data[field] = sg.text.strip()
        
        # Parse all data
        player_data = {
            'Name': name,
            **parse_finishes(finish_data),
            **parse_strokes_gained(sg_data)
        }
        
        players.append(player_data)
        
    return pd.DataFrame(players)

def format_tournament_history(df: pd.DataFrame) -> pd.DataFrame:
    """Format tournament history data for analysis"""
    # Calculate average finish (excluding DNPs)
    finish_cols = ['24', '2022-23', '2021-22', '2020-21', '2019-20']
    df['avg_finish'] = df[finish_cols].replace('CUT', 65).astype(float).mean(axis=1, skipna=True)
    
    # Calculate measured years (number of tournaments played)
    df['measured_years'] = df[finish_cols].notna().sum(axis=1)
    
    # Calculate made cuts percentage based on measured years
    def _calculate_made_cuts_pct(row):
        # Count total tournaments played (non-NA values)
        total_tournaments = 0
        # Count tournaments where player made the cut
        made_cuts = 0
        
        for col in finish_cols:
            value = row[col]
            # Check if the player played in this tournament
            if pd.notna(value):
                total_tournaments += 1
                # Check if they made the cut
                if value != 'CUT':
                    made_cuts += 1
        
        # Calculate percentage
        if total_tournaments > 0:
            return made_cuts / total_tournaments
        return 0.0
    
    df['made_cuts_pct'] = df.apply(_calculate_made_cuts_pct, axis=1)
    
    # Sort by strokes gained total and average finish
    df = df.sort_values(['sg_total', 'avg_finish'], ascending=[False, True])
    
    return df

def get_tournament_name(soup: BeautifulSoup) -> str:
    """
    Extract tournament name from the page and format it with underscores.
    
    Args:
        soup: BeautifulSoup object of the page HTML
        
    Returns:
        Tournament name with spaces replaced by underscores
    """
    tournament_header = soup.find('h1', class_='chakra-text')
    if tournament_header:
        tournament_name = tournament_header.text.strip()
        return tournament_name.replace(' ', '_')
    return "Unknown_Tournament"

def get_tournament_history(url: str) -> Optional[pd.DataFrame]:
    """
    Main function to process tournament history data using Selenium.
    
    Args:
        url: URL of the PGA Tour tournament history page
        
    Returns:
        DataFrame containing tournament history data or None if scraping fails
    """
    # Set up Firefox options
    firefox_options = Options()
    firefox_options.add_argument("--headless")  # Run in headless mode
    
    try:
        # Initialize the driver
        driver = webdriver.Firefox(options=firefox_options)
        driver.get(url)
        
        # Wait for table to load
        wait = WebDriverWait(driver, 20)
        table = wait.until(EC.presence_of_element_located((By.CLASS_NAME, "chakra-table")))
        
        # Get the page source after JavaScript has loaded
        html_content = driver.page_source
        
        # Parse HTML
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Get tournament name
        tournament_name = get_tournament_name(soup)
        
        table = soup.find('table', class_='chakra-table')
        
        # Extract and format data
        df = extract_player_data(table)
        df = format_tournament_history(df)
        
        # Add tournament name to DataFrame
        df['tournament'] = tournament_name
        
        return df
        
    except Exception as e:
        print(f"Error scraping data: {e}")
        return None
        
    finally:
        # Close the browser
        try:
            driver.quit()
        except:
            pass

if __name__ == "__main__":
    # Example usage
    TOURNEY = "Farmers_Insurance_Open"
    url = f"https://www.pgatour.com/tournaments/2025/{TOURNAMENT_LIST_2025[TOURNEY]['pga-url']}/field/tournament-history"
    df = get_tournament_history(url)
    
    if df is not None:
        print(f"\nTournament: {df['tournament'].iloc[0]}")
        print("\nTop 10 players by strokes gained total:")
        print(df[['Name', 'sg_total', 'avg_finish', 'made_cuts_pct']].head(10))
        
        # Save to CSV with tournament name
        tournament_name = df['tournament'].iloc[0]
        # Create directory if it doesn't exist
        os.makedirs(f'2025/{tournament_name}', exist_ok=True)
        df.to_csv(f'2025/{tournament_name}/tournament_history.csv', index=False)
    else:
        print("Failed to retrieve tournament history data")