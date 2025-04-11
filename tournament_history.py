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
from pga_v5 import fix_names, calculate_tournament_history_score_internal

'''
Script to scrape tournament history data from the PGA Tour website for a specific tournament.
This spans back 5 years.

Output:
    - tournament/{tournament_name}/tournament_history.csv: the tournament history data
'''

def parse_finishes(row_data: Dict[str, str]) -> Dict[str, float]:
    """Parse finishing positions for last 5 years"""
    years = ["24", "2022-23", "2021-22", "2020-21"]
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
        print(value)
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
    
    # Get header row to determine which years are present
    header_row = html_table.find('tr', class_='css-1odhsqu')
    if not header_row:
        return pd.DataFrame()
        
    # Get all header cells and extract years
    header_cells = header_row.find_all('th')
    available_years = []
    
    # Map column names to their indices
    column_indices = {'rounds': None, 'sg_start': None}
    
    for i, cell in enumerate(header_cells):
        button = cell.find('button')
        if button:
            text = button.text.strip()
            
            # Check if it's a year
            if text in ['24', '2022-23', '2021-22', '2020-21']:
                available_years.append(text)
            # Check if it's the rounds column (marks start of SG stats)
            elif text == 'Rounds':
                column_indices['rounds'] = i
                column_indices['sg_start'] = i + 1
    
    print(f"\nProcessing with:")
    print(f"Available years: {available_years}")
    print(f"Column indices: {column_indices}")
    
    for row in html_table.find_all('tr', class_='css-79elbk'):  # Add the player row class
        cells = row.find_all('td')
        if len(cells) < 7:
            continue
            
        # Get player name
        name_cell = cells[0].find('span', class_='chakra-text')
        if not name_cell:
            continue
        
        name = name_cell.text.strip()
        if ',' in name:
            last_name, first_name = name.split(',', 1)
            name = f"{first_name.strip()} {last_name.strip()}"
        name = fix_names(name)
        
        print(f"\nProcessing player: {name}")
        
        # Get recent finishes
        finish_data = {}
        for i, year in enumerate(available_years):
            finish = cells[i+1].find('span', class_='chakra-text')
            if finish:
                finish_data[year] = finish.text.strip()
        
        print(f"Finish data: {finish_data}")
        
        # Get strokes gained data
        sg_data = {
            'Rounds': '',
            'SG: OTT': '',
            'SG: APP': '',
            'SG: ATG': '',
            'SG: P': '',
            'SG: TOT': ''
        }
        
        # Get rounds and SG stats
        if column_indices['rounds'] is not None:
            # Get rounds
            rounds = cells[column_indices['rounds']].find('span', class_='chakra-text')
            if rounds:
                sg_data['Rounds'] = rounds.text.strip()
            
            # Get SG stats
            sg_fields = ['SG: OTT', 'SG: APP', 'SG: ATG', 'SG: P', 'SG: TOT']
            for i, field in enumerate(sg_fields):
                cell_index = column_indices['sg_start'] + i
                if cell_index < len(cells):
                    sg = cells[cell_index].find('span', class_='chakra-text')
                    if sg:
                        sg_data[field] = sg.text.strip()
        
        print(f"SG data before parsing: {sg_data}")
        
        # Parse all data
        player_data = {
            'Name': name,
            **parse_finishes(finish_data),
            **parse_strokes_gained(sg_data)
        }
        
        print(f"Final player data: {player_data}")
        players.append(player_data)
    
    return pd.DataFrame(players)

def format_tournament_history(df: pd.DataFrame) -> pd.DataFrame:
    """Format tournament history data for analysis"""
    # Calculate average finish (excluding DNPs)
    finish_cols = ['24', '2022-23', '2021-22', '2020-21']
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
    
    # Calculate history score using imported function from pga_v5
    df['history_score'] = df.apply(lambda row: calculate_tournament_history_score_internal(row, df), axis=1)
    
    # Sort by history score first, then strokes gained total
    df = df.sort_values(['history_score', 'sg_total'], ascending=[False, False])
    
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
    
    # try:
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
        
    # except Exception as e:
    #     print(f"Error scraping data: {e}")
    #     return None
        
    # finally:
    #     # Close the browser
    #     try:
    #         driver.quit()
    #     except:
    #         pass

if __name__ == "__main__":
    # Example usage
    TOURNEY = "Masters_Tournament"
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