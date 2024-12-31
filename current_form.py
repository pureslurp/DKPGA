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

from pga_v5 import fix_names

def parse_recent_finishes(row_data):
    """Parse the last 5 tournament finishes"""
    finishes = []
    finish_texts = row_data.find_all('p', class_='chakra-text')
    
    for finish in finish_texts:
        text = finish.text.strip()
        if text == 'CUT':
            finishes.append(None)  # or handle cuts differently if needed
        elif text == 'P1':
            finishes.append(1)  # playoff winner counts as a win (1st place)
        else:
            # Remove 'T' from tied positions and convert to integer
            finishes.append(int(text.replace('T', '')))
            
    return finishes

def parse_strokes_gained(row_data):
    """Parse strokes gained statistics"""
    stats = {}
    sg_cols = ['Rounds', 'sg_off_tee', 'sg_approach', 'sg_around_green', 'sg_putting', 'SG: TOT']
    
    # Get all td cells with class css-deko6d (which contain the SG stats)
    cells = row_data.find_all('td', class_='css-deko6d')
    
    # Skip the first cell which contains the finish positions
    cells = cells[1:] if len(cells) > 1 else []
    
    for i, col in enumerate(sg_cols):
        if i < len(cells):
            # Find the chakra-text span within each cell
            value_span = cells[i].find('span', class_='chakra-text')
            if value_span:
                value = value_span.text.strip()
                if value == '-':
                    stats[col] = None
                else:
                    try:
                        stats[col] = int(value) if col == 'Rounds' else float(value)
                    except (ValueError, TypeError):
                        stats[col] = None
            else:
                stats[col] = None
        else:
            stats[col] = None
                    
    return stats

def extract_player_data(table):
    """Extract player form data from HTML table"""
    players = []
    
    for row in table.find_all('tr', class_=lambda x: x and 'player-' in x):
        player_data = {}
        
        # Get player name and nationality
        name_cell = row.find('span', class_='chakra-text css-hmig5c')

        if not name_cell:
            continue
        
        name = name_cell.text.strip()
        if ',' in name:
            last_name, first_name = name.split(',', 1)
            name = f"{first_name.strip()} {last_name.strip()}"
        name = fix_names(name)  # Use existing fix_names function
        player_data['Name'] = name
            
        # Get recent finishes
        finish_cell = row.find('td', class_='css-deko6d')
        if finish_cell:
            player_data['recent_finishes'] = parse_recent_finishes(finish_cell)
        
        # Get strokes gained data
        sg_cells = row.find_all('td', class_='css-deko6d')[1:]  # Skip first cell (finishes)
        sg_data = parse_strokes_gained(row)
        player_data.update(sg_data)
        
        players.append(player_data)
        
    return pd.DataFrame(players)

def format_current_form(df):
    """Format current form data for analysis"""
    # Calculate made cuts percentage
    df['made_cuts_pct'] = df['recent_finishes'].apply(
        lambda x: sum(1 for finish in x if finish is not None) / len(x) if x else 0
    )
    
    # Calculate average finish (excluding cuts)
    df['avg_finish'] = df['recent_finishes'].apply(
        lambda x: np.mean([f for f in x if f is not None]) if x else None
    )
    
    # Sort by total strokes gained
    df = df.sort_values('SG: TOT', ascending=False)
    
    return df

def get_current_form(url: str) -> Optional[pd.DataFrame]:
    """
    Main function to process current form data using Selenium.
    
    Args:
        url: URL of the PGA Tour current form page
        
    Returns:
        DataFrame containing current form data or None if scraping fails
    """
    # Set up Firefox options
    firefox_options = Options()
    firefox_options.add_argument("--headless")
    
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
    table = soup.find('table', class_='chakra-table')
    
    # Extract and format data
    df = extract_player_data(table)
    df = format_current_form(df)
    
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
    url = "https://www.pgatour.com/tournaments/2025/the-sentry/R2025016/field/current-form"
    df = get_current_form(url)
    
    if df is not None:
        print("\nTop 10 players by total strokes gained:")
        print(df[['Name', 'avg_finish', 'made_cuts_pct', 'SG: TOT']].head(10))
        timestamp = time.strftime("%Y%m%d")
        # Save to CSV
        df.to_csv(f'golfers/current_form_{timestamp}.csv', index=False)