# Standard library imports
import time
from typing import Optional

# Third-party imports
import pandas as pd
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

from pga_v5 import fix_names
from utils import TOURNAMENT_LIST_2024, TOURNAMENT_LIST_2025

def parse_stat_value(cell_data):
    """Parse statistical values and their corresponding scores"""
    stat = {}
    
    # Find the main value span which contains both rank and score
    value_span = cell_data.find('span', class_='chakra-text css-1dmexvw')
    
    if value_span:
        # Get the rank (number before the non-breaking space)
        rank_text = value_span.text.split('\xa0')[0]
        stat['rank'] = int(rank_text) if rank_text != '-' else None
        
        # Get the score from the nested p element
        score_p = value_span.find('p')
        if score_p:
            score_text = score_p.text.strip().strip('()')
            try:
                stat['score'] = float(score_text)
            except ValueError:
                stat['score'] = None
        else:
            stat['score'] = None
    else:
        stat['rank'] = None
        stat['score'] = None
    
    return stat

def clean_header_text(header_text: str) -> str:
    """Clean and standardize header text for column names"""
    # Convert to lowercase
    header = header_text.lower()
    
    # Replace special characters and spaces with underscores
    header = (header.replace(' - ', '_')
                   .replace(' ', '_')
                   .replace('-', '_')
                   .replace(':', '')
                   .replace('(', '')
                   .replace(')', '')
                   .replace('%', 'pct')
                   .replace('.', '')
                   .replace('/', '_'))
    
    # Remove any duplicate underscores
    while '__' in header:
        header = header.replace('__', '_')
    
    # Remove leading/trailing underscores
    header = header.strip('_')
    
    return header

def extract_player_data(table):
    """Extract player data from HTML table"""
    players = []
    
    # Get header row to identify stat columns
    headers = ['Name', 'projected_course_fit']  # First two headers are fixed
    
    # Find the header row and extract stat headers
    header_row = table.find('tr', class_='css-1odhsqu')
    if header_row:
        # Get all stat headers (skip Player and Projected Course Fit)
        for header_cell in header_row.find_all('th')[2:]:
            button = header_cell.find('button', class_='css-p11w71')
            if button:
                header_text = button.text.strip()
                clean_header = clean_header_text(header_text)
                headers.append(clean_header)
    
    # Process each player row
    for row in table.find_all('tr', class_=lambda x: x and 'player-' in x):
        # Get all data cells (already skips the name cell)
        data_cells = row.find_all('td', class_=['css-1v8y6rt', 'css-deko6d', 'css-4lb9jb', 'css-1vziul4'])
        
        # Skip players with insufficient data
        rank_cell = data_cells[0]
        rank_span = rank_cell.find('span', class_='chakra-text css-1dmexvw')
        if not rank_span or rank_span.text.strip() == '-':
            continue
            
        player_data = {}
        
        # Get player name
        name_cell = row.find('span', class_='chakra-text css-hmig5c')
        if not name_cell:
            continue
            
        name = name_cell.text.strip()
        name = fix_names(name)
        player_data['Name'] = name
        
        # Process the overall rank (projected course fit)
        player_data['projected_course_fit'] = float(rank_span.text.strip())
        
        # Process the remaining stat cells
        for header, cell in zip(headers[2:], data_cells[1:]):
            value_span = cell.find('span', class_='chakra-text css-1dmexvw')
            if value_span:
                score_p = value_span.find('p', class_=['chakra-text css-4ysu3v', 'chakra-text css-boq55u'])
                if score_p:
                    score_text = score_p.text.strip('()').strip()
                    try:
                        player_data[header] = float(score_text)
                    except ValueError:
                        player_data[header] = None
                else:
                    player_data[header] = None
            else:
                player_data[header] = None
        
        players.append(player_data)
    
    df = pd.DataFrame(players)
    return df

def format_course_fit(df):
    """Format course fit data for analysis"""
    if df.empty:
        return df
    
    # Sort by projected course fit score
    if 'projected_course_fit' in df.columns:
        df = df.sort_values('projected_course_fit', ascending=True)
    
    return df

def get_course_fit(url: str) -> Optional[pd.DataFrame]:
    """Main function to process course fit data using Selenium."""
    firefox_options = Options()
    firefox_options.add_argument("--headless")
    
    # try:
    driver = webdriver.Firefox(options=firefox_options)
    driver.get(url)
    
    wait = WebDriverWait(driver, 20)
    table = wait.until(EC.presence_of_element_located((By.CLASS_NAME, "chakra-table")))
    
    html_content = driver.page_source
    soup = BeautifulSoup(html_content, 'html.parser')
    table = soup.find('table', class_='chakra-table')
    
    if not table:
        raise ValueError("Could not find table in page content")
    
    # Extract and format data
    df = extract_player_data(table)
    
    if df.empty:
        raise ValueError("No data extracted from table")
        
    df = format_course_fit(df)
    
    return df
    
    # except Exception as e:
    #     print(f"Error scraping data: {e}")
    #     return None
        
    # finally:
    #     try:
    #         driver.quit()
    #     except:
    #         pass

if __name__ == "__main__":
    TOURNEY = "The_Sentry"
    # Get the URL from tournament list
    pga_url = TOURNAMENT_LIST_2025[TOURNEY]['pga-url']
    url = f"https://www.pgatour.com/tournaments/2025/{pga_url}/field/course-fit"
    print(url)
    df = get_course_fit(url)
    
    if df is not None:
        print("\nShape of DataFrame:", df.shape)
        print("\nColumns in DataFrame:", df.columns.tolist())
        
        # Print sample of full data
        print("\nSample of data (first 5 rows, all columns):")
        pd.set_option('display.max_columns', None)  # Show all columns
        pd.set_option('display.width', None)  # Don't wrap output
        print(df.head())
        
        # Save to CSV in tournament directory
        df.to_csv(f'2025/{TOURNEY}/course_fit.csv', index=False)
        print(f"\nSaved complete data to 2025/{TOURNEY}/course_fit.csv")
    else:
        print("Failed to retrieve course fit data")