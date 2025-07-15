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
        stat_headers = header_row.find_all('th')[2:]
        for header_cell in stat_headers:
            button = header_cell.find('button', class_=lambda x: x and 'css-' in x)
            if button:
                header_text = button.text.strip()
                clean_header = clean_header_text(header_text)
                headers.append(clean_header)
    
    # Process each player row
    player_rows = table.find_all('tr', class_=lambda x: x and ('player-' in x or 'css-' in x) and x != 'css-1odhsqu')
    
    for row in player_rows:
        # Get all data cells
        data_cells = row.find_all('td', class_=lambda x: x and 'css-' in x)
        if not data_cells:
            continue
        
        # Get player name first
        name_cell = row.find('span', class_='chakra-text css-qvuvio')
        if not name_cell:
            continue
            
        name = name_cell.text.strip()
        name = fix_names(name)
        player_data = {'Name': name}
        
        # Get projected course fit - updated to handle the first column
        if len(data_cells) > 1:  # Make sure we have at least 2 cells
            # First cell is strokes_gained_off_the_tee_par_5
            first_stat = data_cells[0].find('span', class_='chakra-text css-1dmexvw')
            if first_stat:
                try:
                    player_data['strokes_gained_off_the_tee_par_5'] = float(first_stat.text.strip())
                except (ValueError, AttributeError):
                    player_data['strokes_gained_off_the_tee_par_5'] = None
            
            # Second cell is projected_course_fit
            second_stat = data_cells[1].find('span', class_='chakra-text css-1dmexvw')
            if second_stat:
                try:
                    player_data['projected_course_fit'] = float(second_stat.text.strip())
                except (ValueError, AttributeError):
                    player_data['projected_course_fit'] = None
        
        # Process the remaining stat cells
        for header, cell in zip(headers[2:], data_cells[2:]):
            value_span = cell.find('span', class_='chakra-text css-1dmexvw')
            if value_span:
                score_p = value_span.find('p', class_=lambda x: x and 'chakra-text' in x)
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
    
    return pd.DataFrame(players)

def format_course_fit(df):
    """Format course fit data for analysis"""
    if df.empty:
        return df
    
    # Calculate 70th percentile of non-null values
    percentile_70 = df['projected_course_fit'].dropna().quantile(0.7)
    
    # Fill missing values with 70th percentile
    df['projected_course_fit'] = df['projected_course_fit'].fillna(percentile_70)
    
    # Sort by projected course fit score
    df = df.sort_values('projected_course_fit', ascending=True)
    
    return df

def get_course_fit(url: str) -> Optional[pd.DataFrame]:
    """Main function to process course fit data using Selenium."""
    firefox_options = Options()
    firefox_options.add_argument("--headless")
    
    driver = webdriver.Firefox(options=firefox_options)
    driver.get(url)
    
    try:
        wait = WebDriverWait(driver, 20)
        table = wait.until(EC.presence_of_element_located((By.CLASS_NAME, "chakra-table")))
        
        html_content = driver.page_source
        soup = BeautifulSoup(html_content, 'html.parser')
        table = soup.find('table', class_='chakra-table')
        
        if not table:
            print("Debug: Table not found in page content")
            raise ValueError("Could not find table in page content")
        
        print("Debug: Found table, attempting to extract data")
        # Extract and format data
        df = extract_player_data(table)
        
        print(f"Debug: DataFrame shape after extraction: {df.shape if df is not None else 'None'}")
        
        if df.empty:
            print("Debug: DataFrame is empty after extraction")
            raise ValueError("No data extracted from table")
            
        df = format_course_fit(df)
        
        return df
    
    except Exception as e:
        print(f"Debug: Error occurred: {str(e)}")
        return None
        
    finally:
        try:
            driver.quit()
        except:
            pass

if __name__ == "__main__":
    TOURNEY = "Genesis_Scottish_Open"
    # Get the URL from tournament list
    pga_url = TOURNAMENT_LIST_2025[TOURNEY]['pga-url']
    url = f"https://www.pgatour.com/tournaments/2025/{pga_url}/field?feature=course-fit"
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