import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
import re
from urllib.parse import urlparse
import os
from selenium.webdriver.firefox.options import Options

from utils import TOURNAMENT_LIST_2025, fix_names

def clean_score(score):
    """Convert score text to numeric value."""
    if score == 'E' or score == '-':
        return 0
    try:
        return int(score.replace('+', '').replace('-', '-'))
    except:
        return None

def parse_round_score(score):
    """Convert round score to numeric value."""
    if score == '-' or not score:
        return None
    try:
        return int(score)
    except:
        return None

def clean_position(position):
    """Clean position text by removing 'T' prefix and converting WD to CUT."""
    if position in ['CUT', 'WD']:
        return 'CUT'
    return position.replace('T', '')

def extract_player_data(row):
    """Extract player data from a table row."""
    cells = row.find_all(['td', 'th'])
    
    if len(cells) < 8:  # Skip rows without enough data
        return None
        
    # Extract position and clean it
    position = clean_position(cells[0].text.strip())
    if position == 'CUT':  # Updated condition since WD will now be CUT
        status = position
    else:
        status = 'Active'
        
    # Extract player name and apply fix_names
    player_cell = cells[2]
    player_name = fix_names(player_cell.find('span', class_='css-hmig5c').text.strip())
    
    # Extract country
    country_img = player_cell.find('img', class_='css-c0eggu')
    country = country_img['alt'] if country_img else None
    
    # Extract total score
    total_score = clean_score(cells[3].text.strip())
    
    # Extract round scores
    r1 = parse_round_score(cells[6].text.strip())
    r2 = parse_round_score(cells[7].text.strip())
    r3 = parse_round_score(cells[8].text.strip())
    r4 = parse_round_score(cells[9].text.strip())
    
    return {
        'Position': position,
        'Status': status,
        'Player': player_name,
        'Country': country,
        'Total': total_score,
        'R1': r1,
        'R2': r2,
        'R3': r3,
        'R4': r4
    }

def validate_pgatour_url(url):
    """Validate if the URL is from pgatour.com."""
    parsed_url = urlparse(url)
    if parsed_url.netloc != 'www.pgatour.com':
        raise ValueError("URL must be from www.pgatour.com")
    return True

def setup_driver():
    """Set up and return a Firefox WebDriver instance."""
    options = Options()
    options.add_argument("--headless")
    return webdriver.Firefox(options=options)

def fetch_leaderboard(url):
    """Fetch leaderboard data from a PGA Tour URL using Selenium."""
    try:
        # Validate URL
        validate_pgatour_url(url)
        
        # Set up the driver
        driver = setup_driver()
        
        try:
            # Load the page
            driver.get(url)
            
            # Wait for the table to load
            wait = WebDriverWait(driver, 10)
            table = wait.until(EC.presence_of_element_located((By.TAG_NAME, 'table')))
            
            # Process the rows
            rows = table.find_elements(By.TAG_NAME, 'tr')
            data = []
            
            for row in rows:
                # Skip header and advertisement rows
                if 'css-4g6ai3' in row.get_attribute('class') or 'no-print' in row.get_attribute('class'):
                    continue
                    
                # Extract player data using Selenium elements
                cells = row.find_elements(By.TAG_NAME, 'td')
                if len(cells) < 8:  # Skip rows without enough data
                    continue
                
                # Extract only position and player name (with fix_names)
                position = cells[0].text.strip()
                player_name = fix_names(cells[2].find_element(By.CLASS_NAME, 'css-hmig5c').text.strip())
                
                data.append({
                    'Player': player_name,
                    'Position': position,
                })
            
            return pd.DataFrame(data)
            
        finally:
            driver.quit()
            
    except TimeoutException:
        raise Exception("Timeout waiting for leaderboard table to load")
    except Exception as e:
        raise Exception(f"Error processing leaderboard: {str(e)}")

def main(urls, tourney):
    """Main function to fetch and combine historical leaderboard data."""
    try:
        # Create empty dictionary to store results
        all_results = {}
        
        # Process each URL and year
        for year, url in urls.items():
            print(f"Fetching {year} data...")
            df = fetch_leaderboard(url)
            if df is not None:
                # Convert to dictionary with Player as key and Position as value
                year_results = df.set_index('Player')['Position'].to_dict()
                all_results[year] = year_results
        
        # Create DataFrame from all results
        combined_df = pd.DataFrame(all_results)
        
        # Clean positions by removing 'T' and converting WD to CUT
        for col in combined_df.columns:
            combined_df[col] = combined_df[col].apply(lambda x: clean_position(x) if pd.notna(x) else x)
        
        # Calculate measured_years and made_cuts_pct
        measured_years = combined_df.notna().sum(axis=1)
        made_cuts = combined_df.apply(lambda x: x[x.notna()].str.match(r'^(CUT|WD)$').sum(), axis=1)
        made_cuts_pct = ((measured_years - made_cuts) / measured_years).round(3)
        
        combined_df['measured_years'] = measured_years
        combined_df['made_cuts_pct'] = made_cuts_pct
        
        # Calculate average finish (treating CUT/WD as 65th place)
        def convert_position(pos):
            if pd.isna(pos) or pos in ['CUT', 'WD']:
                return 65
            return int(pos)
            
        position_df = combined_df[combined_df.columns[:-2]].applymap(convert_position)
        combined_df['avg_finish'] = position_df.mean(axis=1).round(1)
        
        # Rename columns to match desired format
        column_mapping = {
            '2025': '24',
            '2024': '2022-23',
            '2023': '2021-22',
            '2022': '2020-21',
            '2021': '2019-20'
        }
        combined_df = combined_df.rename(columns=column_mapping)
        
        # Rename the index to 'Name'
        combined_df.index.name = 'Name'
        
        # Create directory if it doesn't exist
        save_dir = os.path.join('2025', tourney)
        os.makedirs(save_dir, exist_ok=True)
        
        # Save DataFrame to CSV
        save_path = os.path.join(save_dir, 'course_history.csv')
        combined_df.to_csv(save_path)
        print(f"Data saved to {save_path}")
        
        return combined_df
    except Exception as e:
        print(f"Error: {str(e)}")
        return None

if __name__ == "__main__":
    # Example usage
    tourney = "The_Genesis_Invitational"
    urls = {
        '2025': f"https://www.pgatour.com/tournaments/2025/farmers-insurance-open/R2025004",
        '2024': "https://www.pgatour.com/tournaments/2024/farmers-insurance-open/R2024004",
        '2023': "https://www.pgatour.com/tournaments/2023/farmers-insurance-open/R2023004",
        '2022': "https://www.pgatour.com/tournaments/2022/farmers-insurance-open/R2022004",
        '2021': "https://www.pgatour.com/tournaments/2021/farmers-insurance-open/R2021004"
    }
    
    df = main(urls, tourney)
    if df is not None:
        print("\nHistorical Results:")
        print(df)