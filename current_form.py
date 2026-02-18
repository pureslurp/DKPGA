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
from utils import TOURNAMENT_LIST_2026

def parse_recent_finishes(row_data):
    """Parse the last 5 tournament finishes"""
    try:
        finishes = []
        finish_texts = row_data.find_all('p', class_=['css-1eypu3w', 'css-1o72dcd', 'css-z0g4ki'])
        
        for finish in finish_texts:
            text = finish.text.strip()
            
            if text == 'CUT':
                finishes.append('CUT')
            elif text in ['', '-']:
                finishes.append(None)
            elif text == 'P1':
                finishes.append(1)
            else:
                try:
                    finishes.append(int(text.replace('T', '')))
                except ValueError:
                    finishes.append(None)
                    
        return finishes
        
    except Exception as e:
        print(f"Error in parse_recent_finishes: {str(e)}")
        print(f"Full error details: ", e.__class__.__name__)
        import traceback
        print(traceback.format_exc())
        raise

def parse_strokes_gained(row_data):
    """Parse strokes gained statistics"""
    stats = {}
    sg_cols = ['Rounds', 'sg_off_tee', 'sg_approach', 'sg_around_green', 'sg_putting']
    
    # Get all td cells in order
    all_tds = row_data.find_all('td')
    
    # Based on HTML structure:
    # td[0] = finishes (css-18rncge with p tags)
    # td[1] = empty (css-vv5ndq)
    # td[2] = rounds (css-18rncge)
    # td[3] = made_cuts_pct (css-18rncge) - we'll skip this
    # td[4] = sg_off_tee (css-18rncge)
    # td[5] = sg_approach (css-18rncge)
    # td[6] = sg_around_green (css-18rncge)
    # td[7] = sg_putting (css-86bwll or css-18rncge)
    
    # Map indices: Rounds=2, sg_off_tee=4, sg_approach=5, sg_around_green=6, sg_putting=7
    col_indices = [2, 4, 5, 6, 7]
    
    for i, col in enumerate(sg_cols):
        cell_index = col_indices[i] if i < len(col_indices) else None
        
        if cell_index is not None and cell_index < len(all_tds):
            cell = all_tds[cell_index]
            value_span = cell.find('span', class_='chakra-text')
            if value_span:
                value = value_span.text.strip()
                if value in ['-', '']:
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
    
    # Calculate sg_total as sum of all SG components
    sg_components = [stats[col] for col in ['sg_off_tee', 'sg_approach', 'sg_around_green', 'sg_putting']]
    stats['sg_total'] = sum(x for x in sg_components if x is not None) if any(x is not None for x in sg_components) else None
                    
    return stats

def extract_player_data(table):
    """Extract player form data from HTML table"""
    try:
        rows = table.find_all('tr', class_=lambda x: x and 'player-' in x)
        
        players = []
        for row in rows:
            try:
                # Try multiple selectors for player name
                name_cell = (row.find('span', class_='css-1v9q6zy') or
                           row.find('span', class_='css-qvuvio') or
                           row.find('span', class_='chakra-text css-qvuvio') or
                           row.find('span', class_='css-hmig5c'))
                
                if not name_cell:
                    continue
                
                name = name_cell.text.strip()
                if ',' in name:
                    last_name, first_name = name.split(',', 1)
                    name = f"{first_name.strip()} {last_name.strip()}"
                name = fix_names(name)
                
                player_data = {'Name': name}
                
                # Find finish cell - look for td with finishes (contains p tags with finish text)
                all_tds = row.find_all('td', class_='css-18rncge')
                finish_cell = None
                for td in all_tds:
                    # Check if this td contains finish information (has p tags with finish classes)
                    if td.find('p', class_=lambda x: x and ('css-1eypu3w' in x or 'css-1o72dcd' in x or 'css-z0g4ki' in x)):
                        finish_cell = td
                        break
                
                if finish_cell:
                    finishes = parse_recent_finishes(finish_cell)
                    player_data['recent_finishes'] = finishes
                else:
                    player_data['recent_finishes'] = []
                
                sg_data = parse_strokes_gained(row)
                player_data.update(sg_data)
                
                players.append(player_data)
                
            except Exception as e:
                print(f"Error processing row: {str(e)}")
                print(f"Full error details: ", e.__class__.__name__)
                import traceback
                print(traceback.format_exc())
                continue
        
        if not players:
            return pd.DataFrame()
        
        return pd.DataFrame(players)
        
    except Exception as e:
        print(f"Error in extract_player_data: {str(e)}")
        print(f"Full error details: ", e.__class__.__name__)
        import traceback
        print(traceback.format_exc())
        raise

def format_current_form(df):
    """Format current form data for analysis"""
    # Check if DataFrame is empty
    if df.empty:
        return df
    
    # Calculate made cuts percentage (exclude None values, which mean didn't play)
    def made_cuts(x):
        if not isinstance(x, list):
            return 0
        numerator = sum(1 for finish in x if finish is not None and finish != 'CUT')
        denominator = sum(1 for finish in x if finish is not None)
        return numerator / denominator if denominator > 0 else 0
    
    # Apply calculations - ensure we return a Series, not DataFrame
    df['made_cuts_pct'] = df['recent_finishes'].apply(made_cuts)
    
    # Calculate average finish (excluding cuts and tournaments not played)
    df['avg_finish'] = df['recent_finishes'].apply(
        lambda x: np.mean([f for f in x if isinstance(f, (int, float))]) if x else None
    )
    
    # Sort by total strokes gained
    df = df.sort_values('sg_total', ascending=False)
    
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
        table = soup.find('table', class_='chakra-table')
        
        # Extract and format data
        df = extract_player_data(table)
        df = format_current_form(df)
        
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
    TOURNEY = "AT&T_Pebble_Beach_Pro-Am"
    pga_url = TOURNAMENT_LIST_2026[TOURNEY]["pga-url"]
    url = f"https://www.pgatour.com/tournaments/2026/{pga_url}/field/current-form"
    df = get_current_form(url)
    
    if df is not None:
        print("\nTop 10 players by total strokes gained:")
        print(df[['Name', 'avg_finish', 'made_cuts_pct', 'sg_total']].head(10))
        timestamp = time.strftime("%Y%m%d")
        # Save to CSV
        df.to_csv(f'2026/{TOURNEY}/current_form.csv', index=False)