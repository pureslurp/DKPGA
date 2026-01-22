# Standard library imports
import os
from typing import Optional

# Third-party imports
import pandas as pd
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

from utils import TOURNAMENT_LIST_2026
from pga_v5 import fix_names

def get_tee_times(url: str) -> Optional[pd.DataFrame]:
    """
    Fetch tee time data for a PGA Tour tournament using Selenium and merge with current scores.
    
    Args:
        url: URL of the PGA Tour tee times page
        
    Returns:
        DataFrame containing tee time data and current scores or None if scraping fails
    """
    # Set up Firefox options
    firefox_options = Options()
    firefox_options.add_argument("--headless")
    
    try:
        # Initialize the driver
        driver = webdriver.Firefox(options=firefox_options)
        driver.get(url)
        
        # Wait for tee time rows to load
        wait = WebDriverWait(driver, 20)
        wait.until(EC.presence_of_element_located((By.CLASS_NAME, "css-79elbk")))
        
        # Get the page source after JavaScript has loaded
        html_content = driver.page_source
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Extract tee time data
        rows = soup.find_all('tr', class_='css-79elbk')
        tee_times = []
        
        print(f"Found {len(rows)} tee time rows")  # Debug print
        
        for row in rows:
            time_cell = row.find('span', class_='css-1r55614')
            # Find all player divs in the row
            player_divs = row.find_all('div', class_='css-1bntj9o')
            
            if time_cell and player_divs:
                time = time_cell.text.strip()
                print(f"Processing tee time: {time}")  # Debug print
                
                # Process each player in the group
                for player_div in player_divs:
                    # Find the player name in the text content
                    player_name_elem = player_div.find('span', class_='css-qvuvio')
                    if player_name_elem:
                        player_name = player_name_elem.text.strip()
                        player_name = fix_names(player_name)
                        print(f"  Found player: {player_name}")  # Debug print
                        tee_times.append([time, player_name])
                    else:
                        print(f"  No player name found")  # Debug print
            else:
                print(f"Missing time cell or player divs")  # Debug print
        
        print(f"Total tee times collected: {len(tee_times)}")  # Debug print
        
        if not tee_times:
            print("No tee times were collected, raising exception")
            raise Exception("No tee times were found in the parsed HTML")
        
        # Create DataFrame
        df = pd.DataFrame(tee_times, columns=["Tee Time", "Player Name"])
        
        # Convert player names to lowercase for matching
        df['Player Name'] = df['Player Name'].str.lower()
        
        # Try to merge with current scores first
        try:
            scores_df = pd.read_csv(f'2026/{TOURNEY}/current_tournament_scores.csv')
            scores_df['Name'] = scores_df['Name'].str.lower()
            
            # Merge with scores
            df = pd.merge(df, scores_df, left_on='Player Name', right_on='Name', how='left')
            
            # Drop duplicate Name column
            df = df.drop('Name', axis=1)
        except Exception as e:
            print(f"Warning: Could not merge current scores: {e}")
        
        # Now handle the wave calculations
        df['Time_Obj'] = pd.to_datetime(df['Tee Time'], format='%I:%M %p').dt.time
        
        # Calculate time difference between consecutive tee times
        time_diffs = []
        for i in range(len(df)):
            if i == 0:
                time_diffs.append(pd.Timedelta(0))
                continue
                
            curr_time = pd.to_datetime(df['Tee Time'].iloc[i], format='%I:%M %p')
            prev_time = pd.to_datetime(df['Tee Time'].iloc[i-1], format='%I:%M %p')
            time_diffs.append(curr_time - prev_time)
        
        # Find the largest gap (should be between AM and PM waves)
        time_diff_seconds = [td.total_seconds() for td in time_diffs]
        if any(td > 3600 for td in time_diff_seconds):
            # If there's a gap > 1 hour, use that as the split
            wave_split_idx = time_diffs.index(max([td for td in time_diffs if td.total_seconds() > 3600]))
        else:
            # If no clear AM/PM split, just split the field in half
            wave_split_idx = len(df) // 2
            
        # Assign waves based on the split
        df['Wave'] = 'AM'
        df.loc[wave_split_idx:, 'Wave'] = 'PM'
        
        # Drop the temporary time object column
        df = df.drop('Time_Obj', axis=1)
        
        # Reorder columns
        columns_order = ['Tee Time', 'Player Name', 'Wave', 'R1', 'R2', 'R3', 'R4', 'Total']
        df = df[columns_order]
        
        return df
        
    except Exception as e:
        print(f"Error scraping tee time data: {e}")
        return None
        
    finally:
        # Close the browser
        try:
            driver.quit()
        except:
            pass

if __name__ == "__main__":
    # Example usage
    TOURNEY = "Sony_Open_in_Hawaii"
    tee_times_path = f'2026/{TOURNEY}/tee_times.csv'
    
    # Check if tee times file already exists
    if os.path.exists(tee_times_path):
        print("Loading existing tee times data...")
        df = pd.read_csv(tee_times_path)
    else:
        print("Fetching new tee times data...")
        url = f"https://www.pgatour.com/tournaments/2026/{TOURNAMENT_LIST_2026[TOURNEY]['pga-url']}/tee-times"
        df = get_tee_times(url)
    
    if df is not None:
        # Check if we need to update scores (if any round columns are missing or empty)
        needs_score_update = False
        for round_col in ['R1', 'R2', 'R3', 'R4']:
            if round_col not in df.columns or df[round_col].isna().all():
                needs_score_update = True
                break
        
        if needs_score_update:
            print("Updating scores data...")
            try:
                scores_df = pd.read_csv(f'2026/{TOURNEY}/current_tournament_scores.csv')
                scores_df['Name'] = scores_df['Name'].str.lower()
                
                # Drop existing score columns if they exist
                score_cols = ['R1', 'R2', 'R3', 'R4', 'Total']
                df = df.drop([col for col in score_cols if col in df.columns], axis=1)
                
                # Merge with scores
                df = pd.merge(df, scores_df[['Name'] + score_cols], 
                             left_on='Player Name', right_on='Name', how='left')
                
                # Drop duplicate Name column
                df = df.drop('Name', axis=1)
                
                # Reorder columns
                columns_order = ['Tee Time', 'Player Name', 'Wave', 'R1', 'R2', 'R3', 'R4', 'Total']
                df = df[columns_order]
                
                print("Scores updated successfully")
            except Exception as e:
                print(f"Warning: Could not merge current scores: {e}")
        else:
            print("Existing score data found, skipping update")
        
        # Only proceed with calculations if we have the required columns
        if all(col in df.columns for col in ['Wave', 'R1', 'R2']):
            # Calculate R1 wave scoring averages
            r1_am_avg = df[df['Wave'] == 'PM']['R1'].mean()
            r1_pm_avg = df[df['Wave'] == 'AM']['R1'].mean()
            r1_wave_diff = r1_am_avg - r1_pm_avg
            
            # Calculate R2 wave scoring averages (waves are switched)
            r2_am_avg = df[df['Wave'] == 'AM']['R2'].mean()  # PM wave in R1 played AM in R2
            r2_pm_avg = df[df['Wave'] == 'PM']['R2'].mean()  # AM wave in R1 played PM in R2
            r2_wave_diff = r2_am_avg - r2_pm_avg
            
            # Calculate overall wave differentials
            am_total = (r1_am_avg + r2_pm_avg)  # Total for players who played AM/PM
            pm_total = (r1_pm_avg + r2_am_avg)  # Total for players who played PM/AM
            total_wave_diff = am_total - pm_total
            
            print("\nRound 1 Wave Scoring Averages:")
            print(f"AM: {r1_am_avg:.2f}")
            print(f"PM: {r1_pm_avg:.2f}")
            print(f"Difference (AM - PM): {r1_wave_diff:.2f}")
            
            print("\nRound 2 Wave Scoring Averages:")
            print(f"AM: {r2_am_avg:.2f}")
            print(f"PM: {r2_pm_avg:.2f}")
            print(f"Difference (AM - PM): {r2_wave_diff:.2f}")
            
            print("\nOverall Wave Differentials:")
            print(f"AM/PM: {am_total:.2f}")
            print(f"PM/AM: {pm_total:.2f}")
            print(f"Total Difference (AM/PM - PM/AM): {total_wave_diff:.2f}")
            
        else:
            print("\nMissing required columns for calculations. Available columns:")
            print(df.columns.tolist())
        
        # Save updated data
        os.makedirs(f'2026/{TOURNEY}', exist_ok=True)
        df.to_csv(tee_times_path, index=False)
    else:
        print("Failed to retrieve or load tee time data")
