from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.select import Select
from selenium.webdriver.firefox.options import Options
import pandas as pd
import time
import os
from io import StringIO
from utils import TOURNAMENT_LIST_2025
from pga_v5 import fix_names

options = Options()

def round_scores(driver, select2, round):
    select2.select_by_visible_text(round)
    scores = driver.page_source
    scores_pd = pd.read_html(StringIO(scores))
    df_scores = scores_pd[-1]
    df_scores = df_scores.drop(df_scores.columns[[0, 10, 20, 21]], axis=1)
    return df_scores

def get_completed_rounds_scores(url):
    """
    Get current tournament scores for all players who have completed their rounds
    
    Args:
        url (str): ESPN tournament URL or tournament ID
    
    Returns:
        pd.DataFrame: DataFrame with player names and their scores for completed rounds
    """
    # Handle URL formatting
    url = str(url)  # Convert to string in case an integer ID is passed
    if not url.startswith('http'):
        url = f'https://www.espn.com/golf/leaderboard?tournamentId={url}'
    
    # Initialize webdriver
    driver = webdriver.Firefox(executable_path='/usr/local/bin/geckodriver', 
                             service_log_path=os.path.devnull, 
                             options=options)
    
    # Get raw leaderboard data
    raw_data = pd.read_html(url)[0]
    driver.get(url)
    
    # Initialize results dataframe
    df_scores = pd.DataFrame(columns=["Name", "R1", "R2", "R3", "R4", "Total"])
    
    # Process each player
    for index, row in raw_data.iterrows():
        player = fix_names(row['PLAYER'])
        try:
            # Click on player name
            element = driver.find_element(By.XPATH, f'// a[contains(text(), "{player}")]')
            element.click()
            time.sleep(1)
            
            # Get player's detailed scores
            select = driver.find_element(By.CLASS_NAME, 'Leaderboard__Player__Detail')
            select2 = Select(select.find_element(By.CLASS_NAME, 'dropdown__select'))
            
            # Initialize round scores
            round_totals = {'R1': None, 'R2': None, 'R3': None, 'R4': None}
            
            # Try to get scores for each round
            for round_num in range(1, 5):
                try:
                    round_data = round_scores(driver, select2, f"Round {round_num}")
                    # Sum the scores for the round, excluding any "-" or incomplete holes
                    round_total = sum(pd.to_numeric(round_data.iloc[1], errors='coerce').fillna(0))
                    if round_total > 0:  # Only include if round has actual scores
                        round_totals[f'R{round_num}'] = round_total
                except:
                    break
            
            # Calculate total of completed rounds
            total_score = sum(score for score in round_totals.values() if score is not None)
            
            # Add player data to dataframe
            player_data = {
                "Name": player.lower(),
                "R1": round_totals['R1'],
                "R2": round_totals['R2'],
                "R3": round_totals['R3'],
                "R4": round_totals['R4'],
                "Total": total_score
            }
            df_scores = pd.concat([df_scores, pd.DataFrame([player_data])], ignore_index=True)
            
            # Close player detail view
            element.click()
            
        except Exception as e:
            print(f"Error processing {player}: {str(e)}")
            continue
    
    # Clean up
    driver.close()
    driver.quit()
    
    return df_scores

if __name__ == "__main__":
    tournament_name = "Texas_Children's_Houston_Open"
    tournament_id = TOURNAMENT_LIST_2025[tournament_name]["ID"]
    scores = get_completed_rounds_scores(tournament_id)
    scores.to_csv(f'2025/{tournament_name}/current_tournament_scores.csv', index=False)
