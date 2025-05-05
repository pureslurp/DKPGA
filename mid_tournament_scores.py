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
    url = str(url)
    if not url.startswith('http'):
        url = f'https://www.espn.com/golf/leaderboard?tournamentId={url}'
    
    # Initialize webdriver
    driver = webdriver.Firefox(executable_path='/usr/local/bin/geckodriver', 
                             service_log_path=os.path.devnull, 
                             options=options)
    
    driver.get(url)
    time.sleep(2)  # Give page time to load
    
    # Find all player rows
    player_rows = driver.find_elements(By.CLASS_NAME, 'PlayerRow__Overview')
    df_scores = pd.DataFrame(columns=["Name", "R1", "R2", "R3", "R4", "Total"])
    
    for row in player_rows:
        try:
            # Extract player name from the anchor tag
            player_element = row.find_element(By.CLASS_NAME, 'leaderboard_player_name')
            player = fix_names(player_element.text)
            
            # Get all table cells in the row
            cells = row.find_elements(By.CLASS_NAME, 'Table__TD')
            
            # ESPN table structure: cells[7] = R1, cells[8] = R2, cells[9] = R3, cells[10] = R4
            round_scores = {
                'R1': cells[7].text,
                'R2': cells[8].text,
                'R3': cells[9].text,
                'R4': cells[10].text
            }
            
            # Convert '--' or '-' to None and calculate total
            for key in round_scores:
                if round_scores[key] in ['--', '-'] or not round_scores[key]:
                    round_scores[key] = None
                else:
                    round_scores[key] = int(round_scores[key])
            
            total_score = sum(score for score in round_scores.values() if score is not None)
            
            # Add player data to dataframe
            player_data = {
                "Name": player.lower(),
                "R1": round_scores['R1'],
                "R2": round_scores['R2'],
                "R3": round_scores['R3'],
                "R4": round_scores['R4'],
                "Total": total_score
            }
            df_scores = pd.concat([df_scores, pd.DataFrame([player_data])], ignore_index=True)
            
        except Exception as e:
            print(f"Error processing row: {str(e)}")
            continue
    
    # Clean up
    driver.close()
    driver.quit()
    
    return df_scores

if __name__ == "__main__":
    tournament_name = "THE_CJ_CUP_Byron_Nelson"
    tournament_id = TOURNAMENT_LIST_2025[tournament_name]["ID"]
    scores = get_completed_rounds_scores(tournament_id)
    scores.to_csv(f'2025/{tournament_name}/current_tournament_scores.csv', index=False)
