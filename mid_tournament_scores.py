from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.select import Select
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.firefox.service import Service
import pandas as pd
import time
import os
from io import StringIO
from utils import TOURNAMENT_LIST_2026
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
    
    # Initialize webdriver with modern Service object
    service = Service('/usr/local/bin/geckodriver')
    driver = webdriver.Firefox(service=service, options=options)
    
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
            
            # Based on the HTML structure, the cells are:
            # cells[0] = expand/collapse icon
            # cells[1] = position (1, T2, etc.)
            # cells[2] = player name cell (contains flag and name)
            # cells[3] = current score
            # cells[4] = total score
            # cells[5] = tee time
            # cells[6] = R1 score
            # cells[7] = R2 score
            # cells[8] = R3 score
            # cells[9] = R4 score
            # cells[10] = total score (same as cells[4])
            
            round_scores = {
                'R1': cells[6].text if len(cells) > 6 else '--',
                'R2': cells[7].text if len(cells) > 7 else '--',
                'R3': cells[8].text if len(cells) > 8 else '--',
                'R4': cells[9].text if len(cells) > 9 else '--'
            }
            
            # Convert '--' or '-' to None and calculate total
            for key in round_scores:
                if round_scores[key] in ['--', '-'] or not round_scores[key]:
                    round_scores[key] = None
                else:
                    try:
                        round_scores[key] = int(round_scores[key])
                    except ValueError:
                        round_scores[key] = None
            
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
    tournament_name = "Sony_Open_in_Hawaii"
    tournament_id = TOURNAMENT_LIST_2026[tournament_name]["ID"]
    scores = get_completed_rounds_scores(tournament_id)
    scores.to_csv(f'2026/{tournament_name}/current_tournament_scores.csv', index=False)
