from curses import raw
import selenium
from selenium.webdriver.support.select import Select
from selenium.webdriver.common.by import By
from selenium.webdriver.firefox.options import Options
from selenium import webdriver
import time
import pandas as pd
import numpy as np
import os
import sys
from io import StringIO

options = Options()

def per_hole_scoring(score):
    '''
    Double Eagle or Better = 13
    Eagle = 8
    Birdie = 3
    Par = 0.5
    Bogey = -0.5
    Double Bogey = -1
    Worse than Double Bogey = -1
    '''
    if score is None:
        return 0
    pts = 0
    for rh in score:
        if rh is None:  # Skip incomplete holes
            continue
        if rh <= -3:  # Double Eagle or Better
            p = 13
        elif rh == -2:
            p = 8
        elif rh == -1:
            p = 3
        elif rh == 0:
            p = 0.5
        elif rh == 1:
            p = -0.5
        else:
            p = -1
        pts += p
    return pts

def streaks_and_bonuses(r1, r2, r3, r4, par):
    '''
    Streak of 3 birdies or better = 3
    Bogey Free Round = 3
    All 4 Rounds Under 70 Strokes = 5
    '''
    try:
        if r4 is not None:
            tot1 = par + sum(r1)
            tot2 = par + sum(r2)
            tot3 = par + sum(r3)
            tot4 = par + sum(r4)
            r_array = [r1, r2, r3, r4]

            if tot1 < 70 and tot2 < 70 and tot3 < 70 and tot4 < 70:
                under_70_pts = 5
            else:
                under_70_pts = 0
        elif r3 is not None:
            tot1 = par + sum(r1)
            tot2 = par + sum(r2)
            tot3 = par + sum(r3)
            r_array = [r1, r2, r3]

            if tot1 < 70 and tot2 < 70 and tot3 < 70:
                under_70_pts = 5
            else:
                under_70_pts = 0
        else:
            tot1 = par + sum(r1)
            tot2 = par + sum(r2)
            r_array = [r1, r2]

            if tot1 < 70 and tot2 < 70:
                under_70_pts = 5
            else:
                under_70_pts = 0
        
        bogey_free = 0
        for r in r_array:
            if r is None:  # Skip None rounds
                continue
            has_bogey = False
            for h in r:
                if h is None:  # Skip None holes
                    continue
                if h > 0:
                    has_bogey = True
                    break
            if not has_bogey:
                bogey_free += 3
        
        birdie_streak = 0
        for r in r_array:
            if r is None:  # Skip None rounds
                continue
            consecutive_birdies = 0
            for h in r:
                if h is None:  # Skip None holes
                    consecutive_birdies = 0
                    continue
                if h < 0:
                    consecutive_birdies += 1
                    if consecutive_birdies >= 3:
                        birdie_streak += 3
                        consecutive_birdies = 0  # Reset after counting a streak
                else:
                    consecutive_birdies = 0
        return bogey_free + birdie_streak + under_70_pts
    except:
        return 0

def hole_in_one(df_score):
    """
    Calculate hole in one bonus points
    Args:
        df_score: DataFrame containing par and score for holes
    Returns:
        int: Bonus points (5 points per hole in one)
    """
    try:
        if df_score is None:
            return 0
            
        par = df_score.iloc[0].values  # First row contains par values
        score = df_score.iloc[1].values  # Second row contains scores
        
        bonus = 0
        for hole_score, hole_par in zip(score, par):
            try:
                if float(hole_score) == 1:  # Hole in one
                    bonus += 5
            except (ValueError, TypeError):
                continue
        return bonus
    except:
        return 0


def place_points(place):
    '''
    1st = 30
    2nd = 20
    3rd = 18
    4th = 16
    5th = 14
    6th = 12
    7th = 10
    8th = 9
    9th = 8
    10th = 7
    11 - 15 = 6
    16 - 20 = 5
    21 - 25 = 4
    26 - 30 = 3
    31 - 40 = 2
    41 - 50 = 1
    '''
    if place == 1:
        pts = 30
    elif place == 2:
        pts = 20
    elif place == 3:
        pts = 18
    elif place == 4:
        pts = 16
    elif place == 5:
        pts = 14
    elif place == 6:
        pts = 12
    elif place == 7:
        pts = 10
    elif place == 8:
        pts = 9
    elif place == 9:
        pts = 8
    elif place == 10:
        pts = 7
    elif place < 16:
        pts = 6
    elif place < 21:
        pts = 5
    elif place < 26:
        pts = 4
    elif place < 31:
        pts = 3
    elif place < 41:
        pts = 2
    elif place < 51:
        pts = 1
    else:
        pts = 0
    return pts
    
def pos_rewrite(x: str):
    '''A function that converts a position from a str to an integer, including ties'''
    data = x.split(' ')
    pos = data[-1]
    if pos[0] == 'T':
        pos = int(pos[1:])
    elif pos == 'CUT':
        pos = 100
    else:
        try:
            pos = int(pos)
        except:
            pos = np.nan
    return pos

def find_par(df):
    winner = df.iloc[0]
    try:
        # Only attempt conversion if not 'CUT'
        if isinstance(winner['TOT'], str) and winner['TOT'].upper() == 'CUT':
            return None
        return (int(winner['TOT']) - int(winner['SCORE'])) / 4
    except (ValueError, TypeError):
        return None

def find_net_score(df_score):
    try:
        hole = np.arange(0, 18, 1)
        par = df_score.iloc[0].values
        score = df_score.iloc[1].values
        r_net_score = []
        for h in hole:
            try:
                h_score = float(score[h]) - float(par[h])
                r_net_score.append(h_score)
            except (ValueError, TypeError):
                r_net_score.append(None)
    except:
        return None
    return r_net_score

def validate_tournament_conditions(raw_data, r1, r2, r3, r4):
    """
    Validates tournament conditions and returns a dictionary of validation results
    Returns:
        dict: Contains validation flags and any adjusted values
    """
    validation = {
        'is_valid': True,
        'tournament_complete': True,
        'rounds_completed': 0,
        'is_playoff': False,
        'player_status': 'active',  # active, withdrawn, or dq
        'adjusted_par': None,
        'error_message': None
    }
    
    try:
        # Check tournament status and rounds completed
        rounds = [r1, r2, r3, r4]
        validation['rounds_completed'] = sum(1 for r in rounds if r is not None)
        
        # Determine if tournament was shortened or extended
        expected_rounds = 4  # Default for most tournaments
        if validation['rounds_completed'] != expected_rounds:
            validation['tournament_complete'] = False
        
        # Check for playoff (indicated by 'P' in position or extra holes)
        if 'P' in str(raw_data['POS'].iloc[0]):
            validation['is_playoff'] = True
        
        # Check player status (WD or DQ)
        pos = raw_data['POS'].iloc[0]
        if isinstance(pos, str):
            if 'WD' in pos:
                validation['player_status'] = 'withdrawn'
            elif 'DQ' in pos:
                validation['player_status'] = 'dq'
        
        # Validate par calculation
        try:
            winner_score = int(raw_data.iloc[0]['SCORE'])
            winner_total = int(raw_data.iloc[0]['TOT'])
            calculated_par = (winner_total - winner_score) / validation['rounds_completed']
            # Check if par is a reasonable number (usually between 69-73)
            if 65 <= calculated_par <= 75:
                validation['adjusted_par'] = calculated_par
            else:
                validation['error_message'] = "Calculated par seems incorrect"
                validation['is_valid'] = False
        except (ValueError, KeyError, ZeroDivisionError) as e:
            validation['error_message'] = f"Error calculating par: {str(e)}"
            validation['is_valid'] = False
        
    except Exception as e:
        validation['is_valid'] = False
        validation['error_message'] = str(e)
    
    return validation

def adjust_scoring_for_conditions(validation, score_dict):
    """
    Adjusts scoring based on tournament conditions
    Args:
        validation: Dictionary from validate_tournament_conditions
        score_dict: Dictionary containing various score components
    Returns:
        float: Adjusted total score
    """
    adjusted_score = 0
    
    # Only count position points if player completed tournament
    if validation['player_status'] == 'active':
        adjusted_score += score_dict.get('position_points', 0)
    
    # Add per-hole scoring
    adjusted_score += score_dict.get('hole_points', 0)
    
    # Handle bonuses based on tournament conditions
    if validation['tournament_complete']:
        # Add under-70 bonus only if tournament wasn't shortened
        adjusted_score += score_dict.get('under_70_bonus', 0)
    
    # Add bogey-free round bonuses
    adjusted_score += score_dict.get('bogey_free_bonus', 0)
    
    # Add birdie streak bonuses
    adjusted_score += score_dict.get('birdie_streak_bonus', 0)
    
    # Add hole-in-one bonuses
    adjusted_score += score_dict.get('hole_in_one_bonus', 0)
    
    return adjusted_score

def round_dk_score(r1_score, r2_score, r3_score, r4_score, pos, par, raw_data):
    """
    Calculate DraftKings golf score with validation
    """
    # Validate tournament conditions
    validation = validate_tournament_conditions(raw_data, r1_score, r2_score, r3_score, r4_score)
    if not validation['is_valid']:
        print(f"Warning: {validation['error_message']}")
    
    # Calculate individual score components
    score_dict = {
        'position_points': place_points(pos),
        'hole_points': 0,
        'under_70_bonus': 0,
        'bogey_free_bonus': 0,
        'birdie_streak_bonus': 0,
        'hole_in_one_bonus': 0
    }
    
    # Calculate per-hole scoring
    tournament_score = [r for r in [r1_score, r2_score, r3_score, r4_score] if r is not None]
    for r_score in tournament_score:
        net_score = find_net_score(r_score)
        score_dict['hole_points'] += per_hole_scoring(net_score)
        score_dict['hole_in_one_bonus'] += hole_in_one(r_score)
    
    # Calculate streaks and bonuses
    bonus_score = streaks_and_bonuses(
        find_net_score(r1_score),
        find_net_score(r2_score),
        find_net_score(r3_score),
        find_net_score(r4_score),
        validation.get('adjusted_par', par)
    )
    
    # Split bonus score into components (assuming implementation of new helper functions)
    score_dict['under_70_bonus'] = bonus_score  # This should be split into individual components
    
    # Apply adjustments based on tournament conditions
    final_score = adjust_scoring_for_conditions(validation, score_dict)
    
    return final_score

def round_scores(driver, select2, round):
    select2.select_by_visible_text(round)
    scores = driver.page_source
    scores_pd = pd.read_html(StringIO(scores))
    df_scores = scores_pd[-1]
    df_scores = df_scores.drop(df_scores.columns[[0, 10, 20, 21]], axis=1)
    return df_scores

def world_rank_rewrite(data):
    pos = data
    try:
        if pos[0] == 'T':
            pos = int(pos[1:])
        else:
            pos = int(pos)
    except:
        pos = np.nan
    return pos

def world_rank_rewrite_new(data):
    pos = data[10:]
    try:
        if pos[0] == 'T':
            pos = int(pos[1:])
        else:
            pos = int(pos)
    except:
        pos = np.nan
    return pos

from selenium.webdriver.support.ui import Select
from selenium.webdriver.common.by import By
from selenium import webdriver
import pandas as pd
import time

def check_world_rank(df_merge):
    # Open the website
    url = 'https://www.pgatour.com/stats/detail/186'
    driver = webdriver.Firefox()
    driver.get(url)
    time.sleep(5)  # Wait for the page to load

    # Find the "Season" button using text
    season_button = driver.find_element(By.XPATH, "//button[contains(@aria-label, 'Season')]")
    season_button.click()
    time.sleep(2)

    # Select the "2022-2023" season
    season_option = driver.find_element(By.XPATH, "//button[text()='2022-2023']")
    season_option.click()
    time.sleep(5)  # Wait for the page to update with the correct data
    
    # Get the page source after selecting the season
    result = driver.page_source
    
    # Close the driver
    driver.close()
    driver.quit()
    
    # Parse the data from the page using StringIO
    data = pd.read_html(StringIO(result))[0]
    
    # Process the data
    data['Name'] = data['Player'].apply(lambda x: series_lower_new(x))
    print(data.head())
    
    data['Rank'] = data["Total Points"].rank(ascending=False)
    data = data[["Rank", "Name"]]
    
    # Merge with the existing dataframe
    df_merge = pd.merge(df_merge, data, how='left', on='Name')
    print(df_merge.head())
    
    return df_merge


def check_spelling_errors(data: str):
    '''A function that checks for common misspells'''
    if data.lower() == 'matthew fitzpatrick':
        return 'matt fitzpatrick'
    elif data.lower() == 'tyrell hatton':
        return 'tyrrell hatton'
    elif data.lower() == 'taylor gooch':
        return 'talor gooch'
    elif data.lower() == 'cam champ':
        return 'cameron champ'
    elif data.lower() == 'cam davis':
        return 'cameron davis'
    elif data.lower() == 'sung-jae im':
        return 'sungjae im'
    elif data.lower() == 'hason kokrak':
        return 'jason kokrak'
    elif data.lower() == 'sebastián muñoz':
        return 'sebastian munoz'
    elif data.lower() == 'k.h. lee' or data.lower() == 'kyounghoon lee' or data.lower() == 'lee kyoung-hoon':
        return 'kyoung-hoon lee'
    elif data.lower() == 'charles howell':
        return 'charles howell iii'
    elif data.lower() == 'sung-hoon kang' or data.lower() == 's.h. kang':
        return 'sung kang'
    elif data.lower() == 'charl schwarztel':
        return 'charl schwartzel'
    elif data.lower() == 'roger slaon':
        return 'roger sloan'
    elif data.lower() == 'scott pierce':
        return 'scott piercy'
    elif data.lower() == 'vincent whaley':
        return 'vince whaley'
    elif data.lower() == 'stephan jaegar':
        return 'stephan jaeger'
    elif data.lower() == 'mathhias schwab':
        return 'matthias schwab'
    elif data.lower() == 'kang sung-hoon':
        return 'sung kang'
    elif data.lower() == 'jorda spieth':
        return 'jordan spieth'
    elif data.lower() == 'christopher gotterup':
        return 'chris gotterup'
    elif data.lower() == 'louis oosthuzien':
        return 'louis oosthuizen'
    elif data.lower() == 'sungmoon bae':
        return 'sung-moon bae'
    elif data.lower() == 'ludvig åberg':
        return 'ludvig aberg'
    else:
        return data.lower()

def series_lower(data: str):
    '''A function that converts a string to a lower case, while checking for errors'''
    name_fix = check_spelling_errors(data).strip()
    return name_fix

def series_lower_new(data: str):
    
    try:
        name_fix = check_spelling_errors(data).strip()
    except:
        name_fix = data
    return name_fix

#MAIN

#USER INPUT
# url = "https://www.espn.com/golf/leaderboard?tournamentId=401353220"
##

def tournament_id(url):
    url = url.split('=')
    id = url[-1]
    return(id)

def dk_points_df(url):
    url = f'https://www.espn.com/golf/leaderboard?tournamentId={url}'
    print(url)
    driver = webdriver.Firefox(executable_path='/usr/local/bin/geckodriver',service_log_path=os.path.devnull, options=options)
    raw_data = pd.read_html(url)
    raw_data = raw_data[-1]
    t_id = tournament_id(url)
    raw_data['POS'] = raw_data['POS'].apply(lambda x: pos_rewrite(x))
    #raw_data = raw_data.drop(['EARNINGS', 'FEDEX PTS'], axis=1)
    par = find_par(raw_data)
    driver.get(url)
    df_total_points = pd.DataFrame(columns=["Name", "DK Score"])

    for index, row in raw_data.iterrows():
        player = row['PLAYER']
        pos = row["POS"]

        # get element 
        element = driver.find_element(By.XPATH, f'// a[contains(text(), "{player}")]')

        # click the element
        element.click()
        time.sleep(1)
        
        if pos < 100:
            select = driver.find_element(By.CLASS_NAME, 'Leaderboard__Player__Detail')
            select2 = Select(select.find_element(By.CLASS_NAME, 'dropdown__select'))
            r1 = round_scores(driver, select2, "Round 1")
            r1 = r1.mask(r1 == "-", 4)
            r2 = round_scores(driver, select2, "Round 2")
            r3 = round_scores(driver, select2, "Round 3")
            try:
                r4 = round_scores(driver, select2, "Round 4")
            except:
                r4 = None
                
            row_data = pd.DataFrame([{
                "Name": series_lower(row['PLAYER']), 
                "DK Score": round_dk_score(r1, r2, r3, r4, pos, par, raw_data.loc[raw_data['PLAYER'] == row['PLAYER']])
            }])
            df_total_points = pd.concat([df_total_points, row_data], ignore_index=True)
        else:
            try:
                select = driver.find_element(By.CLASS_NAME, 'Leaderboard__Player__Detail')
                select2 = Select(select.find_element(By.CLASS_NAME, 'dropdown__select'))
                r1 = round_scores(driver, select2, "Round 1")
                r1 = r1.mask(r1 == "-", 4)
                r2 = round_scores(driver, select2, "Round 2")
                r3 = None
                r4 = None
                row_data = pd.DataFrame([{"Name": series_lower(row['PLAYER']), 
                                        "DK Score": round_dk_score(r1, r2, r3, r4, pos, par, raw_data.loc[raw_data['PLAYER'] == row['PLAYER']])}])
                df_total_points = pd.concat([df_total_points, row_data], ignore_index=True)
            except:
                print(f'Player {row["PLAYER"]} is N/A')
                pass

        
        element.click()

    df_total_points = check_world_rank(df_total_points)
    # Save to 2026 directory (update year as needed)
    os.makedirs('past_results/2026', exist_ok=True)
    df_total_points.to_csv(f'past_results/2026/dk_points_id_{t_id}.csv', index=False)
    driver.close()
    driver.quit() 
    return df_total_points
##