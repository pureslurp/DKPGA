#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 18 20:17:02 2021

@author: seanraymor
"""
from audioop import getsample
from dataclasses import replace
from itertools import count

import numpy as np
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import warnings
import selenium
from selenium.webdriver.support.select import Select
from selenium import webdriver
from selenium.webdriver.firefox.options import Options
import pandas as pd
warnings.simplefilter(action='ignore', category=Warning)
import requests
import time
import math
from bs4 import BeautifulSoup
import sys
sys.path.insert(0, "/Users/seanraymor/Documents/Python Scripts/DKPGA")
from pga_dk_scoring import *
from os import listdir

options = Options()

OptimizedLineup = pd.DataFrame(columns=['Player1','Player2','Player3','Player4','Player5','Player6','TOT','Salary'])
MaximizedLineup = pd.DataFrame(columns=['Player1','Player2','Player3','Player4','Player5','Player6','TOT','Salary'])
NewMaximizedLineup = pd.DataFrame(columns=['Player1','Player2','Player3','Player4','Player5','Player6','TOT','Salary'])

def find_top_players_for(df: pd.DataFrame, df_merge: pd.DataFrame):
    '''A function that finds players that were used more than twice in a series of lineups given

    Args:
        df (DataFrame): the dataframe of lineups that need to be counted
        df_merge (DataFrame): the origin dataframe that contains the player, salary, score, and value
    
    Returns:
        topPlayers (DataFrame): A dataframe of players that were used more than twice
    
    '''
    j = 0
    topTier = pd.DataFrame(columns=['Count'])
    for index, row in df.iterrows():
        lineup = getID(row[0:6], df_merge)
        for x in lineup:
            temp = df_merge.loc[x]['Name + ID']
            temp = temp.iloc[0]
            topTier.loc[j] = temp
            j = j + 1
            
    topTier = topTier[topTier.groupby('Count')['Count'].transform('size') > 2]
    topPlayers = pd.DataFrame(topTier['Count'].value_counts())
    topPlayers['Name + ID'] = topPlayers.index
    
    return topPlayers

def split_odds(data: str):
    '''A function that converts odds as a string to an integer (e.g. +500 to 500)

    Args:
        data (str): The odds represented as a string

    Return:
        data (int): The odds converted to an integer, returns nan if invalid
    '''
    try:
        data = data.split('+')
        data = int(data[1])
    except:
        data = np.nan
    return data

def odds_name(data: str):
    '''A function that converts a the name from the pga dynamic website into a name that is recognized by draftkings

    Args:
        data (str): The name of the player as it is displayed in the pga html.

    Returns:
        name (str): The name in the correct format for draftkings, return nan if invalid
    '''
    try:
        data = data.split(' ')
        if len(data) > 2:
            if data[0][-1] == '.':
                name = data[0][:-2] + ' ' + data[1].strip() + ' ' + data[2].strip()
            else:
                name = data[0].strip() + ' ' + data[1][:-2] + ' ' + data[2].strip()
        else:
            if data[1] == 'Smith':
                name = data[0][:-3] + ' ' + data[1].strip()
            elif data[1] == 'Pan' or data[1] == 'Poston':
                name = data[0][:-4] + ' ' + data[1].strip()
            else:
                name = data[0][:-2] + ' ' + data[1].strip()
    except:
        name = np.nan
    
    return name

def pga_odds_pga(df_merge: pd.DataFrame, upper_bound: int = 15):
    '''A function that scrapes the odds of each player from the pga website and merges into the origin dataframe

    Args:
        df_merge (DataFrame): The origin dataframe that contains the player, salary, score, value
        upper_bound (int): The amount of points you want to award to the player with the best odds, scaled down to the worst player at 0

    Returns:
        dk_merge (DataFrame): The origin dataframe with an extra column that has the odds of each player ranked
    '''
    driver = webdriver.Firefox(service_log_path=os.path.devnull, options=options)
    driver.get('https://www.pgatour.com/odds.html#/')
    driver.implicitly_wait(120)
    time.sleep(3)
    result = driver.page_source
    driver.close()
    driver.quit()   
    dfs = pd.read_html(result)
    dfs = dfs[1]
    dfs.drop(['Pos','TotTotal','Thru','RdRound'], axis=1, inplace=True)
    
    dfs['Odds'] = dfs['Odds'].apply(lambda x: split_odds(x))
    dfs['Player'] = dfs['Player'].apply(lambda x: odds_name(x))
    dfs.rename(columns={'Player':'Name'}, inplace=True)
    dfs = dfs.dropna()
    dfs['Name'] = dfs['Name'].apply(lambda x: series_lower(x))
    dk_merge = pd.merge(df_merge, dfs, how='left', on='Name')


    oddsRank = dk_merge['Odds'].rank(pct=True, ascending=False)

    dk_merge['Odds'].fillna(0)
    dk_merge['Odds'] = oddsRank * upper_bound


    dk_merge.sort_values(by='Odds',ascending=True,inplace=True)

    return dk_merge

def drop_players_lower_than(df, salaryThreshold):
    '''A function that drops players from a dataframe that have a salary less than what is specified

    Args:
        df (DataFrame): the dataframe that contains the players and their salaries
        salaryThreshold (int): the salary cut off to be used

    Results:
        df (DataFrame): A new dataframe that only contains players above the salary threshold
    '''
    return df[df['Salary'] >= salaryThreshold]

def objective(x: list, df_merge: pd.DataFrame):
    '''Calculate the total points of a lineup (find objective)'''
    p0 = []
    for iden in x:
        p0.append(float(df_merge.loc[iden]['Total']))
    return sum(p0)

def constraint(x: list, df_merge: pd.DataFrame):
    '''Check if a lineup is within the DK salary range 50_000'''
    s0 = []
    for iden in x:
        s0.append(float(df_merge.loc[int(iden)]['Salary']))
    if (50000 - sum(s0)) > 0:
        return True
    else:
        return False

def genIter(df_merge: pd.DataFrame):
    '''Generate a random lineup'''
    r0 = []
    while len(r0) < 6:
        r = np.random.randint(0,high=len(df_merge))
        if r not in r0:
            r0.append(r)
    return r0

def getNames(lineup: list, df_merge: pd.DataFrame):
    '''Convert IDs into Names'''
    n0 = []
    for iden in lineup:
        n0.append(df_merge.loc[int(iden)]['Name + ID'])
    return n0

def get_salary(lineup: list, df_merge: pd.DataFrame):
    '''get the total salary for a particular lineup'''
    s0 = []
    for iden in lineup:
        s0.append(float(df_merge.loc[iden]['Salary']))
    return sum(s0)

def rewrite(data: list):
    '''convert string to integer in dataframe'''
    try:
        if data[0] == 'T':
            output = int(data[1:])
        else:
            output = int(data)
    except:
        output = 0
    return output

def getID(lineup: list, df_merge: pd.DataFrame):
    '''convert lineup names to ID'''
    i0 = []
    for iden in lineup: 
        i0.append(df_merge[df_merge['Name + ID'] == iden].index)
    return i0

def optimize(df: pd.DataFrame, salary: int, budget: int):
    '''Optimize a lineup by checking for a more valuable player within price range

        Args:
            df (dataFrame): the current dataFrame of players
            salary (int): Current players salary
            budget (int): Available salary to spend above current player
        
        Return:
            player (string): Most valuable player in salary range
    '''
    upper_bound = int(salary+(min(budget,1000)))
    lower_bound = int(salary) - 1000
    df.sort_values(by=['Value'],ascending = False, inplace=True)
    window = df[(df['Salary'] <= upper_bound) & (df['Salary'] > lower_bound)]
    return window.iloc[0]['Name + ID']

def maximize(df: pd.DataFrame, salary: int, budget: int):     
    '''Maximize a lineup by checking for a player that scores more points within price range

        Args:
            df (dataFrame): the current dataFrame of players
            salary (int): Current players salary
            budget (int): Available salary to spend above current player
        
        Return:
            player (string): Player with most points within salary range
    '''
    upper_bound = int(salary+(min(budget,1000)))
    lower_bound = int(salary) - 500
    df.sort_values(by=['Total'],ascending = False, inplace=True)
    window = df[(df['Salary'] <= upper_bound) & (df['Salary'] > lower_bound)]
    return window.iloc[0]['Name + ID']

def replace_outlier(df: pd.DataFrame, salary: int, budget: int, maxNames: list):
    '''A function that finds a replacement player because of an excess amount of budget available'''
    upperbound = int(salary) + budget
    lowerbound = int(salary) + 1000
    df = df[~df.loc[:, 'Name + ID'].isin(maxNames)]
    df.sort_values(by=['Total'], ascending = False, inplace=True)
    window = df[(df['Salary'] <= upperbound) & (df['Salary'] > lowerbound)]
    return window.iloc[0]['Name + ID']

def find_lowest_salary(lineup: list, df_merge: pd.DataFrame):
    '''A function that finds the lowest salary from a lineup'''
    low = 15000
    for player in lineup:
        p_row = df_merge.loc[df_merge['Name + ID'] == player]
        sal = p_row.iloc[0]['Salary']
        if sal < low:
            low = sal
            lowPlayer = p_row.iloc[0]['Name + ID']
    return lowPlayer


def remove_outliers_main(topTierLineup: pd.DataFrame, df_merge: pd.DataFrame):
    '''A function that replaces players from lineups that have an excess budget (i.e. greater than $2000 available'''
    for index, row in topTierLineup.iterrows():
        maxNames = [row[0],row[1],row[2],row[3],row[4],row[5]]
        ID = getID(maxNames, df_merge)
        
        budget = 50000 - get_salary(getID(maxNames, df_merge), df_merge)
        if budget >= 2000:
            lowID = df_merge[df_merge['Name + ID'] == find_lowest_salary(maxNames, df_merge)].index
            try:
                replacement = replace_outlier(df_merge, df_merge.loc[lowID]['Salary'], budget, maxNames)
            except:
                print('no available replacement')
                replacement = find_lowest_salary(maxNames, df_merge)
            if replacement not in maxNames:
                maxNames[maxNames.index(find_lowest_salary(maxNames, df_merge))] = replacement
                print(f'Replacing {find_lowest_salary(maxNames, df_merge)} with {replacement} at index {index}')
            else:
                print('no available replacement - dup')
        
        maxNameID = getID(maxNames, df_merge)
        maxNames.append(objective(maxNameID, df_merge))
        maxNames.append(get_salary(maxNameID, df_merge))
        MaximizedLineup.loc[index] = maxNames
        MaximizedLineup.sort_values(by='TOT', ascending=False, inplace=True)
        MaximizedLineup.drop_duplicates(subset=["TOT"],inplace=True)    
    
    return MaximizedLineup

def calculate_oversub_count(current_df: pd.DataFrame, df_merge: pd.DataFrame, sub: float=0.66, csv: bool=False):
    '''A function that counts the amount of times a player is used in a series of lineups, retured as a percentage

    Args:
        current_df (DataFrame): The df that contains the lineups that will be counted
        df_merge (DataFrame): The df that contains the players, salary, scores, and value
        sub (float): the limit that is used to determine if a player is oversubscribed to
        csv (bool): a boolean that dictates if you'd like to return all players (true) or just the ones that are being oversubscribed to

    Return:
        countPlayers or overSubList: Returns a dataframe with all player counts or a list of the oversubscribed players, respectively.
    
    '''
    countPlayers = pd.DataFrame(columns=['Name + ID', 'Salary', 'Value'])
    countPlayers['Name + ID'] = df_merge['Name + ID']
    countPlayers['Salary'] = df_merge['Salary']
    countPlayers['Value'] = df_merge['Value']
    overSubList = []
    tp = find_top_players_for(current_df, df_merge)
    countPlayers = pd.merge(countPlayers, tp, how='left',on='Name + ID')
    
    countPlayers['Count'] = countPlayers['Count'] / len(current_df.index)
    countPlayers['Count'] = countPlayers['Count'].fillna(0)
    if csv:
        return countPlayers
    overSubDF = countPlayers[countPlayers['Count'] > 0.66]
    overSubList = overSubDF['Name + ID'].tolist()
    return overSubList

def optimize_ownership(current_df: pd.DataFrame, df_merge: pd.DataFrame):
    '''A function that replaces players that are oversubscribed to'''
    current_df.sort_values(by='TOT', ascending=True, inplace=True)
    prev_ownership = []
    for index, row in current_df.iterrows():
        maxNames = [row[0],row[1],row[2],row[3],row[4],row[5]]
        budget = 50000 - get_salary(getID(maxNames, df_merge), df_merge)
        curr_ownership = calculate_oversub_count(current_df, df_merge)
        if set(prev_ownership) != set(curr_ownership):
            print(curr_ownership)
        prev_ownership = curr_ownership
        exclude = list(set(curr_ownership + maxNames))
        df_Sub = df_merge[~df_merge['Name + ID'].isin(exclude)]
        for player in range(0, len(maxNames)):
            if row[player] in curr_ownership:
                ID = getID(maxNames, df_merge)
                if maximize(df_Sub, df_merge.loc[ID[player]]['Salary'],budget) not in maxNames:
                    maxNames[player] = maximize(df_Sub, df_merge.loc[ID[player]]['Salary'],budget)
                    budget = 50000 - get_salary(getID(maxNames, df_merge), df_merge)

        maxNameID = getID(maxNames, df_merge)
        maxNames.append(objective(maxNameID, df_merge))
        maxNames.append(get_salary(maxNameID, df_merge))
        current_df.loc[index] = maxNames


    current_df.sort_values(by='TOT', ascending=False, inplace=True)

    return current_df

def duplicates(x):
    '''Check for duplicates in the lineup'''
    duplicates = False
    elemOfList = list(x)
    for elem in elemOfList:
        if elemOfList.count(elem) > 1:
            print('drop')
            duplicates = True
        
    return duplicates

def optimize_main(topTierLineup: pd.DataFrame, df_merge: pd.DataFrame):
    '''Iterate over lineups to try to optimize all players (maximize value)

        Args: 
            topTierLineup (DataFrame): lineup from a row of dataframe as a list

        Return:
            Optimizedlineup (DataFrame): lineup with maximized value    
    '''
    for index, row in topTierLineup.iterrows():
        maxNames = [row[0],row[1],row[2],row[3],row[4],row[5]]
        budget = 50000 - get_salary(getID(maxNames, df_merge), df_merge)
        for player in range(0,len(maxNames)):
            ID = getID(maxNames, df_merge)
            if optimize(df_merge, df_merge.loc[ID[player]]['Salary'],budget) not in maxNames:
                print(f"replacing {df_merge.loc[ID[player]]['Salary']} with {optimize(df_merge, df_merge.loc[ID[player]]['Salary'],budget)}")
                maxNames[player] = optimize(df_merge, df_merge.loc[ID[player]]['Salary'],budget)
                budget = 50000 - get_salary(getID(maxNames, df_merge), df_merge)

        maxNameID = getID(maxNames, df_merge)
        maxNames.append(objective(maxNameID, df_merge))
        maxNames.append(get_salary(maxNameID, df_merge))
        OptimizedLineup.loc[index] = maxNames
        
    for index, row in OptimizedLineup.iterrows():
        if duplicates(row) == True:
            OptimizedLineup.drop(index, inplace = True)

    OptimizedLineup.reset_index(drop=True, inplace=True)
    
    return OptimizedLineup

def maximize_main(OptimizedLinup: pd.DataFrame, df_merge: pd.DataFrame):
    '''Iterate over lineups to try to maximize all players (maximize points)

        Args: 
            OptimizedLineup (DataFrame): lineup from a row of dataframe as a list
            df_merge (DataFrame): DK PGA main dataFrame

        Return:
            Maximizedlineup (DataFrame): lineup with maximized points    
    '''
    for index, row in OptimizedLinup.iterrows():
        maxNames = [row[0],row[1],row[2],row[3],row[4],row[5]]
        
        budget = 50000 - get_salary(getID(maxNames, df_merge), df_merge)
        
        for player in range(0,len(maxNames)):
            ID = getID(maxNames, df_merge)
            if maximize(df_merge, df_merge.loc[ID[player]]['Salary'],budget) not in maxNames:
                print(f"replacing {df_merge.loc[ID[player]]['Salary']} with {maximize(df_merge, df_merge.loc[ID[player]]['Salary'],budget)}")
                maxNames[player] = maximize(df_merge, df_merge.loc[ID[player]]['Salary'],budget)
                budget = 50000 - get_salary(getID(maxNames, df_merge), df_merge)
        
        maxNameID = getID(maxNames, df_merge)
        maxNames.append(objective(maxNameID, df_merge))
        maxNames.append(get_salary(maxNameID, df_merge))
        MaximizedLineup.loc[index] = maxNames

    
    MaximizedLineup.reset_index(drop=True, inplace=True)
    MaximizedLineup.sort_values(by='TOT', ascending=False, inplace=True)
    MaximizedLineup.drop_duplicates(subset=["TOT"],inplace=True)
    
    return MaximizedLineup

#needs disposition
def delete_unused_columns(df_merge: pd.DataFrame):
    '''A function that deletes columns that only contian 0s or NaN'''
    col_list = []
    for (columnName, columnData) in df_merge.iteritems():
        if sum(columnData.to_numeric.values) == 0:
            col_list.append(columnName)

def fix_player_name(name: str):
    '''A function that rearranges a string containing the players name from last name, first name to first name, last name (e.g. Woods Tiger->Tiger Woods)'''
    name_spl = name.split(' ')
    ln_strip = name_spl[0].strip()
    fn_strip = name_spl[1].strip()
    full_name = fn_strip + ' ' + ln_strip
    return full_name

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

def past_results(df_merge: pd.DataFrame, url: str, lowerBound: float=0, upperBound: float=4, playoff: bool=False, pr_i: int=0):
    '''Check for past tournament results 
    
    Args:
        df_merge (DataFrame): The current dataFrame that is storing all the draftkings data
        url (string): The url for the past results that will be webscraped (recommend to be espn link)
        lowerBound (int): The lowerBound of the scale to be used to weight the scores
        upperBound (int): The upperBound of the scale to be used to weight the scores
        playoff (Bool): Indication if the past event was in a playoff or not (needed to make sure we grab the right table)
        pr_i (int): If you are specifying multiple past events, indicate which iteration you are on (1st = 0, 2nd = 1)

    Returns:
        df_merge(DataFrame): The dataFrame storing data after past results data is applied
    '''
    dk_pastResults = pd.read_html(url)
    dk_pastResults = dk_pastResults[-1]
    dk_pastResults[f'POS{pr_i}'] = dk_pastResults['POS'].apply(lambda x: pos_rewrite(x))
    dk_pastResults.drop(['POS','SCORE','R1','R2','R3','R4','EARNINGS','FEDEX PTS','TOT'],axis=1,inplace=True)
    dk_pastResults.rename(columns={'PLAYER':'Name'}, inplace=True)
    dk_pastResults['Name'] = dk_pastResults['Name'].apply(lambda x: series_lower(x))
    dk_pastResults[f'POS{pr_i}'] = dk_pastResults[f'POS{pr_i}'].fillna(99)
    df_merge = pd.merge(df_merge, dk_pastResults,how='left',on='Name')
    df_merge[f'POS{pr_i}'] = df_merge[f'POS{pr_i}'].fillna(98)
    df_merge.sort_values(by=[f'POS{pr_i}'],inplace=True)
    count = df_merge[f'POS{pr_i}'].count()
    min_val = min(df_merge[f'POS{pr_i}'].rank(pct=True, ascending=False))
    pastResultRank = (df_merge[f'POS{pr_i}'].rank(pct=True, ascending=False)-(min_val)) * df_merge[f'POS{pr_i}'].rank(pct=True, ascending=False)/(1-min_val)
    df_merge[f'POS{pr_i}'] = pastResultRank * upperBound + lowerBound

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
    else:
        return data.lower()

def series_lower(data: str):
    '''A function that converts a string to a lower case, while checking for errors'''
    name_fix = check_spelling_errors(data).strip()
    return name_fix

def DK_csv_assignemnt(path: str, name: str, lowerBound:float=0, upperBound:float=0):
    '''find exported csv from DK and assign it to dataFrame with upper and lower bound values'''
    df = pd.read_csv('{}CSVs/{}'.format(path,name))
    df.drop(['Position','ID','Roster Position','Game Info', 'TeamAbbrev'],axis=1,inplace=True)
    avgPointsScale = np.linspace(lowerBound, upperBound,len(df['AvgPointsPerGame'].dropna()))
    df.sort_values(by=['AvgPointsPerGame'],inplace=True)
    df['AvgPointsPerGame'] = avgPointsScale
    df['Name'] = df['Name'].apply(lambda x: series_lower(x))
    return df

def find_last_x_majors(player, events):
    event_url_array = []
    event_list = [353222, 353226, 353232, 243410, 243414, 243418, 243010, 219478, 219333]
    for le in event_list:
        if len(event_url_array) < events:
            url = f'https://www.espn.com/golf/leaderboard/_/tournamentId/401{le}'
            dk_pastResults = pd.read_html(url)
            dk_pastResults = dk_pastResults[-1]
            dk_pastResults['PLAYER'] = dk_pastResults['PLAYER'].apply(lambda x: series_lower(x))
            if player in dk_pastResults['PLAYER'].values:
                data = dk_pastResults[dk_pastResults['PLAYER'] == player]
                if str(data['SCORE'].values[0]) == 'WD' or str(data['SCORE'].values[0]) == 'DQ' or str(data['SCORE'].values[0]) == 'MDF':
                    print(f"{player} WD or DQ")
                else:
                    event_url_array.append(url)
        else:
            break
    return event_url_array

def find_last_x_events(player, events):
    event_url_array = []
    event_list = [20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 31, 32, 73, 94, 55, 54, 53, 39, 38, 37, 36]
    for le in event_list:
        if len(event_url_array) < events:
            url = f'https://www.espn.com/golf/leaderboard/_/tournamentId/4013532{le}'
            dk_pastResults = pd.read_html(url)
            dk_pastResults = dk_pastResults[-1]
            dk_pastResults['PLAYER'] = dk_pastResults['PLAYER'].apply(lambda x: series_lower(x))
            if player in dk_pastResults['PLAYER'].values:
                data = dk_pastResults[dk_pastResults['PLAYER'] == player]
                if str(data['SCORE'].values[0]) == 'WD' or str(data['SCORE'].values[0]) == 'DQ'  or str(data['SCORE'].values[0]) == 'MDF':
                    print(f"{player} WD or DQ")
                else:
                    event_url_array.append(url)
        else:
            break
    return event_url_array

def find_csv_filenames(path, suffix='.csv'):
    filenames = listdir(path)
    return [filename for filename in filenames if filename.endswith(suffix)]

def tournament_id_from_csv(csv_name):
    csv_name = csv_name.split('_')
    id = csv_name[-1]
    id = id.split('.')
    id = id[0]
    return(id)

def last_x_majors_dk_points(df_merge, events=6, upper_bound=15):
    new_df = pd.DataFrame(columns=['Name',f'L{events}MajPts'])
    cnt = 0
    len_merge = len(df_merge)
    for index, row in df_merge.iterrows():
        cnt += 1
        tot = 0
        player = row["Name"]
        urls = find_last_x_majors(player, events)
        if len(urls) > 0:
            for url in urls:
                t_id = tournament_id(url)
                csv_arr = []
                for csv_n in find_csv_filenames('past_results/2022'):
                    csv_arr.append(tournament_id_from_csv(csv_n))
                if t_id in csv_arr:
                    pr_df = pd.read_csv(f'past_results/2022/dk_points_id_{t_id}.csv')
                else:
                    print(f'writing {t_id} to past results')
                    pr_df = dk_points_df(url)
                pr_df['Name'] = pr_df['Name'].apply(lambda x: series_lower(x))
                nr = pr_df[pr_df["Name"] == player]
                tot += float(nr.iloc[0,1])
            if len(urls) > 2:
                tot = (tot / len(urls))
            else:
                tot = tot / 3
            new_row = [player, tot]
            if cnt % 5 == 0:
                print(f'{(cnt/len_merge)*100}% players in majors complete')
            df_len = len(new_df)
            new_df.loc[df_len] = new_row
    df_merge = pd.merge(df_merge, new_df, how='left',on='Name')
    fitRank = df_merge[f'L{events}MajPts'] / df_merge[f'L{events}MajPts'].max()
    df_merge[f'L{events}MajPts'] = fitRank * upper_bound
    df_merge[f'L{events}MajPts'] = df_merge[f'L{events}MajPts'].fillna(0)
    return df_merge

def last_x_events_dk_points(df_merge, events=10, upper_bound=15):
    new_df = pd.DataFrame(columns=['Name',f'L{events}Pts'])
    cnt = 0
    len_merge = len(df_merge)
    for index, row in df_merge.iterrows():
        cnt += 1
        tot = 0
        player = row["Name"]
        urls = find_last_x_events(player, events)
        if len(urls) > 0:
            for url in urls:
                t_id = tournament_id(url)
                csv_arr = []
                for csv_n in find_csv_filenames('past_results/2022'):
                    csv_arr.append(tournament_id_from_csv(csv_n))
                if t_id in csv_arr:
                    pr_df = pd.read_csv(f'past_results/2022/dk_points_id_{t_id}.csv')
                else:
                    print(f'writing {t_id} to past results')
                    pr_df = dk_points_df(url)
                pr_df['Name'] = pr_df['Name'].apply(lambda x: series_lower(x))
                nr = pr_df[pr_df["Name"] == player]
                tot += float(nr.iloc[0,1])
            if len(urls) > 2:
                tot = (tot / len(urls))
            else:
                tot = tot / 3
            new_row = [player, tot]
            if cnt % 5 == 0:
                print(f'{(cnt/len_merge)*100}% players in events complete')
            df_len = len(new_df)
            new_df.loc[df_len] = new_row
    df_merge = pd.merge(df_merge, new_df, how='left',on='Name')
    fitRank = df_merge[f'L{events}Pts'] / df_merge[f'L{events}Pts'].max()
    df_merge[f'L{events}Pts'] = fitRank * upper_bound
    df_merge[f'L{events}Pts'] = df_merge[f'L{events}Pts'].fillna(0)
    return df_merge


def last_x_event_points(df_merge, events=10, upper_bound=15):
    new_df = pd.DataFrame(columns=['Name',f'L{events}Pts'])
    cnt = 0
    len_merge = len(df_merge)
    for index, row in df_merge.iterrows():
        cnt += 1
        tot = 0
        player = row["Name"]
        urls = find_last_x_events(player, events)
        if len(urls) > 0:
            for url in urls:
                pr_df = past_results(df_merge, url, upperBound=2)
                nr = pr_df[pr_df["Name"] == player]
                tot = tot + float(nr['POS0'])
            tot = (tot / len(urls))
            new_row = [player, tot]
            if cnt % 5 == 0:
                print(f'{(cnt/len_merge)*100}% players in events complete')
            df_len = len(new_df)
            new_df.loc[df_len] = new_row
    df_merge = pd.merge(df_merge, new_df, how='left',on='Name')
    fitRank = df_merge[f'L{events}Pts'].rank(pct=True, ascending=True)
    df_merge[f'L{events}Pts'] = fitRank * upper_bound
    df_merge[f'L{events}Pts'] = df_merge[f'L{events}Pts'].fillna(0)
    return df_merge


def df_total_and_reformat(df_merge: pd.DataFrame):
    '''reformate dataFrame to be easily exported'''
    column_list = list(df_merge.columns[3:])
    df_merge['Total'] = df_merge[column_list].sum(axis=1)
    df_merge['Value'] = (df_merge['Total'] / df_merge['Salary']) * 1000
    df_merge = df_merge[['Name + ID', 'Salary', 'Total', 'Value']]
    df_merge.sort_values(by='Salary',ascending=False,inplace=True)
    df_merge.drop_duplicates(inplace=True, ignore_index=True)
    return df_merge
