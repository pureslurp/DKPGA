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

def duo_data_ln(names, df_merge):
    split_names = names.split('/')
    if len(split_names) == 1:
        split_names = names.split('.')
    name1 = split_names[0].strip().lower()
    name2 = split_names[1].strip().lower()
    if name1 == 'potson':
        name1 = 'poston'
    elif name1 == 'poluter':
        name1 = 'poulter'
    elif name1 == 'stnaley':
        name1 = 'stanley'
    if name2 == 'potson':
        name2 = 'poston'
    elif name2 == 'poluter':
        name2 = 'poulter'
    elif name2 == 'stnaley':
        name2 = 'stanley'
    for index, row in df_merge.iterrows():
        name = row['Name']
        split_name = name.split(' ')
        if len(split_name) == 3:
            if split_name[1] == 'varner':
                ln = split_name[1]
            else:
                ln = split_name[1].strip() + ' ' + split_name[2].strip()
        else:
            ln = split_name[1].strip()
        if ln == name1:
            name1 = name
        elif ln == name2:
            name2 = name
        else:
            pass
    return name1, name2

def pga_odds_vegas_duo(df_merge):
    url = 'https://www.vegasinsider.com/golf/odds/futures/'  
    req = requests.get(url)
    soup = BeautifulSoup(req.content, 'html.parser')
    odds_list = soup.find_all('ul')
    odds_list = odds_list[18]
    print(odds_list)
    odds_list = odds_list.find_all('li')
    names = []
    odd_array = []
    for data in odds_list:
        data = data.text
        data = data.split('+')
        name1, name2 = duo_data_ln(data[0], df_merge)
        odds = data[1]
        names.append(name1)
        names.append(name2)
        odd_array.append(odds)
        odd_array.append(odds)
    odds_df = pd.DataFrame(columns=['Name', 'Odds'])
    odds_df['Name'] = names
    odds_df['Odds'] = odd_array
    odds_df['Name'] = odds_df['Name'].apply(lambda x: series_lower(x))
    print(odds_df.head())
    dk_merge = pd.merge(df_merge, odds_df, how='left', on='Name')
    dk_merge["Odds"] = pd.to_numeric(dk_merge["Odds"])
    dk_merge.sort_values(by='Odds',ascending=False,inplace=True)
    print(dk_merge.head())

    oddsRank = dk_merge['Odds'].rank(pct=True, ascending=False)


    dk_merge['Odds'].fillna(0)
    dk_merge['Odds'] = oddsRank * 15


    dk_merge.sort_values(by='Odds',ascending=True,inplace=True)
    print(dk_merge.head())

    return dk_merge


def pga_odds_vegas(df_merge: pd.DataFrame, upper_bound: int = 15):
    '''A function that scrapes odds data from vegas website and merges it into the main df

    Args:
        df_merge (DataFrame): The origin dataframe that contains the player, salary, score, value
        upper_bound (int): The amount of points you want to award to the player with the best odds, scaled down to the worst player at 0

    Returns:
        dk_merge (DataFrame): The origin dataframe with an extra column that has the odds of each player ranked
    '''
    url = 'https://www.vegasinsider.com/golf/odds/futures/'  
    req = requests.get(url)
    soup = BeautifulSoup(req.content, 'html.parser')
    soup = soup.main
    odds_list = soup.find_all('ul')
    odds_list = odds_list[0]
    print(odds_list)
    odds_list = odds_list.find_all('li')
    names = []
    odd_array = []
    for data in odds_list:
        data = data.text
        data = data.split('+')
        name = data[0].strip()
        odds = data[1]
        names.append(name)
        odd_array.append(odds)
    odds_df = pd.DataFrame(columns=['Name', 'Odds'])
    odds_df['Name'] = names
    odds_df['Odds'] = odd_array
    odds_df['Name'] = odds_df['Name'].apply(lambda x: series_lower(x))
    print(odds_df.head())
    dk_merge = pd.merge(df_merge, odds_df, how='left', on='Name')
    dk_merge["Odds"] = pd.to_numeric(dk_merge["Odds"])
    dk_merge.sort_values(by='Odds',ascending=False,inplace=True)
    print(dk_merge.head())

    oddsRank = dk_merge['Odds'].rank(pct=True, ascending=False)


    dk_merge['Odds'].fillna(0)
    dk_merge['Odds'] = oddsRank * upper_bound


    dk_merge.sort_values(by='Odds',ascending=True,inplace=True)
    print(dk_merge.head())

    return dk_merge




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
    
    print(dfs.head())
    print(dfs.info())

    oddsRank = dk_merge['Odds'].rank(pct=True, ascending=False)

    dk_merge['Odds'].fillna(0)
    dk_merge['Odds'] = oddsRank * upper_bound


    dk_merge.sort_values(by='Odds',ascending=True,inplace=True)

    return dk_merge

def check_last_year(data: pd.DataFrame, ly_df:pd.DataFrame, key: str):
    '''A function that checks whether each player has stats that can be used from the previous year

    Args:
        data (DataFrame): The origin dataframe that contains all players and their data
        ly_df (DataFrame): The stats from the previous year from the pga website for the given key
        key (str): The hole yardage

    Returns:
        value (float): The players averages strokes for that hole
    '''
    value = np.nan
    rookie = True
    for index, row in ly_df.iterrows():
        if row['PLAYER NAME'].lower() == data['Name']:
            if math.isnan(data[key]):
                value = row["AVG"]
                rookie = False
            else:
                value = data[key]
                rookie = False
    if rookie == True:
        try:
            value = data[key]
        except:
            value = np.nan

    
    return float(value)


def drop_players_lower_than(df, salaryThreshold):
    '''A function that drops players from a dataframe that have a salary less than what is specified

    Args:
        df (DataFrame): the dataframe that contains the players and their salaries
        salaryThreshold (int): the salary cut off to be used

    Results:
        df (DataFrame): A new dataframe that only contains players above the salary threshold
    '''
    return df[df['Salary'] >= salaryThreshold]


def getEff(df: pd.DataFrame, key: str, count: int):
    '''A function that checks the players average strokes per hole, as a rank, based onthe yardage

    Args:
        df (DataFrame): The origin dataframe that contains the players and their data
        key (str): the length of the hole
        count (int): the amount of times that length of hole appears

    Returns:
        dk_merge (DataFrame): The origin dataframe with a new column of the rank of each player based on the holes yardage
    
    '''
    pgaEff = [['125-150 Eff',3,'https://www.pgatour.com/stats/stat.02519.html'],
              ['150-175 Eff',3,'https://www.pgatour.com/stats/stat.02520.html'],
              ['175-200 Eff',3,'https://www.pgatour.com/stats/stat.02521.html'],
              ['200-225 Eff',3,'https://www.pgatour.com/stats/stat.02522.html'],
              ['225-250 Eff',3,'https://www.pgatour.com/stats/stat.02523.html'],
              ['300-350 Eff',4,'https://www.pgatour.com/stats/stat.02527.html'],
              ['350-400 Eff',4,'https://www.pgatour.com/stats/stat.02528.html'],
              ['400-450 Eff',4,'https://www.pgatour.com/stats/stat.02529.html'],
              ['450-500 Eff',4,'https://www.pgatour.com/stats/stat.02530.html'],
              ['500+ Eff',   4,'https://www.pgatour.com/stats/stat.02531.html'],
              ['500-550 Eff',5,'https://www.pgatour.com/stats/stat.02533.html'],
              ['550-600 Eff',5,'https://www.pgatour.com/stats/stat.02534.html'],
              ['600-650 Eff',5,'https://www.pgatour.com/stats/stat.02535.html']]
    
    columns = ['Range','Par','URL']
    pgaStats = pd.DataFrame(pgaEff,columns=columns)
    url = pgaStats[pgaStats['Range'] == key]['URL'].item()
    eff_df = pd.read_html(url)
    eff_df = eff_df[1]

    
    eff_df.drop(['RANK LAST WEEK','ROUNDS','TOTAL STROKES','TOTAL ATTEMPTS'], axis=1, inplace=True)
    eff_df.drop(eff_df.columns[0],axis=1,inplace=True)
    eff_df.rename(columns={'PLAYER NAME':'Name','AVG':key}, inplace=True)
    eff_df['Name'] = eff_df['Name'].apply(lambda x: series_lower(x))
    dk_merge = pd.merge(df, eff_df, how='left', on='Name')



    if count > 0:
        ly_url = url[:-4] + 'y2021.' + url[-4:]
        eff_df_ly = pd.read_html(ly_url)
        eff_df_ly = eff_df_ly[1]
        dk_merge[key] = dk_merge.apply(lambda x: check_last_year(x, eff_df_ly, key), axis=1)
    
    effRank = dk_merge[key].rank(pct=True, ascending=False)

    dk_merge[key].fillna(0)
    dk_merge[key] = effRank * count * (15/18)


    #effScale = np.concatenate((np.linspace(count*2,0,len(dk_merge[key].dropna())),np.zeros(len(dk_merge)-len(dk_merge[key].dropna()))))
    dk_merge.sort_values(by=key,ascending=True,inplace=True)
    

    return dk_merge


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

def distance(data: str):
    '''Convert feet inches into float (e.g. 12'6" = 12.5)'''
    data = data.split("'")
    data[1] = data[1][:-1]
    x = int(data[0])+(int(data[1])/12)
    return x

def eaglePerc(data: list):
    '''Calculate eagle percentage (i.e. total eagles/total par 5s)'''
    return data[1]/data[2]

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
        print(index)
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
        print(index)
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
        print(index)
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

def course_fit(df_merge: pd.DataFrame, lowerBound: float=0, upperBound:float=5):
    '''A function that scrapes data from datagolf.coms course fit tool and weights it based on the inputted bound'''
    driver = webdriver.Firefox(service_log_path=os.path.devnull, options=options)
    driver.get('https://datagolf.com/course-fit-tool')
    driver.implicitly_wait(120)
    time.sleep(3)
    result = driver.page_source
    soup = BeautifulSoup(result, "html.parser")
    #print(soup)
    course_fit_data = soup.find_all("div", class_="datarow")
    course_fit_df = pd.DataFrame(columns=['Name', 'Adjustment'])
    i = 0
    for player_row in course_fit_data:
        adj = player_row.find_all('div', class_='ev-text')
        course_fit_df.loc[i] = [player_row['name'], float(adj[-1].text)]
        i += 1
    driver.close()
    driver.quit()   
    
    course_fit_df['Adjustment'] = course_fit_df['Adjustment'].fillna(0)
    course_fit_df['Name'] = course_fit_df['Name'].apply(lambda x: series_lower(x))
    df_merge = pd.merge(df_merge, course_fit_df, how='left',on='Name')
    fitRank = df_merge['Adjustment'].rank(pct=True, ascending=True)
    df_merge['Adjustment'] = fitRank * upperBound + lowerBound
    df_merge['Adjustment'] = df_merge['Adjustment'].fillna(0)
    print(df_merge.head())
    return df_merge

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

def duo_data_fn(names):
    split_names = names.split('/')
    name1 = split_names[0].strip()
    name2 = split_names[1].strip()
    return name1, name2


def past_results_dyn_duo(df_merge, url, lowerBound=0, upperBound=4, pr_i=0):
    driver = webdriver.Firefox(service_log_path=os.path.devnull, options=options)
    driver.get(url)
    driver.implicitly_wait(120)
    time.sleep(10)

    select = Select(driver.find_element_by_id('pastResultsYearSelector'))
    if pr_i == 0:
        select.select_by_value('2021.018')
    else:
        select.select_by_value('2019.018')
    time.sleep(5)
    result = driver.page_source
    dk_pastResults = pd.read_html(result)
    dk_pastResults = dk_pastResults[1]
    new_col = ['PLAYER', 'POS', 'R1', 'R2', 'R3', 'R4', 'TOTAL SCORE', 'TO PAR', 'OFFICIAL MONEY', 'FEDEXCUP POINTS']
    dk_pastResults.columns = new_col
    dk_pastResults[f'POS{pr_i}'] = dk_pastResults['POS'].apply(lambda x: pos_rewrite(x))
    dk_pastResults.drop(['POS', 'R1', 'R2', 'R3', 'R4', 'TOTAL SCORE', 'TO PAR', 'OFFICIAL MONEY', 'FEDEXCUP POINTS'],axis=1,inplace=True)
    #print(dk_pastResults.head())
    driver.close()
    driver.quit()  
    #dk_pastResults[f'POS{pr_i}'] = dk_pastResults['POS'].apply(lambda x: rewrite(x))
    
    #dk_pastResults.drop(['RESULT','GROUP RECORD','OFFICIALMONEY','FEDEXCUPPOINTS','POS'],axis=1,inplace=True)
    print(dk_pastResults.head())
    #dk_pastResults.drop(dk_pastResults[dk_pastResults[f'TOT{pr_i}'] < 250].index,inplace=True)
    new_df = pd.DataFrame(columns=['Name',f'POS{pr_i}'])
    names = []
    pos = []
    for index, row in dk_pastResults.iterrows():
        players = row['PLAYER']
        name1, name2 = duo_data_fn(players)
        names.append(name1)
        names.append(name2)
        pos.append(row[f'POS{pr_i}'])
        pos.append(row[f'POS{pr_i}'])
        #new_first_row = 

    new_df['Name'] = names
    new_df[f'POS{pr_i}'] = pos
    print(new_df.head())
    new_df['Name'] = new_df['Name'].apply(lambda x: series_lower(x))
    df_merge = pd.merge(df_merge, new_df,how='left',on='Name')
    df_merge[f'POS{pr_i}'] = df_merge[f'POS{pr_i}'].fillna(99)
    df_merge.sort_values(by=[f'POS{pr_i}'],inplace=True)
    pastResultRank = df_merge[f'POS{pr_i}'].rank(pct=True, ascending=False)
    df_merge[f'POS{pr_i}'] = pastResultRank * upperBound + lowerBound
    #df_merge[f'POS{pr_i}'] = df_merge[f'POS{pr_i}'].fillna(0)

    return df_merge


def past_results_dyn(df_merge: pd.DataFrame, url: str, lowerBound: int=0, upperBound:float=4, pr_i:int=0):
    '''A function that scrapes data from a dynamic website and ranks the players based on their past results'''
    driver = webdriver.Firefox(service_log_path=os.path.devnull, options=options)
    driver.get(url)
    driver.implicitly_wait(120)
    time.sleep(10)
    select = Select(driver.find_element_by_id('pastResultsYearSelector'))
    if pr_i == 0:
        select.select_by_value('2021.019')
    else:
        select.select_by_value('2020.019')
    time.sleep(5)
    result = driver.page_source
    dk_pastResults = pd.read_html(result)
    dk_pastResults = dk_pastResults[1]
    new_col = ['PLAYER', 'POS', 'R1', 'R2', 'R3', 'R4', 'TOTAL SCORE', 'TO PAR', 'OFFICIAL MONEY', 'FEDEXCUP POINTS']
    dk_pastResults.columns = new_col
    dk_pastResults[f'POS{pr_i}'] = dk_pastResults['POS'].apply(lambda x: pos_rewrite(x))
    dk_pastResults.drop(['POS', 'R1', 'R2', 'R3', 'R4', 'TOTAL SCORE', 'TO PAR', 'OFFICIAL MONEY', 'FEDEXCUP POINTS'],axis=1,inplace=True)
    driver.close()
    driver.quit()  
    dk_pastResults.rename(columns={'PLAYER':'Name'}, inplace=True)
    dk_pastResults['Name'] = dk_pastResults['Name'].apply(lambda x: series_lower(x))
    df_merge = pd.merge(df_merge, dk_pastResults,how='left',on='Name')
    df_merge[f'POS{pr_i}'] = df_merge[f'POS{pr_i}'].fillna(99)
    df_merge.sort_values(by=[f'POS{pr_i}'],inplace=True)
    pastResultRank = df_merge[f'POS{pr_i}'].rank(pct=True, ascending=False)
    df_merge[f'POS{pr_i}'] = pastResultRank * upperBound + lowerBound
    return df_merge

def past_results_match(df_merge, url, lowerBound=0, upperBound=4, pr_i=0):
    driver = webdriver.Firefox(service_log_path=os.path.devnull, options=options)
    driver.get(url)
    driver.implicitly_wait(120)
    time.sleep(10)
    result = driver.page_source
    # soup = BeautifulSoup(result, "html.parser")
    # dk_pastResults = pd.DataFrame(columns=['Name', 'Adjustment'])
    dk_pastResults = pd.read_html(result)
    dk_pastResults = dk_pastResults[1]
    print(dk_pastResults.head())
    driver.close()
    driver.quit()  
    dk_pastResults[f'POS{pr_i}'] = dk_pastResults['POS'].apply(lambda x: rewrite(x))
    
    dk_pastResults.drop(['RESULT','GROUP RECORD','OFFICIALMONEY','FEDEXCUPPOINTS','POS'],axis=1,inplace=True)
    print(dk_pastResults.head())
    #dk_pastResults.drop(dk_pastResults[dk_pastResults[f'TOT{pr_i}'] < 250].index,inplace=True)
    dk_pastResults.rename(columns={'PLAYER':'Name'}, inplace=True)
    dk_pastResults['Name'] = dk_pastResults['Name'].apply(lambda x: series_lower(x))
    df_merge = pd.merge(df_merge, dk_pastResults,how='left',on='Name')
    df_merge.sort_values(by=[f'POS{pr_i}'],inplace=True)
    pastResultRank = df_merge[f'POS{pr_i}'].rank(pct=True, ascending=False)
    df_merge[f'POS{pr_i}'] = pastResultRank * upperBound + lowerBound
    df_merge[f'POS{pr_i}'] = df_merge[f'POS{pr_i}'].fillna(0)

    return df_merge

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

def driving_distance(df_merge: pd.DataFrame, lowerBound:float=0, upperBound:float=5):
    '''Check for players with longest driving distance'''
    dk_distance = pd.read_html('https://www.pgatour.com/stats/stat.101.html')
    dk_distance = dk_distance[1]
    dk_distance.drop(['RANK LAST WEEK','ROUNDS','TOTAL DISTANCE','TOTAL DRIVES'], axis=1, inplace=True)
    dk_distance.drop(dk_distance.columns[0],axis=1,inplace=True)
    dk_distance.rename(columns={'PLAYER NAME':'Name','AVG.':'DriveDist'}, inplace=True)
    dk_distance['Name'] = dk_distance['Name'].apply(lambda x: series_lower(x))
    df_merge = pd.merge(df_merge, dk_distance, how='left',on='Name')

    driveDistScale = np.concatenate((np.linspace(upperBound,lowerBound,len(df_merge['DriveDist'].dropna())),np.zeros(len(df_merge)-len(df_merge['DriveDist'].dropna()))))
    df_merge.sort_values(by='DriveDist',ascending=False,inplace=True)
    df_merge['DriveDist'] = driveDistScale
    return df_merge

def driving_accuracy(df_merge: pd.DataFrame, lowerBound:float=0, upperBound:float=5):
    ''''Check for players with best driving accuracy (FIR)'''
    dk_accuracy = pd.read_html('https://www.pgatour.com/stats/stat.102.html')
    dk_accuracy = dk_accuracy[1]
    dk_accuracy.drop(['RANK LAST WEEK','ROUNDS','FAIRWAYS HIT','POSSIBLE FAIRWAYS'], axis=1, inplace=True)
    dk_accuracy.drop(dk_accuracy.columns[0],axis=1,inplace=True)
    dk_accuracy.rename(columns={'PLAYER NAME':'Name','%':'DriveAcc'}, inplace=True)
    dk_accuracy['Name'] = dk_accuracy['Name'].apply(lambda x: series_lower(x))
    df_merge = pd.merge(df_merge, dk_accuracy, how='left',on='Name')

    driveDistScale = np.concatenate((np.linspace(upperBound,lowerBound,len(df_merge['DriveAcc'].dropna())),np.zeros(len(df_merge)-len(df_merge['DriveAcc'].dropna()))))
    df_merge.sort_values(by='DriveAcc',ascending=False,inplace=True)
    df_merge['DriveAcc'] = driveDistScale
    return df_merge

def putting(df_merge: pd.DataFrame, lowerBound:float=0, upperBound:float=5):
    '''Check for players with most strokes gained putting'''
    dk_putting = pd.read_html('https://www.pgatour.com/stats/stat.02564.html')
    dk_putting = dk_putting[1]
    dk_putting.drop(['RANK LAST WEEK','ROUNDS','TOTAL SG:PUTTING','MEASURED ROUNDS'], axis=1, inplace=True)
    dk_putting.drop( dk_putting.columns[0],axis=1,inplace=True)
    dk_putting.rename(columns={'PLAYER NAME':'Name','AVERAGE':'PuttGain'}, inplace=True)
    dk_putting['Name'] = dk_putting['Name'].apply(lambda x: series_lower(x))
    df_merge = pd.merge(df_merge, dk_putting, how='left',on='Name')

    puttScale = np.concatenate((np.linspace(upperBound,lowerBound,len(df_merge['PuttGain'].dropna())),np.zeros(len(df_merge)-len(df_merge['PuttGain'].dropna()))))
    df_merge.sort_values(by='PuttGain',ascending=False,inplace=True)
    df_merge['PuttGain'] = puttScale
    return df_merge

def around_green(df_merge: pd.DataFrame, lowerBound: float=0, upperBound: float=5):
    '''Check for players with most storkes gained around green'''
    dk_around_green = pd.read_html('https://www.pgatour.com/stats/stat.02569.html')
    dk_around_green = dk_around_green[1]
    dk_around_green.drop(['RANK LAST WEEK','ROUNDS','TOTAL SG:ARG','MEASURED ROUNDS'], axis=1, inplace=True)
    dk_around_green.drop( dk_around_green.columns[0],axis=1,inplace=True)
    dk_around_green.rename(columns={'PLAYER NAME':'Name','AVERAGE':'ARGGain'}, inplace=True)
    dk_around_green['Name'] = dk_around_green['Name'].apply(lambda x: series_lower(x))
    df_merge = pd.merge(df_merge, dk_around_green, how='left',on='Name')

    puttScale = np.concatenate((np.linspace(upperBound, lowerBound,len(df_merge['ARGGain'].dropna())),np.zeros(len(df_merge)-len(df_merge['ARGGain'].dropna()))))
    df_merge.sort_values(by='ARGGain',ascending=False,inplace=True)
    df_merge['ARGGain'] = puttScale
    return df_merge

def weight_efficiencies(df: pd.DataFrame, course_df: pd.DataFrame):
    '''weight efficiencies based on the course'''
    eff125 = course_df[(course_df['Yards'] < 150) & (course_df['Yards'] > 125)]['Yards'].count()
    eff150 = course_df[(course_df['Yards'] < 175) & (course_df['Yards'] > 150)]['Yards'].count()
    eff175 = course_df[(course_df['Yards'] < 200) & (course_df['Yards'] > 175)]['Yards'].count()
    eff200 = course_df[(course_df['Yards'] < 225) & (course_df['Yards'] > 200)]['Yards'].count()
    eff225 = course_df[(course_df['Yards'] < 250) & (course_df['Yards'] > 225)]['Yards'].count()
    eff300 = course_df[(course_df['Yards'] < 350) & (course_df['Yards'] > 300)]['Yards'].count()
    eff350 = course_df[(course_df['Yards'] < 400) & (course_df['Yards'] > 350)]['Yards'].count()
    eff400 = course_df[(course_df['Yards'] < 450) & (course_df['Yards'] > 400)]['Yards'].count()
    eff450 = course_df[(course_df['Yards'] < 500) & (course_df['Yards'] > 450)]['Yards'].count()
    eff500_p4 = course_df[(course_df['Yards'] > 500) & (course_df['Par'] == 4)]['Yards'].count()
    eff500 = course_df[(course_df['Yards'] < 550) & (course_df['Yards'] > 500) & (course_df['Par'] == 5)]['Yards'].count()
    eff550 = course_df[(course_df['Yards'] < 600) & (course_df['Yards'] > 550) & (course_df['Par'] == 5)]['Yards'].count()
    eff600 = course_df[(course_df['Yards'] < 650) & (course_df['Yards'] > 600)]['Yards'].count()

    scales = [eff125, eff150,eff175,eff200,eff225,eff300,eff350,eff400,eff450, eff500_p4, eff500, eff550,eff600]
    keys = ['125-150 Eff','150-175 Eff','175-200 Eff','200-225 Eff','225-250 Eff','300-350 Eff','350-400 Eff','400-450 Eff','450-500 Eff','500+ Eff','500-550 Eff','550-600 Eff','600-650 Eff']
    
    #merge efficiencies
    for i in range(len(scales)):
        if i == 0:
            df_merge = getEff(df,keys[i],scales[i])
        else:
            df_merge = getEff(df_merge,keys[i],scales[i])
    
    print(keys,scales)
    return df_merge

def google_credentials():
    '''get google credentials for googleSheets API'''
    scope = ["https://spreadsheets.google.com/feeds",'https://www.googleapis.com/auth/spreadsheets',"https://www.googleapis.com/auth/drive.file","https://www.googleapis.com/auth/drive"]
    creds = ServiceAccountCredentials.from_json_keyfile_name('creds.json',scope)
    client = gspread.authorize(creds)
    return scope, creds, client

def assign_course_df(client):
    '''Find googleSheet and pull in all data'''
    courseSheet = client.open("DK PGA Course Analysis").sheet1
    course = courseSheet.get_all_values()
    headers = course.pop(0)
    course_df = pd.DataFrame(course,columns=headers)
    course_df['Yards'] = course_df['Yardage'].apply(lambda x: rewrite(x))
    course_df['Par'] = course_df['Par'].apply(lambda x: rewrite(x))
    return course_df

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
                if str(data['SCORE'].values[0]) == 'WD' or str(data['SCORE'].values[0]) == 'DQ':
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
                if str(data['SCORE'].values[0]) == 'WD' or str(data['SCORE'].values[0]) == 'DQ':
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
                    print(f'found {t_id}')
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
                print(f'{(cnt/len_merge)*100}% complete')
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
                    print(f'found {t_id}')
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
                print(f'{(cnt/len_merge)*100}% complete')
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
                print(f'{(cnt/len_merge)*100}% complete')
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

def par5_eaglePercentage(df_merge:pd.DataFrame, lowerBound:float=0, upperBound:float=5):
    '''Check for players with best eagle percentage on Par 5s'''
    dk_eagle = pd.read_html('https://www.pgatour.com/stats/stat.448.html')
    dk_eagle = dk_eagle[1]
    dk_eagle.drop(['RANK LAST WEEK','ROUNDS','TOTAL HOLE OUTS'], axis=1, inplace=True)
    dk_eagle.drop(dk_eagle.columns[0],axis=1,inplace=True)
    dk_eagle['%'] = dk_eagle.apply(lambda x: eaglePerc(x),axis=1)
    dk_eagle.drop(['TOTAL','TOTAL PAR 5 HOLES'],axis=1,inplace=True)
    dk_eagle.rename(columns={'PLAYER NAME':'Name','%':'Eagles'}, inplace=True)
    dk_eagle['Name'] = dk_eagle['Name'].apply(lambda x: series_lower(x))
    df_merge = pd.merge(df_merge, dk_eagle, how='left',on='Name')

    eagleScale = np.concatenate((np.linspace(upperBound, lowerBound,len(df_merge['Eagles'].dropna())),np.zeros(len(df_merge)-len(df_merge['Eagles'].dropna()))))
    df_merge.sort_values(by='Eagles',ascending=False,inplace=True)
    df_merge['Eagles'] = eagleScale
    return df_merge

def proximity_125to150(df_merge: pd.DataFrame, lowerBound:float=0, upperBound:float=5):
    '''Check for players with closest proximity when 125 to 150 yards out'''
    #proximity 125-150
    dk_proximity = pd.read_html('https://www.pgatour.com/stats/stat.339.html')
    dk_proximity = dk_proximity[1]
    dk_proximity.drop(['RANK LAST WEEK','ROUNDS','TOTAL DISTANCE (FEET)','# OF ATTEMPTS','RELATIVE TO PAR'], axis=1, inplace=True)
    dk_proximity.drop(dk_proximity.columns[0],axis=1,inplace=True)
    dk_proximity['Proxy'] = dk_proximity["AVG"].apply(lambda x: distance(x))
    dk_proximity.drop(['AVG'],axis=1,inplace=True)
    dk_proximity.rename(columns={'PLAYER NAME':'Name','Proxy':'Proximity'}, inplace=True)
    dk_proximity['Name'] = dk_proximity['Name'].apply(lambda x: series_lower(x))
    df_merge = pd.merge(df_merge, dk_proximity, how='left',on='Name')

    proxScale = np.concatenate((np.linspace(upperBound,lowerBound,len(df_merge['Proximity'].dropna())),np.zeros(len(df_merge)-len(df_merge['Proximity'].dropna()))))
    df_merge.sort_values(by='Proximity',ascending=True,inplace=True)
    df_merge['Proximity'] = proxScale
    return df_merge

def birdie_or_better(df_merge:pd.DataFrame, lowerBound:float=0, upperBound:float=5):
    '''Check for players birdie or better percentage'''
    #birdie or better%
    dk_birdies = pd.read_html('https://www.pgatour.com/stats/stat.352.html')
    dk_birdies = dk_birdies[1]
    dk_birdies.drop(['RANK LAST WEEK','ROUNDS','TOTAL BIRDIES','TOTAL HOLES','GIR RANK'], axis=1, inplace=True)
    dk_birdies.drop(dk_birdies.columns[0],axis=1,inplace=True)
    dk_birdies.rename(columns={'PLAYER NAME':'Name','%':'Birdies'}, inplace=True)
    dk_birdies['Name'] = dk_birdies['Name'].apply(lambda x: series_lower(x))
    df_merge = pd.merge(df_merge, dk_birdies, how='left',on='Name')

    birdScale = np.concatenate((np.linspace(upperBound,lowerBound,len(df_merge['Birdies'].dropna())),np.zeros(len(df_merge)-len(df_merge['Birdies'].dropna()))))
    df_merge.sort_values(by='Birdies',ascending=True,inplace=True)
    df_merge['Birdies'] = birdScale
    return df_merge
