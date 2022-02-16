#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 18 20:17:02 2021

@author: seanraymor
"""
import pandas as pd
import numpy as np
import gspread
from oauth2client.service_account import ServiceAccountCredentials

OptimizedLineup = pd.DataFrame(columns=['Player1','Player2','Player3','Player4','Player5','Player6','TOT'])
MaximizedLineup = pd.DataFrame(columns=['Player1','Player2','Player3','Player4','Player5','Player6','TOT'])

def getEff(df, key, count):
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
    
    dk_merge = pd.merge(df,eff_df, how='left', on='Name')

    effScale = np.concatenate((np.linspace(count*2,0,len(dk_merge[key].dropna())),np.zeros(len(dk_merge)-len(dk_merge[key].dropna()))))
    dk_merge.sort_values(by=key,ascending=True,inplace=True)
    dk_merge[key] = effScale

    return dk_merge


def objective(x, df_merge):
    '''Calculate the total points of a lineup (find objective)'''
    p0 = []
    for iden in x:
        p0.append(float(df_merge.loc[iden]['Total']))
    return sum(p0)

def constraint(x, df_merge):
    '''Check if a lineup is within the DK salary range 50_000'''
    s0 = []
    for iden in x:
        s0.append(float(df_merge.loc[int(iden)]['Salary']))
    if (50000 - sum(s0)) > 0:
        return True
    else:
        return False

def genIter(df_merge):
    '''Generate a random lineup'''
    r0 = []
    while len(r0) < 6:
        r = np.random.randint(0,high=len(df_merge))
        if r not in r0:
            r0.append(r)
    return r0

def getNames(lineup, df_merge):
    '''Convert IDs into Names'''
    n0 = []
    for iden in lineup:
        n0.append(df_merge.loc[int(iden)]['Name + ID'])
    return n0

def distance(data):
    '''Convert feet inches into float (e.g. 12'6" = 12.5)'''
    data = data.split("'")
    data[1] = data[1][:-1]
    x = int(data[0])+(int(data[1])/12)
    return x

def eaglePerc(data):
    '''Calculate eagle percentage (i.e. total eagles/total par 5s)'''
    return data[1]/data[2]

def get_salary(lineup, df_merge):
    '''get the total salary for a particular lineup'''
    s0 = []
    for iden in lineup:
        s0.append(float(df_merge.loc[iden]['Salary']))
    return sum(s0)

def rewrite(data):
    '''convert string to integer in dataframe'''
    try:
        output = int(data)
    except:
        output = 0
    return output

def getID(lineup, df_merge):
    '''convert lineup names to ID'''
    i0 = []
    for iden in lineup: 
        i0.append(df_merge[df_merge['Name + ID'] == iden].index)
    return i0

def optimize(df, salary, budget):
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

def maximize(df, salary, budget):     
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
    #print(window)
    return window.iloc[0]['Name + ID']

def duplicates(x):
    '''Check for duplicates in the lineup'''
    duplicates = False
    elemOfList = list(x)
    for elem in elemOfList:
        if elemOfList.count(elem) > 1:
            print('drop')
            duplicates = True
        
    return duplicates

def optimize_main(topTierLineup, df_merge):
    '''Iterate over lineups to try to optimize all players (maximize value)

        Args: 
            topTierLineup (list): lineup from a row of dataframe as a list

        Return:
            Optimizedlineup (list): lineup with maximized value    
    '''
    for index, row in topTierLineup.iterrows():
        print(index)
        maxNames = [row[0],row[1],row[2],row[3],row[4],row[5]]
        budget = 50000 - get_salary(getID(maxNames, df_merge), df_merge)
        
        for player in range(0,len(maxNames)):
            ID = getID(maxNames, df_merge)
            if optimize(df_merge, df_merge.loc[ID[player]]['Salary'],budget) not in maxNames:
                maxNames[player] = optimize(df_merge, df_merge.loc[ID[player]]['Salary'],budget)
                budget = 50000 - get_salary(getID(maxNames, df_merge), df_merge)

        newTotal = objective(getID(maxNames, df_merge), df_merge)
        maxNames.append(newTotal)
        OptimizedLineup.loc[index] = maxNames
        
    for index, row in OptimizedLineup.iterrows():
        if duplicates(row) == True:
            OptimizedLineup.drop(index, inplace = True)

    OptimizedLineup.reset_index(drop=True, inplace=True)
    
    return OptimizedLineup

def maximize_main(OptimizedLinup, df_merge):
    '''Iterate over lineups to try to maximize all players (maximize points)

        Args: 
            OptimizedLineup (list): lineup from a row of dataframe as a list

        Return:
            Maximizedlineup (list): lineup with maximized points    
    '''
    for index, row in OptimizedLinup.iterrows():
        print(index)
        maxNames = [row[0],row[1],row[2],row[3],row[4],row[5]]
        
        budget = 50000 - get_salary(getID(maxNames, df_merge), df_merge)
        
        for player in range(0,len(maxNames)):
            ID = getID(maxNames, df_merge)
            if maximize(df_merge, df_merge.loc[ID[player]]['Salary'],budget) not in maxNames:
                maxNames[player] = maximize(df_merge, df_merge.loc[ID[player]]['Salary'],budget)
                budget = 50000 - get_salary(getID(maxNames, df_merge), df_merge)
        
        newTotal = objective(getID(maxNames, df_merge), df_merge)
        maxNames.append(newTotal)
        MaximizedLineup.loc[index] = maxNames

    
    MaximizedLineup.reset_index(drop=True, inplace=True)
    MaximizedLineup.sort_values(by='TOT', ascending=False, inplace=True)
    MaximizedLineup.drop_duplicates(subset=["TOT"],inplace=True)

    
    
    return MaximizedLineup

def past_results(df_merge, url, lowerBound=0, upperBound=5, playoff=False, pr_i=0):
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
    if playoff:
        dk_pastResults = dk_pastResults[1]
    else:
        dk_pastResults = dk_pastResults[0]
    
    
    dk_pastResults[f'TOT{pr_i}'] = dk_pastResults['TOT'].apply(lambda x: rewrite(x))
    dk_pastResults.drop(['POS','SCORE','R1','R2','R3','R4','EARNINGS','FEDEX PTS','TOT'],axis=1,inplace=True)
    dk_pastResults.drop(dk_pastResults[dk_pastResults[f'TOT{pr_i}'] < 170].index,inplace=True)
    dk_pastResults.rename(columns={'PLAYER':'Name'}, inplace=True)
    df_merge = pd.merge(df_merge, dk_pastResults,how='left',on='Name')


    pastResultsScale = np.concatenate((np.linspace(upperBound,lowerBound,len(df_merge[f'TOT{pr_i}'].dropna())),np.zeros(len(df_merge)-len(df_merge[f'TOT{pr_i}'].dropna()))))
    df_merge.sort_values(by=[f'TOT{pr_i}'],inplace=True)
    df_merge[f'TOT{pr_i}'] = pastResultsScale
    

    return df_merge

def driving_distance(df_merge, lowerBound=0, upperBound=5):
    '''Check for players with longest driving distance'''
    dk_distance = pd.read_html('https://www.pgatour.com/stats/stat.101.html')
    dk_distance = dk_distance[1]
    dk_distance.drop(['RANK LAST WEEK','ROUNDS','TOTAL DISTANCE','TOTAL DRIVES'], axis=1, inplace=True)
    dk_distance.drop(dk_distance.columns[0],axis=1,inplace=True)
    dk_distance.rename(columns={'PLAYER NAME':'Name','AVG.':'DriveDist'}, inplace=True)
    df_merge = pd.merge(df_merge, dk_distance, how='left',on='Name')

    driveDistScale = np.concatenate((np.linspace(upperBound,lowerBound,len(df_merge['DriveDist'].dropna())),np.zeros(len(df_merge)-len(df_merge['DriveDist'].dropna()))))
    df_merge.sort_values(by='DriveDist',ascending=False,inplace=True)
    df_merge['DriveDist'] = driveDistScale
    return df_merge

def driving_accuracy(df_merge, lowerBound=0, upperBound=5):
    ''''Check for players with best driving accuracy (FIR)'''
    dk_accuracy = pd.read_html('https://www.pgatour.com/stats/stat.102.html')
    dk_accuracy = dk_accuracy[1]
    dk_accuracy.drop(['RANK LAST WEEK','ROUNDS','FAIRWAYS HIT','POSSIBLE FAIRWAYS'], axis=1, inplace=True)
    dk_accuracy.drop(dk_accuracy.columns[0],axis=1,inplace=True)
    dk_accuracy.rename(columns={'PLAYER NAME':'Name','%':'DriveAcc'}, inplace=True)
    df_merge = pd.merge(df_merge, dk_accuracy, how='left',on='Name')

    driveDistScale = np.concatenate((np.linspace(upperBound,lowerBound,len(df_merge['DriveAcc'].dropna())),np.zeros(len(df_merge)-len(df_merge['DriveAcc'].dropna()))))
    df_merge.sort_values(by='DriveAcc',ascending=False,inplace=True)
    df_merge['DriveAcc'] = driveDistScale
    return df_merge

def putting(df_merge, lowerBound=0, upperBound=5):
    '''Check for players with most strokes gained putting'''
    dk_putting = pd.read_html('https://www.pgatour.com/stats/stat.02564.html')
    dk_putting = dk_putting[1]
    dk_putting.drop(['RANK LAST WEEK','ROUNDS','TOTAL SG:PUTTING','MEASURED ROUNDS'], axis=1, inplace=True)
    dk_putting.drop( dk_putting.columns[0],axis=1,inplace=True)
    dk_putting.rename(columns={'PLAYER NAME':'Name','AVERAGE':'PuttGain'}, inplace=True)
    df_merge = pd.merge(df_merge, dk_putting, how='left',on='Name')

    puttScale = np.concatenate((np.linspace(upperBound,lowerBound,len(df_merge['PuttGain'].dropna())),np.zeros(len(df_merge)-len(df_merge['PuttGain'].dropna()))))
    df_merge.sort_values(by='PuttGain',ascending=False,inplace=True)
    df_merge['PuttGain'] = puttScale
    return df_merge

def around_green(df_merge, lowerBound=0, upperBound=5):
    '''Check for players with most storkes gained around green'''
    dk_around_green = pd.read_html('https://www.pgatour.com/stats/stat.02569.html')
    dk_around_green = dk_around_green[1]
    dk_around_green.drop(['RANK LAST WEEK','ROUNDS','TOTAL SG:ARG','MEASURED ROUNDS'], axis=1, inplace=True)
    dk_around_green.drop( dk_around_green.columns[0],axis=1,inplace=True)
    dk_around_green.rename(columns={'PLAYER NAME':'Name','AVERAGE':'ARGGain'}, inplace=True)
    df_merge = pd.merge(df_merge, dk_around_green, how='left',on='Name')

    puttScale = np.concatenate((np.linspace(upperBound, lowerBound,len(df_merge['ARGGain'].dropna())),np.zeros(len(df_merge)-len(df_merge['ARGGain'].dropna()))))
    df_merge.sort_values(by='ARGGain',ascending=False,inplace=True)
    df_merge['ARGGain'] = puttScale
    return df_merge

def weight_efficiencies(df, course_df):
    '''weight efficiencies based on the course'''
    ## weight the efficiencies
    eff125 = course_df[(course_df['Yards'] < 150) & (course_df['Yards'] > 125)]['Yards'].count()
    eff150 = course_df[(course_df['Yards'] < 175) & (course_df['Yards'] > 150)]['Yards'].count()
    eff175 = course_df[(course_df['Yards'] < 200) & (course_df['Yards'] > 175)]['Yards'].count()
    eff200 = course_df[(course_df['Yards'] < 225) & (course_df['Yards'] > 200)]['Yards'].count()
    eff225 = course_df[(course_df['Yards'] < 250) & (course_df['Yards'] > 225)]['Yards'].count()
    eff300 = course_df[(course_df['Yards'] < 350) & (course_df['Yards'] > 300)]['Yards'].count()
    eff350 = course_df[(course_df['Yards'] < 400) & (course_df['Yards'] > 350)]['Yards'].count()
    eff400 = course_df[(course_df['Yards'] < 450) & (course_df['Yards'] > 400)]['Yards'].count()
    eff450 = course_df[(course_df['Yards'] < 500) & (course_df['Yards'] > 450)]['Yards'].count()
    eff500 = course_df[(course_df['Yards'] < 550) & (course_df['Yards'] > 500) & (course_df['Par'] == 5)]['Yards'].count()
    eff500_p4 = course_df[(course_df['Yards'] > 500) & (course_df['Par'] == 4)]['Yards'].count()
    eff550 = course_df[(course_df['Yards'] < 600) & (course_df['Yards'] > 550) & (course_df['Par'] == 5)]['Yards'].count()
    eff600 = course_df[(course_df['Yards'] < 650) & (course_df['Yards'] > 600)]['Yards'].count()

    scales = [eff125, eff150,eff175,eff200,eff225,eff300,eff350,eff400,eff450,eff500, eff500_p4, eff550,eff600]
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

def DK_csv_assignemnt(path, name, lowerBound=10, upperBound=30):
    '''find exported csv from DK and assign it to dataFrame with upper and lower bound values'''
    df = pd.read_csv('{}CSVs/{}'.format(path,name))
    df.drop(['Position','ID','Roster Position','Game Info', 'TeamAbbrev'],axis=1,inplace=True)
    avgPointsScale = np.linspace(10, 30,len(df['AvgPointsPerGame'].dropna()))
    df.sort_values(by=['AvgPointsPerGame'],inplace=True)
    df['AvgPointsPerGame'] = avgPointsScale
    return df

def df_total_and_reformat(df_merge):
    '''reformate dataFrame to be easily exported'''
    column_list = list(df_merge.columns[3:])
    df_merge['Total'] = df_merge[column_list].sum(axis=1)
    df_merge['Value'] = (df_merge['Total'] / df_merge['Salary']) * 1000
    df_merge = df_merge[['Name + ID', 'Salary', 'Total', 'Value']]
    df_merge.sort_values(by='Salary',ascending=False,inplace=True)
    return df_merge

def par5_eaglePercentage(df_merge, lowerBound=0, upperBound=5):
    '''Check for players with best eagle percentage on Par 5s'''
    dk_eagle = pd.read_html('https://www.pgatour.com/stats/stat.448.html')
    dk_eagle = dk_eagle[1]
    dk_eagle.drop(['RANK LAST WEEK','ROUNDS','TOTAL HOLE OUTS'], axis=1, inplace=True)
    dk_eagle.drop(dk_eagle.columns[0],axis=1,inplace=True)
    dk_eagle['%'] = dk_eagle.apply(lambda x: eaglePerc(x),axis=1)
    dk_eagle.drop(['TOTAL','TOTAL PAR 5 HOLES'],axis=1,inplace=True)
    dk_eagle.rename(columns={'PLAYER NAME':'Name','%':'Eagles'}, inplace=True)
    df_merge = pd.merge(df_merge, dk_eagle, how='left',on='Name')

    eagleScale = np.concatenate((np.linspace(upperBound, lowerBound,len(df_merge['Eagles'].dropna())),np.zeros(len(df_merge)-len(df_merge['Eagles'].dropna()))))
    df_merge.sort_values(by='Eagles',ascending=False,inplace=True)
    df_merge['Eagles'] = eagleScale
    return df_merge

def proximity_125to150(df_merge, lowerBound=0, upperBound=5):
    '''Check for players with closest proximity when 125 to 150 yards out'''
    #proximity 125-150
    dk_proximity = pd.read_html('https://www.pgatour.com/stats/stat.339.html')
    dk_proximity = dk_proximity[1]
    dk_proximity.drop(['RANK LAST WEEK','ROUNDS','TOTAL DISTANCE (FEET)','# OF ATTEMPTS','RELATIVE TO PAR'], axis=1, inplace=True)
    dk_proximity.drop(dk_proximity.columns[0],axis=1,inplace=True)
    dk_proximity['Proxy'] = dk_proximity["AVG"].apply(lambda x: distance(x))
    dk_proximity.drop(['AVG'],axis=1,inplace=True)
    dk_proximity.rename(columns={'PLAYER NAME':'Name','Proxy':'Proximity'}, inplace=True)
    df_merge = pd.merge(df_merge, dk_proximity, how='left',on='Name')

    proxScale = np.concatenate((np.linspace(upperBound,lowerBound,len(df_merge['Proximity'].dropna())),np.zeros(len(df_merge)-len(df_merge['Proximity'].dropna()))))
    df_merge.sort_values(by='Proximity',ascending=True,inplace=True)
    df_merge['Proximity'] = proxScale
    return df_merge

def birdie_or_better(df_merge, lowerBound=0, upperBound=5):
    '''Check for players birdie or better percentage'''
    #birdie or better%
    dk_birdies = pd.read_html('https://www.pgatour.com/stats/stat.352.html')
    dk_birdies = dk_birdies[1]
    dk_birdies.drop(['RANK LAST WEEK','ROUNDS','TOTAL BIRDIES','TOTAL HOLES','GIR RANK'], axis=1, inplace=True)
    dk_birdies.drop(dk_birdies.columns[0],axis=1,inplace=True)
    dk_birdies.rename(columns={'PLAYER NAME':'Name','%':'Birdies'}, inplace=True)
    df_merge = pd.merge(df_merge, dk_birdies, how='left',on='Name')

    birdScale = np.concatenate((np.linspace(upperBound,lowerBound,len(df_merge['Birdies'].dropna())),np.zeros(len(df_merge)-len(df_merge['Birdies'].dropna()))))
    df_merge.sort_values(by='Birdies',ascending=True,inplace=True)
    df_merge['Birdies'] = birdScale
    return df_merge
