#!/usr/bin/env python
# coding: utf-8

##libraries
import numpy as np
import pandas as pd
from pgastats import getEff
import gspread
from oauth2client.service_account import ServiceAccountCredentials

##user input
iterations = 300000

##function definitions

#Calculate total points of a lineup
def objective(x):
    p0 = []
    for iden in x:
        p0.append(float(df_merge.loc[iden]['Total']))
    return sum(p0)

#Check if a lineup is within the Dk Salary range
def constraint(x):
    s0 = []
    for iden in x:
        s0.append(float(df_merge.loc[int(iden)]['Salary']))
    if (50000 - sum(s0)) > 0:
        return True
    else:
        return False

#Gneerate a random lineup
def genIter():
    r0 = []
    while len(r0) < 6:
        r = np.random.randint(0,high=len(df_merge))
        if r not in r0:
            r0.append(r)
    return r0

#Convert IDs into Names
def getNames(lineup):
    n0 = []
    for iden in lineup:
        n0.append(df_merge.loc[int(iden)]['Name + ID'])
    return n0

#Convert feet inches into float (e.g. 12'6" = 12.5)
def distance(data):
    data = data.split("'")
    data[1] = data[1][:-1]
    x = int(data[0])+(int(data[1])/12)
    return x

#Calculate eagle percentage (i.e. total eagles/total par 5s)
def eaglePerc(data):
    return data[1]/data[2]

def get_salary(lineup):
    s0 = []
    for iden in lineup:
        s0.append(float(df_merge.loc[iden]['Salary']))
    return sum(s0)

#convert string to integer in dataframe
def rewrite(data):
    try:
        output = int(data)
    except:
        output = 0
    return output

def getID(lineup):
    i0 = []
    for iden in lineup: 
        i0.append(df_merge[df_merge['Name + ID'] == iden].index)
    return i0

def optimize(df, salary, budget):
    upper_bound = int(salary+(min(budget,1000)))
    lower_bound = int(salary) - 1000
    df.sort_values(by=['Value'],ascending = False, inplace=True)
    window = df[(df['Salary'] <= upper_bound) & (df['Salary'] > lower_bound)]
    #print(window)
    return window.iloc[0]['Name + ID']

def maximize(df, salary, budget):     
    upper_bound = int(salary+(min(budget,1000)))
    lower_bound = int(salary) - 500
    df.sort_values(by=['Total'],ascending = False, inplace=True)
    window = df[(df['Salary'] <= upper_bound) & (df['Salary'] > lower_bound)]
    #print(window)
    return window.iloc[0]['Name + ID']

def duplicates(x):
    duplicates = False
    elemOfList = list(x)
    for elem in elemOfList:
        if elemOfList.count(elem) > 1:
            print('drop')
            duplicates = True
        
    return duplicates

def optimize_main(topTierLineup):
    for index, row in topTierLineup.iterrows():
        print(index)
        maxNames = [row[0],row[1],row[2],row[3],row[4],row[5]]
        # print(get_salary(getID(maxNames)))
        # print(maxNames)
        # print(maxIter)
        
        budget = 50000 - get_salary(getID(maxNames))
        
        for player in range(0,len(maxNames)):
            ID = getID(maxNames)
            if optimize(df_merge, df_merge.loc[ID[player]]['Salary'],budget) not in maxNames:
                maxNames[player] = optimize(df_merge, df_merge.loc[ID[player]]['Salary'],budget)
                budget = 50000 - get_salary(getID(maxNames))
        
        # print(objective(getID(maxNames)))
        # print(maxNames)
        # print(get_salary(getID(maxNames)))
        newTotal = objective(getID(maxNames))
        maxNames.append(newTotal)
        OptimizedLineup.loc[index] = maxNames
        
    for index, row in OptimizedLineup.iterrows():
        if duplicates(row) == True:
            OptimizedLineup.drop(index, inplace = True)

    OptimizedLineup.reset_index(drop=True, inplace=True)
    
    return OptimizedLineup

def maximize_main(OptimizedLinup):
    for index, row in OptimizedLinup.iterrows():
        print(index)
        maxNames = [row[0],row[1],row[2],row[3],row[4],row[5]]
        # print(get_salary(getID(maxNames)))
        # print(maxNames)
        # print(maxIter)
        
        budget = 50000 - get_salary(getID(maxNames))
        
        for player in range(0,len(maxNames)):
            ID = getID(maxNames)
            if maximize(df_merge, df_merge.loc[ID[player]]['Salary'],budget) not in maxNames:
                maxNames[player] = maximize(df_merge, df_merge.loc[ID[player]]['Salary'],budget)
                budget = 50000 - get_salary(getID(maxNames))
        
        # print(objective(getID(maxNames)))
        # print(maxNames)
        # print(get_salary(getID(maxNames)))
        newTotal = objective(getID(maxNames))
        maxNames.append(newTotal)
        MaximizedLineup.loc[index] = maxNames

    MaximizedLineup.reset_index(drop=True, inplace=True)
    
    return MaximizedLineup


def past_results(df_merge):
    dk_pastResults = pd.read_html('https://www.espn.com/golf/leaderboard/_/tournamentId/401148239')
    dk_pastResults = dk_pastResults[0]
    dk_pastResults.drop(['POS','SCORE','R1','R2','R3','R4','EARNINGS','FEDEX PTS'],axis=1,inplace=True)
    dk_pastResults['TOT'] = dk_pastResults['TOT'].apply(lambda x: rewrite(x))
    dk_pastResults.drop(dk_pastResults[dk_pastResults['TOT'] < 170].index,inplace=True)
    dk_pastResults.rename(columns={'PLAYER':'Name'}, inplace=True)
    df_merge = pd.merge(df_merge, dk_pastResults,how='left',on='Name')

    pastResultsScale = np.concatenate((np.linspace(5,0,len(df_merge['TOT'].dropna())),np.zeros(len(df_merge)-len(df_merge['TOT'].dropna()))))
    df_merge.sort_values(by=['TOT'],inplace=True)
    df_merge['TOT'] = pastResultsScale

    
    

    return df_merge

def driving_distance(df_merge):
    dk_distance = pd.read_html('https://www.pgatour.com/stats/stat.101.html')
    dk_distance = dk_distance[1]
    dk_distance.drop(['RANK LAST WEEK','ROUNDS','TOTAL DISTANCE','TOTAL DRIVES'], axis=1, inplace=True)
    dk_distance.drop(dk_distance.columns[0],axis=1,inplace=True)
    dk_distance.rename(columns={'PLAYER NAME':'Name','AVG.':'DriveDist'}, inplace=True)
    df_merge = pd.merge(df_merge, dk_distance, how='left',on='Name')

    driveDistScale = np.concatenate((np.linspace(5,0,len(df_merge['DriveDist'].dropna())),np.zeros(len(df_merge)-len(df_merge['DriveDist'].dropna()))))
    df_merge.sort_values(by='DriveDist',ascending=False,inplace=True)
    df_merge['DriveDist'] = driveDistScale

    

    return df_merge

def weight_efficiencies(df, course_df):
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
    scope = ["https://spreadsheets.google.com/feeds",'https://www.googleapis.com/auth/spreadsheets',"https://www.googleapis.com/auth/drive.file","https://www.googleapis.com/auth/drive"]
    creds = ServiceAccountCredentials.from_json_keyfile_name('creds.json',scope)
    client = gspread.authorize(creds)
    return scope, creds, client

def assign_course_df(client):
    courseSheet = client.open("DK PGA Course Analysis").sheet1
    course = courseSheet.get_all_values()
    headers = course.pop(0)
    course_df = pd.DataFrame(course,columns=headers)
    course_df['Yards'] = course_df['Yardage'].apply(lambda x: rewrite(x))
    course_df['Par'] = course_df['Par'].apply(lambda x: rewrite(x))
    return course_df

def DK_csv_assignemnt(path, name):
    df = pd.read_csv('{}{}'.format(path,name))
    df.drop(['Position','ID','Roster Position','Game Info', 'TeamAbbrev'],axis=1,inplace=True)
    avgPointsScale = np.linspace(10, 30,len(df['AvgPointsPerGame'].dropna()))
    df.sort_values(by=['AvgPointsPerGame'],inplace=True)
    df['AvgPointsPerGame'] = avgPointsScale
    return df

def df_total_and_reformat(df_merge):
    column_list = list(df_merge.columns[3:])
    df_merge['Total'] = df_merge[column_list].sum(axis=1)
    df_merge['Value'] = (df_merge['Total'] / df_merge['Salary']) * 1000
    df_merge = df_merge[['Name + ID', 'Salary', 'Total', 'Value']]
    df_merge.sort_values(by='Salary',ascending=False,inplace=True)
    return df_merge


path = 'DKPGA/ZOZO/'
name = 'DKSalaries-ZOZO.csv'

df = DK_csv_assignemnt(path, name)

scope, creds, client = google_credentials()

course_df = assign_course_df(client)

df_merge = weight_efficiencies(df, course_df)

df_merge = past_results(df_merge)
df_merge = driving_distance(df_merge)

df_merge.to_csv('{}DKData.csv'.format(path), index = False) #optional line if you want to see data in CSV

df_merge = df_total_and_reformat(df_merge)

df_merge.to_csv('{}DKFinal.csv'.format(path), index = False) #optional line if you want to see data in CSV

print(df_merge.head())

#Calculate Mean and Standard Deviation
mean = df_merge['Total'].mean()*6
sigma = df_merge['Total'].std()*6

#define variables
maxIter = 0                                    #placeholder for best lineup
i = 0                                          #main iterable
j = 0                                          #top tier iterable
k = 0                                          #top tier lineupdf iterable
topTier = pd.DataFrame(columns=['Player'])     #a dataframe of the top lineups (mean+sigma)
topTierLineup = pd.DataFrame(columns=['Player1','Player2','Player3','Player4','Player5','Player6','TOT'])
OptimizedLineup = pd.DataFrame(columns=['Player1','Player2','Player3','Player4','Player5','Player6','TOT'])
MaximizedLineup = pd.DataFrame(columns=['Player1','Player2','Player3','Player4','Player5','Player6','TOT'])


#main loop

while i < iterations:
    #get a sample
    lineup = genIter()
    lineup.sort()
    #assign sample
    currentIter = objective(lineup)
    #check if sample is better than current best sample
    if currentIter > maxIter and constraint(lineup):
        #reassign
        maxIter = currentIter
        maxLineup = lineup
    #check if sample is a top tier sample
    if currentIter > (mean+(sigma)) and constraint(lineup):
        #add players to top tier dataframe
        topTierData = getNames(lineup)
        topTierData.append(currentIter)
        topTierLineup.loc[k] = topTierData
        k = k + 1
        for x in lineup:
            topTier.loc[j] = df_merge.loc[x]['Name + ID']
            j = j + 1
    #iterate only if it is a valid lineup
    if constraint(lineup):
        i = i + 1
    #counter
    if i % 1000 == 0:
        print(i)

#print data for easy view

OptimizedLineup = optimize_main(topTierLineup)
MaximizedLineup = maximize_main(OptimizedLineup)

MaximizedLineup.to_csv('{}Maximized_Lineups.csv'.format(path),index=False)

print(MaximizedLineup.head())

