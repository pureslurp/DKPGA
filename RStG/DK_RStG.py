#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd


#user input
iterations = 100000

#function definitions

def objective(x):
    p0 = []
    for iden in x:
        p0.append(float(dk_merge.loc[int(iden)]['Total']))
    return sum(p0)

def constraint(x):
    s0 = []
    for iden in x:
        s0.append(float(dk_merge.loc[int(iden)]['Salary']))
    if (50000 - sum(s0)) > 0:
        return True
    else:
        return False

def genIter():
    r0 = []
    while len(r0) < 6:
        r = np.random.randint(0,high=len(dk_merge))
        if r not in r0:
            r0.append(r)
    return r0

def getNames(lineup):
    n0 = []
    for iden in lineup:
        n0.append(dk_merge.loc[int(iden)]['Name'])
    return n0

def date(data):
    data = data.split("'")
    data[1] = data[1][:-1]
    x = int(data[0])+(int(data[1])/12)
    return x

def eaglePerc(data):
    return data[1]/data[2]
    

#main

#draftkings csv
df = pd.read_csv('DKSalaries-RStG.csv')
df.drop(['Position','Name + ID','ID','Roster Position','Game Info', 'TeamAbbrev'],axis=1,inplace=True)

#par3efficiency 150-175
dk_par3efficiency = pd.read_html('https://www.pgatour.com/stats/stat.02520.html')
dk_par3efficiency = dk_par3efficiency[1]
dk_par3efficiency.drop(['RANK LAST WEEK','ROUNDS','TOTAL STROKES','TOTAL ATTEMPTS'], axis=1, inplace=True)
dk_par3efficiency.drop(dk_par3efficiency.columns[0],axis=1,inplace=True)
dk_par3efficiency.rename(columns={'PLAYER NAME':'Name','AVG':'Par3Eff_150-175'}, inplace=True)

#par3efficiency 225-250
dk_par3efficiency1 = pd.read_html('https://www.pgatour.com/stats/stat.02523.html')
dk_par3efficiency1 = dk_par3efficiency1[1]
dk_par3efficiency1.drop(['RANK LAST WEEK','ROUNDS','TOTAL STROKES','TOTAL ATTEMPTS'], axis=1, inplace=True)
dk_par3efficiency1.drop(dk_par3efficiency1.columns[0],axis=1,inplace=True)
dk_par3efficiency1.rename(columns={'PLAYER NAME':'Name','AVG':'Par3Eff_225-250'}, inplace=True)

#par4efficiency 400-450
dk_par4efficiency = pd.read_html('https://www.pgatour.com/stats/stat.02529.html')
dk_par4efficiency = dk_par4efficiency[1]
dk_par4efficiency.drop(['RANK LAST WEEK','ROUNDS','TOTAL STROKES','TOTAL ATTEMPTS'], axis=1, inplace=True)
dk_par4efficiency.drop(dk_par4efficiency.columns[0],axis=1,inplace=True)
dk_par4efficiency.rename(columns={'PLAYER NAME':'Name','AVG':'Par4Eff_400-450'}, inplace=True)

#proximity 125-150
dk_proximity = pd.read_html('https://www.pgatour.com/stats/stat.339.html')
dk_proximity = dk_proximity[1]
dk_proximity.drop(['RANK LAST WEEK','ROUNDS','TOTAL DISTANCE (FEET)','# OF ATTEMPTS','RELATIVE TO PAR'], axis=1, inplace=True)
dk_proximity.drop(dk_proximity.columns[0],axis=1,inplace=True)
dk_proximity['Proxy'] = dk_proximity["AVG"].apply(lambda x: date(x))
dk_proximity.drop(['AVG'],axis=1,inplace=True)
dk_proximity.rename(columns={'PLAYER NAME':'Name','Proxy':'Proximity'}, inplace=True)

#par 5 eagle percentage
dk_eagle = pd.read_html('https://www.pgatour.com/stats/stat.448.html')
dk_eagle = dk_eagle[1]
dk_eagle.drop(['RANK LAST WEEK','ROUNDS','TOTAL HOLE OUTS'], axis=1, inplace=True)
dk_eagle.drop(dk_eagle.columns[0],axis=1,inplace=True)
dk_eagle['%'] = dk_eagle.apply(lambda x: eaglePerc(x),axis=1)
dk_eagle.drop(['TOTAL','TOTAL PAR 5 HOLES'],axis=1,inplace=True)
dk_eagle.rename(columns={'PLAYER NAME':'Name','%':'Eagles'}, inplace=True)

#past results
dk_pastResults = pd.read_html('https://www.espn.com/golf/leaderboard/_/tournamentId/982')
dk_pastResults = dk_pastResults[0]
dk_pastResults.drop(['POS','TO PAR','R1','R2','R3','R4','EARNINGS','FEDEX PTS'],axis=1,inplace=True)
dk_pastResults.drop(dk_pastResults[dk_pastResults['TOT'] < 170].index,inplace=True)
dk_pastResults.rename(columns={'PLAYER':'Name'}, inplace=True)

#merge
dk_merge = pd.merge(df,dk_par3efficiency, how='left', on='Name')
dk_merge = pd.merge(dk_merge,dk_par3efficiency1, how='left', on='Name')
dk_merge = pd.merge(dk_merge, dk_par4efficiency ,how='left',on='Name')
dk_merge = pd.merge(dk_merge, dk_proximity, how='left',on='Name')
dk_merge = pd.merge(dk_merge, dk_eagle, how='left',on='Name')
dk_merge = pd.merge(dk_merge, dk_pastResults,how='left',on='Name')

maxIter = 0
i = 0
j = 0
topTier = pd.DataFrame(columns=['Player'])

base = np.zeros(len(dk_merge))


#scale for avg DK points 10 - 30 uniform
avgPointsScale = np.linspace(10,30,len(dk_merge['AvgPointsPerGame'].dropna()))

#scale for to par efficiency 0 - 10 uniform
parEfficiencyScale = np.concatenate((np.linspace(5,0,len(dk_merge['Par3Eff_225-250'].dropna())),np.zeros(len(dk_merge)-len(dk_merge['Par3Eff_225-250'].dropna()))))

#proximity scale
proximityScale = np.concatenate((np.linspace(5,0,len(dk_merge['Proximity'].dropna())),np.zeros(len(dk_merge)-len(dk_merge['Proximity'].dropna()))))

#birdies scale
eagleScale = np.concatenate((np.linspace(5,0,len(dk_merge['Eagles'].dropna())),np.zeros(len(dk_merge)-len(dk_merge['Eagles'].dropna()))))

#pastresults scale
pastResultsScale = np.concatenate((np.linspace(5,0,len(dk_merge['TOT'].dropna())),np.zeros(len(dk_merge)-len(dk_merge['TOT'].dropna()))))


#fill NaN
dk_merge['TOT'] = dk_merge['TOT'].fillna(300)
dk_merge['Eagles'] = dk_merge['Eagles'].fillna(0)
dk_merge['Par3Eff_150-175'] = dk_merge['Par3Eff_150-175'].fillna(5)
dk_merge['Par3Eff_225-250'] = dk_merge['Par3Eff_225-250'].fillna(5)
dk_merge['Proximity'] = dk_merge['Proximity'].fillna(100)
dk_merge['Par4Eff_400-450'] = dk_merge['Par4Eff_400-450'].fillna(7)




#add avg points scale
dk_merge.sort_values(by=['AvgPointsPerGame'],inplace=True)
dk_merge['APPG'] = avgPointsScale

#add par3efficiency scale
dk_merge.sort_values(by='Par3Eff_150-175',ascending=True,inplace=True)
dk_merge['P3E'] = parEfficiencyScale

#add par3efficiency1 scale
dk_merge.sort_values(by='Par3Eff_225-250',ascending=True,inplace=True)
dk_merge['P3E1'] = parEfficiencyScale

#add par4efficiency scale
dk_merge.sort_values(by='Par4Eff_400-450',ascending=True,inplace=True)
dk_merge['P4E'] = parEfficiencyScale

#add proximity scale
dk_merge.sort_values(by='Proximity',ascending=True,inplace=True)
dk_merge['Prox'] = proximityScale


#add par5efficiency scale
dk_merge.sort_values(by='Eagles',ascending=False,inplace=True)
dk_merge['Eag'] = eagleScale
#print(dk_merge.head())

#add past results
dk_merge.sort_values(by='TOT', ascending=True,inplace=True)
dk_merge['PR'] = pastResultsScale

#reshape
dk_merge.sort_values(by='Salary',ascending=False,inplace=True)
dk_merge.to_csv('DKTest.csv', index = False)
#dk_merge.to_csv('DKRMC_IS.csv', index = False)

dk_merge.drop(['AvgPointsPerGame','Par3Eff_150-175','Par3Eff_225-250','Par4Eff_400-450','Eagles','Proximity','TOT'],axis=1,inplace=True)
column_list = ['APPG','P3E','P4E','Prox','P3E1','Eag','PR']
dk_merge['Total'] = dk_merge[column_list].sum(axis=1)
dk_merge.drop(column_list,axis=1,inplace=True)
#dk_merge.dropna(inplace=True)
#dk_merge.to_csv('DKTest.csv', index = False)


mean = dk_merge['Total'].mean()*6
sigma = dk_merge['Total'].std()*6

print(dk_merge.head())

while i < iterations:
    lineup = genIter()
    lineup.sort()
    currentIter = objective(lineup)
    if currentIter > maxIter and constraint(lineup):
        maxIter = currentIter
        maxLineup = lineup
    if currentIter > (mean+sigma):
        for x in lineup:
            topTier.loc[j] = dk_merge.loc[x]['Name']
            j = j + 1
    if constraint(lineup):
        i = i + 1
    if i % 1000 == 0:
        print(i)

#topTier = topTier.loc[topTier.duplicated(keep=False),:]
topTier = topTier[topTier.groupby('Player')['Player'].transform('size') > 10]
print(topTier['Player'].value_counts())
print(maxIter)
print(getNames(maxLineup))

