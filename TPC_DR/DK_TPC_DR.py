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
    

#main

#draftkings csv
df = pd.read_csv('DKSalaries-TPC-DR.csv')
df.drop(['Position','Name + ID','ID','Roster Position','Game Info', 'TeamAbbrev'],axis=1,inplace=True)

#par3efficiency 200-225
dk_par3efficiency = pd.read_html('https://www.pgatour.com/stats/stat.02522.html')
dk_par3efficiency = dk_par3efficiency[1]
dk_par3efficiency.drop(['RANK LAST WEEK','ROUNDS','TOTAL STROKES','TOTAL ATTEMPTS'], axis=1, inplace=True)
dk_par3efficiency.drop(dk_par3efficiency.columns[0],axis=1,inplace=True)
dk_par3efficiency.rename(columns={'PLAYER NAME':'Name','AVG':'Par3Eff_200-225'}, inplace=True)

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

#strokes gained putting
dk_putting = pd.read_html('https://www.pgatour.com/stats/stat.02564.html')
dk_putting = dk_putting[1]
dk_putting.drop(['RANK LAST WEEK','ROUNDS','TOTAL SG:PUTTING','MEASURED ROUNDS'], axis=1, inplace=True)
dk_putting.drop(dk_putting.columns[0],axis=1,inplace=True)
dk_putting.rename(columns={'PLAYER NAME':'Name','AVERAGE':'Putting'}, inplace=True)

#birdie or better%
dk_birdies = pd.read_html('https://www.pgatour.com/stats/stat.352.html')
dk_birdies = dk_birdies[1]
dk_birdies.drop(['RANK LAST WEEK','ROUNDS','TOTAL BIRDIES','TOTAL HOLES','GIR RANK'], axis=1, inplace=True)
dk_birdies.drop(dk_birdies.columns[0],axis=1,inplace=True)
dk_birdies.rename(columns={'PLAYER NAME':'Name','%':'Birdies'}, inplace=True)

#past results
dk_pastResults = pd.read_html('https://www.espn.com/golf/leaderboard/_/tournamentId/401056548')
dk_pastResults = dk_pastResults[0]
dk_pastResults.drop(['POS','TO PAR','R1','R2','R3','R4','TOT','EARNINGS'],axis=1,inplace=True)
dk_pastResults.drop(dk_pastResults[dk_pastResults['FEDEX PTS'] == 0].index,inplace=True)
dk_pastResults.rename(columns={'PLAYER':'Name'}, inplace=True)

#merge
dk_merge = pd.merge(df,dk_par3efficiency, how='left', on='Name')
dk_merge = pd.merge(dk_merge, dk_par4efficiency ,how='left',on='Name')
dk_merge = pd.merge(dk_merge, dk_proximity, how='left',on='Name')
dk_merge = pd.merge(dk_merge, dk_putting, how='left',on='Name')
dk_merge = pd.merge(dk_merge, dk_birdies, how='left',on='Name')
dk_merge = pd.merge(dk_merge, dk_pastResults,how='left',on='Name')

maxIter = 0
i = 0
j = 0
topTier = pd.DataFrame(columns=['Player'])

base = np.zeros(len(dk_merge))


#scale for avg DK points 10 - 30 uniform
avgPointsScale = np.linspace(10,30,len(dk_merge['AvgPointsPerGame'].dropna()))

#scale for to par efficiency 0 - 10 uniform
parEfficiencyScale = np.concatenate((np.linspace(10,0,len(dk_merge['Par3Eff_200-225'].dropna())),np.zeros(len(dk_merge)-len(dk_merge['Par3Eff_200-225'].dropna()))))

#putting scale
puttingScale = np.concatenate((np.linspace(10,0,len(dk_merge['Putting'].dropna())),np.zeros(len(dk_merge)-len(dk_merge['Putting'].dropna()))))

#proximity scale
proximityScale = np.concatenate((np.linspace(10,0,len(dk_merge['Proximity'].dropna())),np.zeros(len(dk_merge)-len(dk_merge['Proximity'].dropna()))))

#birdies scale
birdiesScale = np.concatenate((np.linspace(10,0,len(dk_merge['Birdies'].dropna())),np.zeros(len(dk_merge)-len(dk_merge['Birdies'].dropna()))))

#pastresults scale
pastResultsScale = np.concatenate((np.linspace(10,0,len(dk_merge['FEDEX PTS'].dropna())),np.zeros(len(dk_merge)-len(dk_merge['FEDEX PTS'].dropna()))))


#fill NaN
dk_merge['FEDEX PTS'] = dk_merge['FEDEX PTS'].fillna(0)
dk_merge['Birdies'] = dk_merge['Birdies'].fillna(0)
dk_merge['Par3Eff_200-225'] = dk_merge['Par3Eff_200-225'].fillna(5)
dk_merge['Putting'] = dk_merge['Putting'].fillna(-2)
dk_merge['Proximity'] = dk_merge['Proximity'].fillna(100)
dk_merge['Par4Eff_400-450'] = dk_merge['Par4Eff_400-450'].fillna(7)




#add avg points scale
dk_merge.sort_values(by=['AvgPointsPerGame'],inplace=True)
dk_merge['APPG'] = avgPointsScale

#add par3efficiency scale
dk_merge.sort_values(by='Par3Eff_200-225',ascending=True,inplace=True)
dk_merge['P3E'] = parEfficiencyScale

#add par4efficiency scale
dk_merge.sort_values(by='Par4Eff_400-450',ascending=True,inplace=True)
dk_merge['P4E'] = parEfficiencyScale

#add proximity scale
dk_merge.sort_values(by='Proximity',ascending=True,inplace=True)
dk_merge['Prox'] = proximityScale

#add putting scale
dk_merge.sort_values(by='Putting',ascending=False,inplace=True)
dk_merge['Putt'] = puttingScale

#add par5efficiency scale
dk_merge.sort_values(by='Birdies',ascending=False,inplace=True)
dk_merge['Bird'] = birdiesScale
#print(dk_merge.head())

#add past results
dk_merge.sort_values(by='FEDEX PTS', ascending=False,inplace=True)
dk_merge['PR'] = pastResultsScale

#reshape
dk_merge.sort_values(by='Salary',ascending=False,inplace=True)
#dk_merge.to_csv('DKTest.csv', index = False)
#dk_merge.to_csv('DKRMC_IS.csv', index = False)

dk_merge.drop(['AvgPointsPerGame','Par3Eff_200-225','Par4Eff_400-450','Putting','Birdies','Proximity','FEDEX PTS'],axis=1,inplace=True)
column_list = ['APPG','P3E','P4E','Prox','Putt','Bird','PR']
dk_merge['Total'] = dk_merge[column_list].sum(axis=1)
dk_merge.drop(column_list,axis=1,inplace=True)
#dk_merge.dropna(inplace=True)
dk_merge.to_csv('DKTest.csv', index = False)


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

