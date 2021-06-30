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

#main

#draftkings csv
df = pd.read_csv('DKSalaries-RMC.csv')
df.drop(['Position','Name + ID','ID','Roster Position','Game Info', 'TeamAbbrev'],axis=1,inplace=True)

#par4efficiency 350-400
dk_par4efficiency = pd.read_html('https://www.pgatour.com/content/pgatour/stats/stat.02528.y2021.html')
dk_par4efficiency = dk_par4efficiency[1]
dk_par4efficiency.drop(['RANK LAST WEEK','ROUNDS','TOTAL STROKES','TOTAL ATTEMPTS'], axis=1, inplace=True)
dk_par4efficiency.drop(dk_par4efficiency.columns[0],axis=1,inplace=True)
dk_par4efficiency.rename(columns={'PLAYER NAME':'Name','AVG':'Par4Eff_350-400'}, inplace=True)

#par4efficiency 450-500
dk_par4efficiency1 = pd.read_html('https://www.pgatour.com/stats/stat.02530.html')
dk_par4efficiency1 = dk_par4efficiency1[1]
dk_par4efficiency1.drop(['RANK LAST WEEK','ROUNDS','TOTAL STROKES','TOTAL ATTEMPTS'], axis=1, inplace=True)
dk_par4efficiency1.drop(dk_par4efficiency1.columns[0],axis=1,inplace=True)
dk_par4efficiency1.rename(columns={'PLAYER NAME':'Name','AVG':'Par4Eff_450-500'}, inplace=True)

#par5efficiency 550-600
dk_par5efficiency = pd.read_html('https://www.pgatour.com/stats/stat.02534.html')
dk_par5efficiency = dk_par5efficiency[1]
dk_par5efficiency.drop(['RANK LAST WEEK','ROUNDS','TOTAL STROKES','TOTAL ATTEMPTS'], axis=1, inplace=True)
dk_par5efficiency.drop(dk_par5efficiency.columns[0],axis=1,inplace=True)
dk_par5efficiency.rename(columns={'PLAYER NAME':'Name','AVG':'Par5Eff_550-600'}, inplace=True)

#driving accuracy
dk_drivingAcc = pd.read_html('https://www.pgatour.com/stats/stat.102.html')
dk_drivingAcc = dk_drivingAcc[1]
dk_drivingAcc.drop(['RANK LAST WEEK','ROUNDS','FAIRWAYS HIT','POSSIBLE FAIRWAYS'], axis=1, inplace=True)
dk_drivingAcc.drop(dk_drivingAcc.columns[0],axis=1,inplace=True)
dk_drivingAcc.rename(columns={'PLAYER NAME':'Name','%':'DriveAcc'}, inplace=True)

#par3efficiency 150-175
dk_par3efficiency = pd.read_html('https://www.pgatour.com/stats/stat.02520.html')
dk_par3efficiency = dk_par3efficiency[1]
dk_par3efficiency.drop(['RANK LAST WEEK','ROUNDS','TOTAL STROKES','TOTAL ATTEMPTS'], axis=1, inplace=True)
dk_par3efficiency.drop(dk_par3efficiency.columns[0],axis=1,inplace=True)
dk_par3efficiency.rename(columns={'PLAYER NAME':'Name','AVG':'Par3Eff_150-175'}, inplace=True)

#past results
dk_pastResults = pd.read_html('https://www.espn.com/golf/leaderboard/_/tournamentId/401155460')
dk_pastResults = dk_pastResults[0]
dk_pastResults.drop(['POS','TO PAR','R1','R2','R3','R4','TOT','EARNINGS'],axis=1,inplace=True)
dk_pastResults.drop(dk_pastResults[dk_pastResults['FEDEX PTS'] == 0].index,inplace=True)
dk_pastResults.rename(columns={'PLAYER':'Name'}, inplace=True)

#merge
dk_merge = pd.merge(df,dk_par4efficiency, how='left', on='Name')
dk_merge = pd.merge(dk_merge, dk_par4efficiency1 ,how='left',on='Name')
dk_merge = pd.merge(dk_merge, dk_par5efficiency, how='left',on='Name')
dk_merge = pd.merge(dk_merge, dk_drivingAcc, how='left',on='Name')
dk_merge = pd.merge(dk_merge, dk_par3efficiency, how='left',on='Name')
dk_merge = pd.merge(dk_merge, dk_pastResults,how='left',on='Name')

maxIter = 0
i = 0
master = []

base = np.zeros(len(dk_merge))


#scale for avg DK points 10 - 30 uniform
avgPointsScale = np.linspace(10,30,len(dk_merge['AvgPointsPerGame'].dropna()))

#scale for to par efficiency 0 - 10 uniform
parEfficiencyScale = np.concatenate((np.linspace(10,0,len(dk_merge['Par4Eff_350-400'].dropna())),np.zeros(len(dk_merge)-len(dk_merge['Par4Eff_350-400'].dropna()))))

#drive accuracy scale
driveAccScale = np.concatenate((np.linspace(10,0,len(dk_merge['DriveAcc'].dropna())),np.zeros(len(dk_merge)-len(dk_merge['DriveAcc'].dropna()))))

#pastresults scale
pastResultsScale = np.concatenate((np.linspace(10,0,len(dk_merge['FEDEX PTS'].dropna())),np.zeros(len(dk_merge)-len(dk_merge['FEDEX PTS'].dropna()))))


#fill NaN
dk_merge['FEDEX PTS'] = dk_merge['FEDEX PTS'].fillna(0)
dk_merge['DriveAcc'] = dk_merge['DriveAcc'].fillna(0)
dk_merge['Par3Eff_150-175'] = dk_merge['Par3Eff_150-175'].fillna(5)
dk_merge['Par4Eff_350-400'] = dk_merge['Par4Eff_350-400'].fillna(6)
dk_merge['Par4Eff_450-500'] = dk_merge['Par4Eff_450-500'].fillna(6)
dk_merge['Par5Eff_550-600'] = dk_merge['Par5Eff_550-600'].fillna(7)



#add avg points scale
dk_merge.sort_values(by=['AvgPointsPerGame'],inplace=True)
dk_merge['APPG'] = avgPointsScale

#add par4efficiency scale
dk_merge.sort_values(by='Par4Eff_350-400',ascending=True,inplace=True)
dk_merge['P4E'] = parEfficiencyScale

#add par4efficiency1 scale
dk_merge.sort_values(by='Par4Eff_450-500',ascending=True,inplace=True)
dk_merge['P4E1'] = parEfficiencyScale

#add par5efficiency scale
dk_merge.sort_values(by='Par5Eff_550-600',ascending=True,inplace=True)
dk_merge['P5E'] = parEfficiencyScale

#add par3efficiency scale
dk_merge.sort_values(by='Par3Eff_150-175',ascending=True,inplace=True)
dk_merge['P3E'] = parEfficiencyScale

#add par5efficiency scale
dk_merge.sort_values(by='DriveAcc',ascending=False,inplace=True)
dk_merge['DA'] = driveAccScale
#print(dk_merge.head())

#add past results
dk_merge.sort_values(by='FEDEX PTS', ascending=False,inplace=True)
dk_merge['PR'] = pastResultsScale

#reshape
dk_merge.sort_values(by='Salary',ascending=False,inplace=True)
#dk_merge.to_csv('DKTest.csv', index = False)

dk_merge.drop(['AvgPointsPerGame','Par4Eff_350-400','Par4Eff_450-500','DriveAcc','Par5Eff_550-600','Par3Eff_150-175','FEDEX PTS'],axis=1,inplace=True)
column_list = ['APPG','P3E','P4E','P4E1','P5E','DA','PR']
dk_merge['Total'] = dk_merge[column_list].sum(axis=1)
dk_merge.drop(column_list,axis=1,inplace=True)
#dk_merge.dropna(inplace=True)
#dk_merge.to_csv('DKTest.csv', index = False)

print(dk_merge.head())

while i < iterations:
    lineup = genIter()
    lineup.sort()
    if lineup not in master:
        currentIter = objective(lineup)
        if currentIter > maxIter and constraint(lineup):
            maxIter = currentIter
            maxLineup = lineup
        if constraint(lineup):
            master.append(lineup)
            i = i + 1
        if i % 1000 == 0:
            print(i)

print(len(master))
print(maxIter)
print(getNames(maxLineup))

