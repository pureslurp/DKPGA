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

#merge
dk_merge = pd.merge(df,dk_par4efficiency, how='inner', on='Name')
dk_merge = pd.merge(dk_merge, dk_par4efficiency1 ,how='inner',on='Name')
dk_merge = pd.merge(dk_merge, dk_par5efficiency, how='inner',on='Name')
dk_merge = pd.merge(dk_merge, dk_drivingAcc, how='inner',on='Name')
dk_merge = pd.merge(dk_merge, dk_par3efficiency, how='inner',on='Name')


maxIter = 0
i = 0

#scale for avg DK points 10 - 30 uniform
avgPointsScale = np.linspace(10,30,len(dk_merge))
#scale for to par efficiency 0 - 10 uniform
parEfficiencyScale = np.linspace(0,10,len(dk_merge))
#drive accuracy scale
driveAccScale = np.linspace(0,5,len(dk_merge))

#add avg points scale
dk_merge.sort_values(by=['AvgPointsPerGame'],inplace=True)
dk_merge['APPG'] = avgPointsScale

#add par4efficiency scale
dk_merge.sort_values(by='Par4Eff_350-400',ascending=False,inplace=True)
dk_merge['P4E'] = parEfficiencyScale

#add par4efficiency1 scale
dk_merge.sort_values(by='Par4Eff_450-500',ascending=False,inplace=True)
dk_merge['P4E1'] = parEfficiencyScale

#add par5efficiency scale
dk_merge.sort_values(by='Par5Eff_550-600',ascending=False,inplace=True)
dk_merge['P5E'] = parEfficiencyScale

#add par3efficiency scale
dk_merge.sort_values(by='Par3Eff_150-175',ascending=False,inplace=True)
dk_merge['P3E'] = parEfficiencyScale

#add par5efficiency scale
dk_merge.sort_values(by='DriveAcc',inplace=True)
dk_merge['DA'] = driveAccScale

#reshape
dk_merge.sort_values(by='Salary',ascending=False,inplace=True)
dk_merge.drop(['AvgPointsPerGame','Par4Eff_350-400','Par4Eff_450-500','DriveAcc','Par5Eff_550-600','Par3Eff_150-175'],axis=1,inplace=True)
column_list = ['APPG','P3E','P4E','P4E1','P5E','DA']
dk_merge['Total'] = dk_merge[column_list].sum(axis=1)
dk_merge.drop(column_list,axis=1,inplace=True)
dk_merge.dropna(inplace=True)



while i < iterations:
    lineup = genIter()
    currentIter = objective(lineup)
    if currentIter > maxIter and constraint(lineup):
        maxIter = currentIter
        maxLineup = lineup
    i = i + 1    
    if i % 100 == 0:
        print(i)

print(maxIter)
print(getNames(maxLineup))

