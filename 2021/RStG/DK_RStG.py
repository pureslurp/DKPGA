#!/usr/bin/env python
# coding: utf-8

##libraries
import numpy as np
import pandas as pd


##user input
iterations = 300000

##function definitions

#Calculate total points of a lineup
def objective(x):
    p0 = []
    for iden in x:
        p0.append(float(dk_merge.loc[int(iden)]['Total']))
    return sum(p0)

#Check if a lineup is within the Dk Salary range
def constraint(x):
    s0 = []
    for iden in x:
        s0.append(float(dk_merge.loc[int(iden)]['Salary']))
    if (50000 - sum(s0)) > 0:
        return True
    else:
        return False

#Gneerate a random lineup
def genIter():
    r0 = []
    while len(r0) < 6:
        r = np.random.randint(0,high=len(dk_merge))
        if r not in r0:
            r0.append(r)
    return r0

#Convert IDs into Names
def getNames(lineup):
    n0 = []
    for iden in lineup:
        n0.append(dk_merge.loc[int(iden)]['Name'])
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
    

##main

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
dk_proximity['Proxy'] = dk_proximity["AVG"].apply(lambda x: distance(x))
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

#past results course
dk_pastResults = pd.read_html('https://www.espn.com/golf/leaderboard/_/tournamentId/982')
dk_pastResults = dk_pastResults[0]
dk_pastResults.drop(['POS','TO PAR','R1','R2','R3','R4','EARNINGS','FEDEX PTS'],axis=1,inplace=True)
dk_pastResults.drop(dk_pastResults[dk_pastResults['TOT'] < 170].index,inplace=True)
dk_pastResults.rename(columns={'PLAYER':'Name'}, inplace=True)

#past results tournament
dk_pastResults1 = pd.read_html('https://www.espn.com/golf/leaderboard/_/tournamentId/401056547')
dk_pastResults1 = dk_pastResults1[0]
dk_pastResults1.drop(['POS','TO PAR','R1','R2','R3','R4','EARNINGS','FEDEX PTS'],axis=1,inplace=True)
dk_pastResults1.drop(dk_pastResults1[dk_pastResults1['TOT'] < 170].index,inplace=True)
dk_pastResults1.rename(columns={'PLAYER':'Name','TOT':'TOT1'}, inplace=True)

#merge all dataframes - Can probably do this in a loop?
dk_merge = pd.merge(df,dk_par3efficiency, how='left', on='Name')
dk_merge = pd.merge(dk_merge,dk_par3efficiency1, how='left', on='Name')
dk_merge = pd.merge(dk_merge, dk_par4efficiency ,how='left',on='Name')
dk_merge = pd.merge(dk_merge, dk_proximity, how='left',on='Name')
dk_merge = pd.merge(dk_merge, dk_eagle, how='left',on='Name')
dk_merge = pd.merge(dk_merge, dk_pastResults,how='left',on='Name')
dk_merge = pd.merge(dk_merge, dk_pastResults1,how='left',on='Name')

#define variables
maxIter = 0                                    #placeholder for best lineup
i = 0                                          #main iterable
j = 0                                          #top tier iterable
k = 0                                          #top tier lineupdf iterable
topTier = pd.DataFrame(columns=['Player'])     #a dataframe of the top lineups (mean+sigma)
topTierLineup = pd.DataFrame(columns=['Player1','Player2','Player3','Player4','Player5','Player6','TOT'])

#scale for avg DK points 10 - 30 uniform
avgPointsScale = np.linspace(10,30,len(dk_merge['AvgPointsPerGame'].dropna()))

#scale for to par efficiency 0 - 10 uniform
parEfficiencyScale = np.concatenate((np.linspace(10,0,len(dk_merge['Par3Eff_225-250'].dropna())),np.zeros(len(dk_merge)-len(dk_merge['Par3Eff_225-250'].dropna()))))

#proximity scale 0 - 10 uniform
proximityScale = np.concatenate((np.linspace(10,0,len(dk_merge['Proximity'].dropna())),np.zeros(len(dk_merge)-len(dk_merge['Proximity'].dropna()))))

#eagle scale 0 - 10 uniform
eagleScale = np.concatenate((np.linspace(10,0,len(dk_merge['Eagles'].dropna())),np.zeros(len(dk_merge)-len(dk_merge['Eagles'].dropna()))))

#pastresults course scale 0 - 5 uniform
pastResultsScale = np.concatenate((np.linspace(5,0,len(dk_merge['TOT'].dropna())),np.zeros(len(dk_merge)-len(dk_merge['TOT'].dropna()))))

#pastresults tournament scale 0 - 5 uniform
pastResultsScale1 = np.concatenate((np.linspace(5,0,len(dk_merge['TOT1'].dropna())),np.zeros(len(dk_merge)-len(dk_merge['TOT1'].dropna()))))


#fill NaN values
dk_merge['TOT'] = dk_merge['TOT'].fillna(300)
dk_merge['TOT1'] = dk_merge['TOT1'].fillna(300)
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

#add eagle scale
dk_merge.sort_values(by='Eagles',ascending=False,inplace=True)
dk_merge['Eag'] = eagleScale

#add past results course
dk_merge.sort_values(by='TOT', ascending=True,inplace=True)
dk_merge['PR'] = pastResultsScale

#add past results tournament
dk_merge.sort_values(by='TOT1', ascending=True,inplace=True)
dk_merge['PR1'] = pastResultsScale1

#reshape dataframe to sort by Salary
dk_merge.sort_values(by='Salary',ascending=False,inplace=True)
#dk_merge.to_csv('DKData_RStG.csv', index = False)  #optional line if you want to see data in CSV

dk_merge.drop(['AvgPointsPerGame','Par3Eff_150-175','Par3Eff_225-250','Par4Eff_400-450','Eagles','Proximity','TOT','TOT1'],axis=1,inplace=True)
column_list = ['APPG','P3E','P4E','Prox','P3E1','Eag','PR','PR1']
dk_merge['Total'] = dk_merge[column_list].sum(axis=1)
dk_merge.drop(column_list,axis=1,inplace=True)
#dk_merge.to_csv('DKFinal.csv', index = False) #optional line if you want to see data in CSV

#Calculate Mean and Standard Deviation
mean = dk_merge['Total'].mean()*6
sigma = dk_merge['Total'].std()*6

#Sanity Check - preview
print(dk_merge.head())

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
    if currentIter > (mean+(1.25*sigma)) and constraint(lineup):
        #add players to top tier dataframe
        topTierData = getNames(lineup)
        topTierData.append(currentIter)
        topTierLineup.loc[k] = topTierData
        k = k + 1
        for x in lineup:
            topTier.loc[j] = dk_merge.loc[x]['Name']
            j = j + 1
    #iterate only if it is a valid lineup
    if constraint(lineup):
        i = i + 1
    #counter
    if i % 1000 == 0:
        print(i)

#print data for easy view
topTier = topTier[topTier.groupby('Player')['Player'].transform('size') > 10]
topPlayers = pd.DataFrame(topTier['Player'].value_counts())
topPlayers['Name'] = topPlayers.index
#topPlayers.to_csv('DK_TopPlayers.csv', index = False)   #optional line if you want a csv of the most used players
#topTierLineup.to_csv('DK_TopLineup.csv',index = False)  #optional line if you want to view the best lineups
print(maxIter)
print(getNames(maxLineup))

