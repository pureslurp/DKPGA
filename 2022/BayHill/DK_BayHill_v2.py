#!/usr/bin/env python
# coding: utf-8

##libraries
import warnings
import sys
import numpy as np
sys.path.insert(0, "/Users/seanraymor/Documents/Python Scripts/DKPGA")
from pgafunc import *
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import pandas as pd
warnings.simplefilter(action='ignore', category=Warning)


#user input
iterations = 300000

#define variables
maxIter = 0                                    #placeholder for best lineup
i = 0                                          #main iterable
j = 0                                          #top tier iterable
k = 0                                          #top tier lineupdf iterable
topTierLineup = pd.DataFrame(columns=['Player1','Player2','Player3','Player4','Player5','Player6','TOT'])

path = '2022/BayHill/'
name = 'DKSalaries-BayHill.csv'
df = DK_csv_assignemnt(path, name)
scope, creds, client = google_credentials()
course_df = assign_course_df(client)


'''
Options for course/player attributes

Driving:
- Driving Distance = driving_distance(df)
- Driving Accuracy = driving_accuracy(df)

Approach:
- Strokes Gained around green = around_green(df)
- Proximity (125-150) = proximity_125to150(df)

Putting:
- Strokes Gained putting = putting(df)

General:
- Par 5 Eagle Percentage = par5_eaglePercentage(df)
- Birdie or Better Percentage = birdie_or_better(df)
- Yardage efficiency = weight_efficiencies(df, course_df)
- Course fit from datagolf.com = course_fit(df)
- Drop players below a certain salary range = drop_players_lower_than(df, salary)
'''

df = drop_players_lower_than(df, 6500)
df_merge = weight_efficiencies(df, course_df)
df_merge = pga_odds(df_merge)
df_merge = course_fit(df_merge)
df_merge = past_results(df_merge, 'https://www.espn.com/golf/leaderboard?tournamentId=401243004', upperBound=5)
df_merge = past_results(df_merge, 'https://www.espn.com/golf/leaderboard/_/tournamentId/401155427', upperBound=2.5, pr_i=1)
#df_merge = putting(df_merge)

#delete_unused_columns(df_merge)

df_merge.to_csv('{}/CSVs/DKData.csv'.format(path), index = False) #optional line if you want to see data in CSV

df_merge = df_total_and_reformat(df_merge)

df_merge.to_csv('{}/CSVs/DKFinal.csv'.format(path), index = False) #optional line if you want to see data in CSV


#Calculate Mean and Standard Deviation
mean = df_merge['Total'].mean()*6
sigma = df_merge['Total'].std()*6

print(df_merge.head())

while i < iterations:
    #get a sample
    lineup = genIter(df_merge)
    lineup.sort()
    #assign sample
    currentIter = objective(lineup, df_merge)
    #check if sample is better than current best sample
    if currentIter > maxIter and constraint(lineup, df_merge):
        #reassign
        maxIter = currentIter
        maxLineup = lineup
    #check if sample is a top tier sample
    if currentIter > (mean + sigma - 5) and constraint(lineup, df_merge):
        #add players to top tier dataframe
        topTierData = getNames(lineup, df_merge)
        topTierData.append(currentIter)
        topTierLineup.loc[k] = topTierData
        k = k + 1
        
    #iterate only if it is a valid lineup
    if constraint(lineup, df_merge):
        i = i + 1
    #counter
    if i % 1000 == 0:
        print(i)

#optimize data after loop
OptimizedLineup = optimize_main(topTierLineup, df_merge)
MaximizedLineup = maximize_main(OptimizedLineup, df_merge)
MaximizedLineup = remove_outliers_main(MaximizedLineup, df_merge)
topPlayers = find_top_players_for(MaximizedLineup, df_merge)


topPlayers.to_csv('{}CSVs/Top_Players.csv'.format(path),index=False)
MaximizedLineup.to_csv('{}CSVs/Maximized_Lineups.csv'.format(path),index=False)

print(MaximizedLineup.head())

