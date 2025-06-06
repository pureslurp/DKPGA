#!/usr/bin/env python
# coding: utf-8

##libraries
from itertools import count
import warnings
import sys
import numpy as np
sys.path.insert(0, "/Users/seanraymor/Library/Mobile Documents/com~apple~CloudDocs/Documents/Python Scripts/DKPGA")
from pgafunc_v3 import *
import pandas as pd
warnings.simplefilter(action='ignore', category=Warning)

#define variables
maxIter = 0                                    #placeholder for best lineup
i = 0                                          #main iterable
j = 0                                          #top tier iterable
k = 0                                          #top tier lineupdf iterable
topTierLineup = pd.DataFrame(columns=['Player1','Player2','Player3','Player4','Player5','Player6','TOT'])

path = '2023/DGC/'
name = 'DKSalaries-DGC.csv'
df = DK_csv_assignemnt(path, name)
# scope, creds, client = google_credentials()
# course_df = assign_course_df(client)


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

# df = drop_players_lower_than(df, 6200)
dk_points_df(401465535)

df_merge = last_x_events_dk_points(df, events=5)
#df_merge = last_x_event_points(df_merge, events=10, upper_bound=6.25, i=1)


df_merge = pga_odds_pga(df_merge, upper_bound=6)

df_merge.to_csv('{}/CSVs/DKData.csv'.format(path), index = False) #optional line if you want to see data in CSV

df_merge = df_total_and_reformat(df_merge)

df_merge.to_csv('{}/CSVs/DKFinal.csv'.format(path), index = False) #optional line if you want to see data in CSV

#Calculate Mean and Standard Deviation
mean = df_merge['Total'].mean()*6
sigma = df_merge['Total'].std()*6
print(mean, sigma)

print(df_merge.head())

while k < 500000:
    #get a sample
    lineup = genIter(df_merge)
    lineup.sort()
    #assign sample
    currentIter = objective(lineup, df_merge)
    #check if sample is a top tier sample

    # if currentIter > (mean + sigma) and constraint(lineup, df_merge):
        #add players to top tier dataframe
    if constraint(lineup, df_merge):
        topTierData = getNames(lineup, df_merge)
        topTierData.append(currentIter)
        topTierLineup.loc[len(topTierLineup)] = topTierData
        topTierLineup.sort_values(by=["TOT"], ascending=False, inplace=True, ignore_index=True)
        if len(topTierLineup) > 20:
            topTierLineup = topTierLineup.iloc[0:20]
        
        k = k + 1
        
    #iterate only if it is a valid lineup
    #counter
    if k % 1000 == 0:
        print(k)



#optimize data after loop
OptimizedLineup = optimize_main(topTierLineup, df_merge)
MaximizedLineup = maximize_main(OptimizedLineup, df_merge)
MaximizedLineup = remove_outliers_main(MaximizedLineup, df_merge)
MaximizedLineup = optimize_ownership(MaximizedLineup, df_merge)
countPlayers = calculate_oversub_count(MaximizedLineup, df_merge, csv=True)

countPlayers.to_csv('{}CSVs/Player_Ownership.csv'.format(path, index=False))
MaximizedLineup.to_csv('{}CSVs/Maximized_Lineups.csv'.format(path),index=False)

print(MaximizedLineup.head())

