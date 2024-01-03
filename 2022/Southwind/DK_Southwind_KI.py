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

path = '2022/Southwind/'
name = 'DKSalaries-Southwind.csv'
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

df_merge = last_x_event_points_kev(df, events=6) #recent form

df_merge = past_results_kev(df_merge, 'https://www.espn.com/golf/leaderboard/_/tournamentId/401056559') #2019
df_merge = past_results_kev(df_merge, 'https://www.espn.com/golf/leaderboard/_/tournamentId/401155467', pr_i=1) #2020
df_merge = past_results_kev(df_merge, 'https://www.espn.com/golf/leaderboard/_/tournamentId/401243407', pr_i=2) #2021
                                

df_merge.to_csv('{}/CSVs/DKDataKI.csv'.format(path), index = False) #optional line if you want to see data in CSV

df_merge = df_total_and_reformat(df_merge)

df_merge.to_csv('{}/CSVs/DKFinalKI.csv'.format(path), index = False) #optional line if you want to see data in CSV