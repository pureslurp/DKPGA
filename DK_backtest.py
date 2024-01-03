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
import math



def run_iteration(df_merge):
    topTierLineup = pd.DataFrame(columns=['Player1','Player2','Player3','Player4','Player5','Player6','TOT'])
    k = 0
    while k < 400000:
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

    OptimizedLineup = optimize_main(topTierLineup, df_merge)
    MaximizedLineup = maximize_main(OptimizedLineup, df_merge)
    MaximizedLineup = remove_outliers_main(MaximizedLineup, df_merge)
    #MaximizedLineup = optimize_ownership(MaximizedLineup, df_merge)
    return MaximizedLineup

def get_lineup_score(df_merge, max_lineup):
    max_list = max_lineup[0:-2]
    list_score = []
    for player in max_list:
        player_df = df_merge[df_merge["Name + ID"] == player]
        if math.isnan(float(player_df["DK Score"].values)):
            print(player)
            sys.exit()
        list_score.append(float(player_df["DK Score"].values))
    print(np.sum(list_score))
    return np.sum(list_score)
    

def get_max(df_merge, optimize):
    df_merge.sort_values(by=["Total"], ascending=False, inplace=True)
    total_max = df_merge["DK Score"].iloc[0:10].sum()
    df_merge.sort_values(by=["Value"], ascending=False, inplace=True)
    value_max = df_merge["DK Score"].iloc[0:10].sum()
    if total_max + value_max > float(optimize["max"]):
        optimize["max"] = str(round(total_max + value_max,2))
        optimize["odds"] = str(round(df_merge["Odds"].max()*odd,2))
        optimize["data_total"] = str(round(df_merge["Sum"].max()*sum,2))
        optimize["odds_multi"] = str(round(odd,2))
        optimize['data_multi'] = str(round(sum,2))

path_dict = {
    "River" : "dk_points_id_401465534.csv",
    # "Oakdale" : "dk_points_id_401465532.csv",
    "BayHill" : "dk_points_id_401465524.csv",
    "Waialae" : "dk_points_id_401465513.csv",
    "Kapalua" : "dk_points_id_401465512.csv",
    "Stadium" : "dk_points_id_401465518.csv",
    "Torrey" : "dk_points_id_401465516.csv",
    # "Pebble" : "dk_points_id_401465517.csv",
    "Quinta" : "dk_points_id_401465514.csv",
    "DGC" : "dk_points_id_401465535.csv",
    "Deere" : "dk_points_id_401465536.csv",
    "Renaissance" : "dk_points_id_401465537.csv",
    "Twin" : "dk_points_id_401465541.csv",
    "Sedgefield" : "dk_points_id_401465542.csv",
    "Southwind" : "dk_points_id_401465543.csv"
}


odds_multi = np.linspace(0.1, 2, 5)
op_list = []
# optimize_df = pd.DataFrame(columns=['COURSE','LINEUP_DK_SCORE','ODDS','PR','ODDS_MULTI','PR_MULTI'])

for key in path_dict:
    optimize = {
        "odds" : 0,
        "data_total" : 0,
        "odds_multi" : 0,
        "data_multi" : 0,
        "max" : 0
    }
    path = f'2023/{key}/'
    score = 0
    pr = 0
    odds = 0
    
    for sum in odds_multi:
        for odd in odds_multi:
            print(key)
            check = False
            optimize_df = pd.read_csv("DK_Backtest.csv")
            for index, row in optimize_df.iterrows():
                if row["ODDS_MULTI"] == odd and row["PR_MULTI"] == sum and row["COURSE"] == key and not math.isnan(row["LINEUP_DK_SCORE"]) :
                    print("Already accounted for ", key, row["ODDS_MULTI"], row["PR_MULTI"])
                    check = True
                    break
            if not check:
                total_max = 0
                value_max = 0
                df = pd.read_csv('{}CSVs/{}'.format(path,"DKData.csv"))
                try:
                    past_results = pd.read_csv(f'past_results/2022/{path_dict[key]}')
                except:
                    print(path_dict[key][-13:-4])
                    dk_points_df(path_dict[key][-13:-4])
                    past_results = pd.read_csv(f'past_results/2022/{path_dict[key]}')
                past_results.drop(['Rank'],axis=1,inplace=True)
                df_merge = pd.merge(df, past_results, on="Name", how='left')
                print(df_merge.iloc[:,4:-2].columns)
                df_merge["Sum"] = df_merge.iloc[:,4:-2].sum(axis=1)

                df_merge["Sum"] = df_merge["Sum"].rank(pct=True, ascending=True)
                print(df_merge.sort_values(by=["Sum"], ascending=False))

                df_merge["Sum"] = df_merge["Sum"] * 5
                print(df_merge.sort_values(by=["Sum"], ascending=False))
                df_merge = df_merge[["Name + ID", "Name", "Salary", "Odds", "Sum", "DK Score"]]

                df_merge["Odds"] = df_merge["Odds"].rank(pct=True, ascending=True)
                df_merge["Odds"] = df_merge["Odds"] * 10
                
                df_merge["Total"] = (df_merge["Odds"]*odd) + (df_merge["Sum"]*sum)
                
                df_merge["Value"] = df_merge["Total"] / df_merge['Salary']
                print(df_merge.sort_values(by=["Total"], ascending=False))
                print("Odd: ", odd, " PR: ", sum)
                #get_max(df_merge, optimize)
                max_lineup = run_iteration(df_merge).iloc[0].values.flatten().tolist()
                print(max_lineup)
                curr_score = get_lineup_score(df_merge, max_lineup)
                print(curr_score)
                new_row = [key, curr_score, df_merge["Odds"].max()*odd, df_merge["Sum"].max()*sum, odd, sum]
                optimize_df.loc[len(optimize_df)] = new_row
                print(optimize_df.head())
                optimize_df.to_csv("DK_Backtest.csv", index=False)
        




            
    #op_list.append(optimize)

# odds_list = []
# data_list = []
# odds_multi_list = []
# data_multi_list = []
# for op in op_list:
#     odds_list.append(float(op["odds"]))
#     data_list.append(float(op["data_total"]))
#     odds_multi_list.append(float(op["odds_multi"]))
#     data_multi_list.append(float(op['data_multi']))

# print("odds mean and median: ", mean(odds_list), median(odds_list))
# print("past results mean and median: ", mean(data_list), median(data_list))
# print("odds multi mean and median: ", mean(odds_multi_list), median(odds_multi_list))
# print("past results multi mean and median: ", mean(data_multi_list), median(data_multi_list))
        
        


