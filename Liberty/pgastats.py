#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 18 20:17:02 2021

@author: seanraymor
"""
import pandas as pd
import numpy as np

def getEff(df, key, count):
    pgaEff = [['150-175 Eff',3,'https://www.pgatour.com/stats/stat.02520.html'],
              ['175-200 Eff',3,'https://www.pgatour.com/stats/stat.02521.html'],
              ['200-225 Eff',3,'https://www.pgatour.com/stats/stat.02522.html'],
              ['225-250 Eff',3,'https://www.pgatour.com/stats/stat.02523.html'],
              ['300-350 Eff',4,'https://www.pgatour.com/stats/stat.02527.html'],
              ['350-400 Eff',4,'https://www.pgatour.com/stats/stat.02528.html'],
              ['400-450 Eff',4,'https://www.pgatour.com/stats/stat.02529.html'],
              ['450-500 Eff',4,'https://www.pgatour.com/stats/stat.02530.html'],
              ['500+ Eff',   4,'https://www.pgatour.com/stats/stat.02531.html'],
              ['500-550 Eff',5,'https://www.pgatour.com/stats/stat.02533.html'],
              ['550-600 Eff',5,'https://www.pgatour.com/stats/stat.02534.html'],
              ['600-650 Eff',5,'https://www.pgatour.com/stats/stat.02535.html']]
    
    columns = ['Range','Par','URL']
    pgaStats = pd.DataFrame(pgaEff,columns=columns)
    url = pgaStats[pgaStats['Range'] == key]['URL'].item()
    
    eff_df = pd.read_html(url)
    eff_df = eff_df[1]
    eff_df.drop(['RANK LAST WEEK','ROUNDS','TOTAL STROKES','TOTAL ATTEMPTS'], axis=1, inplace=True)
    eff_df.drop(eff_df.columns[0],axis=1,inplace=True)
    eff_df.rename(columns={'PLAYER NAME':'Name','AVG':key}, inplace=True)
    
    dk_merge = pd.merge(df,eff_df, how='left', on='Name')

    effScale = np.concatenate((np.linspace(count*2,0,len(dk_merge[key].dropna())),np.zeros(len(dk_merge)-len(dk_merge[key].dropna()))))
    dk_merge.sort_values(by=key,ascending=True,inplace=True)
    dk_merge[key] = effScale

    return dk_merge

