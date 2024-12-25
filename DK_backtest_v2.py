from Legacy.pga_dk_scoring import dk_points_df
from pga_v4 import fix_names, odds_to_score, run_iter, LineUp
import pandas as pd

# _2024_TOURNAMENT_LIST = [401580329, 401580330]

_2024_TOURNAMENT_LIST = {
    "Valspar_Championship" : 401580341,
    "Texas_Children's_Houston_Open" : 401580342,
    "Valero_Texas_Open" : 401580343,
    "Masters_Tournament": 401580344,
    "RBC_Heritage": 401580345,
    "THE_CJ_CUP_Byron_Nelson" : 401580348,
    "Wells_Fargo_Championship": 401580349,
    "PGA_Championship" : 401580351,
    "Charles_Schwab_Challenge": 401580352,
    "RBC_Canadian_Open": 401580353,
    "the_Memorial_Tournament_pres._by_Workday" : 401580354,
    "U.S._Open" : 401580355,
    "Travelers_Championship": 401580356,
    "Rocket_Mortgage_Classic": 401580357,
    "John_Deere_Classic" : 401580358,
    "Genesis_Scottish_Open": 401580359,
    "The_Open": 401580360,
    "3M_Open": 401580362,
    "Wyndham_Championship": 401580363
}

OPTIMIZE = {
    "Tournament Winner" : [0, 1, 2],
    "Top 5 Finish" : [0, 1, 2],
    "Top 10 Finish" : [0,1,2],
    "Top 20 Finish" : [0,1,2]
}

optimize_df = pd.DataFrame(columns=['COURSE','TOP_LINEUP_DK_SCORE', 'TOP_OVERALL_DK_SCORE','AVG_LINEUP_DK_SCORE','W','T5','T10','T20',"Comb"])

def get_dk_score_per_row(row, df):
    lineup = LineUp(row["G1"], row["G2"], row["G3"], row["G4"], row["G5"], row["G6"])
    return lineup.get_past_score(df)


for k, v in _2024_TOURNAMENT_LIST.items():
    sal_df = pd.read_csv(f'2024/{k}/DKSalaries.csv')
    sal_df["Name"] = sal_df["Name"].apply(lambda x: fix_names(x))
    sal_df = sal_df[sal_df["Salary"] > 6400]

    try:
        past_results = pd.read_csv(f'past_results/2024/dk_points_id_{v}.csv')
    except:
        dk_points_df(v)
        past_results = pd.read_csv(f'past_results/2024/dk_points_id_{v}.csv')
    
    for w in [0, 1, 2]:
        for t5 in [0, 1, 2]:
            for t10 in [0, 1, 2]:
                for t20 in [0, 1, 2]:
                    if w != 0 or t5 != 0 or t10 != 0 or t20 != 0:
                        dk_backtest = pd.read_csv('DK_Backtest.csv')
                        if len(dk_backtest[(dk_backtest["COURSE"] == k) & (dk_backtest["W"] == w) & (dk_backtest["T5"] == t5) & (dk_backtest["T10"] == t10) & (dk_backtest["T20"] == t20)]) == 0:
                            df = pd.read_csv(f'2024/{k}/odds.csv')
                            odds_headers = list(df.columns.values)[1:]
                            df["Name"] = df["Name"].apply(lambda x: fix_names(x))
                            df_merge = pd.merge(past_results, df, on="Name", how="left")
                            df_merge = pd.merge(df_merge, sal_df, on="Name", how="left")
                            print(w, t5, t10, t20)
                            for header in odds_headers:
                                df_merge[header] = df_merge[header].apply(lambda x: odds_to_score(x, header, w=w, t5=t5, t10=t10, t20=t20))
                            df_merge["Total"] = df_merge[odds_headers].sum(axis=1)
                            df_merge["Value"] = df_merge["Total"] / df_merge["Salary"] * 1000
                            dkRoster = run_iter(500000, df_merge)
                            dkRoster.lineups["DK Score"] = dkRoster.lineups.apply(lambda x: get_dk_score_per_row(x, df_merge), axis=1)
                            # top_lineup = LineUp(dkRoster.iloc[0]["G1"], dkRoster.iloc[0]["G2"], dkRoster.iloc[0]["G3"], dkRoster.iloc[0]["G4"], dkRoster.iloc[0]["G5"], dkRoster.iloc[0]["G6"])
                            avg = dkRoster.lineups.loc[:, 'DK Score'].mean()
                            maxValue = dkRoster.lineups.loc[:, 'DK Score'].max()
                            row = [k, dkRoster.lineups.iloc[0]["DK Score"], maxValue, avg, w, t5, t10, t20, f"{w},{t5},{t10},{t20}"]
                            dk_backtest.loc[len(dk_backtest)] = row
                            dk_backtest.to_csv('DK_Backtest.csv', index=False)
                        else:
                            continue



