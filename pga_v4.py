import pandas as pd
import random
from alive_progress import alive_bar

TOURNEY = "TOUR_Championship"

def fix_names(name):
    if name == "Si Woo":
        return "si woo kim"
    elif name == "Byeong Hun":
        return "byeong hun an"
    elif name == "Erik Van":
        return "erik van rooyen"
    elif name == "Adrien Dumont":
        return "adrien dumont de chassart"
    elif name == "Matthias Schmid":
        return "matti schmid"
    elif name == "Samuel Stevens":
        return "sam stevens"
    elif name == "Benjamin Silverman":
        return "ben silverman"
    elif name =="Min Woo":
        return "min woo lee"
    elif name == "Santiago De":
        return "santiago de la fuente"
    elif name == "Jose Maria":
        return "jose maria olazabal"
    elif name == "Niklas Norgaard Moller":
        return "niklas moller"
    elif name == "Jordan L. Smith":
        return "jordan l."
    elif name == "daniel bradbury":
        return "dan bradbury"
    else:
        return name.lower()

def odds_to_score(col, header, w=1, t5=1, t10=1, t20=1):
    '''
    win:  30 pts
    top 5: 14
    top 10: 7
    top 20: 5
    '''
    if col < 0:
        final_line = (col/-110)
    else:
        final_line = (100/col)
    match header:
        case "Tournament Winner":
            return round(final_line * 30 * w, 3)
        case "Top 5 Finish":
            return round(final_line * 14 * t5, 3)
        case "Top 10 Finish":
            return round(final_line * 7 * t10, 3)
        case "Top 20 Finish":
            return round(final_line * 5 * t20, 3)
        
class Golfer:
    def __init__(self, golfer: pd.DataFrame):
        try:
            self.name = golfer["Name + ID"].iloc[0]
            self.salary = golfer["Salary"].iloc[0]
            self.total = golfer["Total"].iloc[0]
            self.value = golfer["Value"].iloc[0]
        except:
            try:
                self.name = golfer["Name + ID"]
                self.salary = golfer["Salary"]
                self.total = golfer["Total"]
                self.value = golfer["Value"]
            except:
                raise Exception("Unable to assign ", golfer)
            
    
    def __str__(self):
        return self.name
    
    def __repr__(self):
        return f"{self.name}"
    
    def __eq__(self, other):
        return self.name == other.name

class LineUp:
    def __init__(self, g1: Golfer, g2: Golfer, g3: Golfer, g4: Golfer, g5: Golfer, g6: Golfer):
        self.golfers = {
            "G1": g1,
            "G2": g2,
            "G3": g3,
            "G4": g4,
            "G5": g5,
            "G6": g6,
        }
    def __str__(self):
        return f"Lineups: {self.golfers}"
    
    def get_salary(self):
        "a function that will return the sum of the LineUps total salary"
        salary = 0
        for key in self.golfers:
            salary += self.golfers[key].salary
        return salary
        
    def get_total(self):
        "a function that will return the sum of the LineUps total projected score"
        total = 0
        for key in self.golfers:
            total += self.golfers[key].total
        return total
    
    def duplicates(self):
        "a function that will check the LineUp for duplicates"
        elem = []
        for key in self.golfers:
            elem.append(self.golfers[key].name)
        if len(elem) == len(set(elem)):
            return False
        else:
            return True
    
    def count_10k(self):
        '''condition one is that there should not be more than 1 golfer above 10000 in a lineup'''
        count = 0
        for key in self.golfers:
            if self.golfers[key].salary >= 10000:
                count += 1
        return count
    
    def count_6k(self):
        count = 0
        for key in self.golfers:
            if self.golfers[key].salary <= 6000:
                count += 1
        return count

    
    def is_valid(self):
        if not self.duplicates() and self.get_salary() <= 50000 and self.count_10k() <= 1 and self.count_6k() <= 1:
            return True
        else:
            return False
        
        
    def to_dict(self):
        "a function that will export the LineUp with salary and total points to a dictionary"
        self.golfers.update(
            {"Salary" : self.get_salary(),
             "TotalPoints" : self.get_total()})
        return self.golfers
    
    def get_lowest_sal_player(self):
        "a function that returns the player with the lowest salary (excluding defense)"
        _low_sal = 10000
        for key, value in self.golfers.items():
            if key != "DST":
                if value.salary < _low_sal:
                    _low_sal = value.salary
                    low_player = value
                    low_player_pos = key
        return low_player, low_player_pos
    
    @property
    def names(self):
        "a function that returns a list of Players in the LineUp"
        names = []
        for key in self.golfers:
            names.append(self.golfers[key].name)
        return names

    def optimize(self, df):
        for pos, golfer in self.golfers.items():
            budget = self.get_salary()
            if self.count_10k() >= 1:
                filt_df = df[df["Salary"] <= 10000]
                if self.count_6k() >= 1:
                    filt_df = filt_df[filt_df["Salary"] >= 6000]
                else: 
                    filt_df = df
            else:
                if self.count_6k() >= 1:
                    filt_df = df[df["Salary"] >= 6000]
                else:
                    filt_df = df
            filt_df = filt_df[(filt_df["Salary"] <= int(golfer.salary) + min(500, 50000-budget)) & (filt_df["Salary"] >= int(golfer.salary) - 500)] 
            for _, r2 in filt_df.iterrows():
                new_player = Golfer(r2)
                if new_player.total > self.golfers[pos].total and new_player.name not in self.names:
                    print(f"Replacing {self.golfers[pos].name} with {new_player.name}" )
                    self.golfers[pos] = new_player
        _low_player, _low_player_pos = self.get_lowest_sal_player()
        _budget = self.get_salary()
        if self.count_10k() >=1:
            df = df[df["Salary"] <= 10000]
        if self.count_6k() >= 1:
            df = df[df["Salary"] >= 6000]
        df = df[(df["Salary"] <= int(_low_player.salary) + 50000-_budget) & (df["Salary"] >= int(_low_player.salary))] 
        for _, r2 in df.iterrows():
            new_player = Golfer(r2)
            if new_player.total > _low_player.total and new_player.name not in self.names:
                print(f"Replacing {_low_player.name} with {new_player.name}")
                _low_player = new_player
                self.golfers[_low_player_pos] = new_player
        return self
    
    def get_past_score(self, backtest_df):
        score = 0
        for key in self.golfers:
            score += float(backtest_df[backtest_df["Name + ID"] == self.golfers[key].name]["DK Score"])
        return score

    def get_position(self, golfer):
        for pos, _golfer in self.golfers.items():
            if golfer.name == _golfer.name:
                return pos

    def check_if_player_exists(self, golfer: Golfer):
        for _, _golfer in self.golfers.items():
            if golfer.name == _golfer.name:
                return True
        return False
    
    def replace_player(self, df, golfer, avoid: list[Golfer] = []):
        budget = self.get_salary()
        pos = self.get_position(golfer)
        avoid = [g.name for g in avoid]
        f = df["Name + ID"].isin(avoid)
        df = df[~f].dropna()
        filt_df = df[(df["Salary"] <= int(golfer.salary) + min(500, 50000-budget)) & (df["Salary"] >= int(golfer.salary) - 1900)] 
        _high_score = 0
        for _, r2 in filt_df.iterrows():
            new_player = Golfer(r2)
            if new_player.total > _high_score and new_player.name not in self.names:
                _high_score = new_player.total
                print(f"Ownership: Replacing {self.golfers[pos].name} with {new_player.name}" )
                self.golfers[pos] = new_player
        return self

class LineUps:
    def __init__(self, lineups: pd.DataFrame):
        self.lineups = lineups


    def optimize_lineups(self, df: pd.DataFrame):
        "a function that optimizes a set of lineups by going through each player and comparing the above and below players"
        for index, lineup in self.lineups.iterrows():
            lineup_obj = LineUp(Golfer(df[df["Name + ID"] == lineup["G1"].name]), Golfer(df[df["Name + ID"] == lineup["G2"].name]),  Golfer(df[df["Name + ID"] == lineup["G3"].name]),  Golfer(df[df["Name + ID"] == lineup["G4"].name]),  Golfer(df[df["Name + ID"] == lineup["G5"].name]), Golfer(df[df["Name + ID"] == lineup["G6"].name]))
            lineup_obj = lineup_obj.optimize(df)
            lineup["TotalPoints"] = lineup_obj.get_total()
            if lineup["TotalPoints"] in self.lineups["TotalPoints"].values:
                print("duplicate")
                continue
            else:
                self.lineups.iloc[index] = list(lineup_obj.to_dict().values())
        
        return self.lineups

    def get_lineups_oversub(self, df, oversub=0.66):
        player_count = pd.DataFrame(columns=["Name + ID", "Count"])
        pos = ["G1", "G2", "G3", "G4", "G5", "G6"]
        for _, player in df.iterrows():
            j = 0
            player = Golfer(player)
            for _, lineup in self.lineups.iterrows():
                for golfer in lineup[0:6]:
                    if golfer.name == player.name:
                        j += 1
            player_count.loc[len(player_count)] = [player.name, j/len(self.lineups)]

        player_count = player_count[player_count["Count"] > oversub]
        df_merge = pd.merge(player_count, df, on="Name + ID", how="left")
        oversub_list = []
        for _, g in df_merge.iterrows():
            oversub_list.append(Golfer(g))
        return oversub_list
    

    def optimize_ownership(self, df):
        initial_oversubbed_players = self.get_lineups_oversub(df)
        subs_needed = False
        if len(initial_oversubbed_players) > 0:
            subs_needed = True
        self.lineups.sort_values(by="TotalPoints", ascending=True, inplace=True, ignore_index=True)
        while subs_needed:    
            for oversubbed_player in initial_oversubbed_players:
                for index, lineup in self.lineups.iterrows():
                    lineup_obj = LineUp(Golfer(df[df["Name + ID"] == lineup["G1"].name]), Golfer(df[df["Name + ID"] == lineup["G2"].name]),  Golfer(df[df["Name + ID"] == lineup["G3"].name]),  Golfer(df[df["Name + ID"] == lineup["G4"].name]),  Golfer(df[df["Name + ID"] == lineup["G5"].name]), Golfer(df[df["Name + ID"] == lineup["G6"].name]))
                    if lineup_obj.count_10k() >= 1 and oversubbed_player.salary <= 10000:
                        new_df = df[df["Salary"] <= 10000]
                    else:
                        new_df = df
                    if lineup_obj.check_if_player_exists(oversubbed_player):
                        lineup_obj.replace_player(new_df, oversubbed_player, initial_oversubbed_players) # Need to figuer out how to pass all oversubs
                        lineup["TotalPoints"] = lineup_obj.get_total()
                        self.lineups.iloc[index] = list(lineup_obj.to_dict().values())
                        oversubbed_players = self.get_lineups_oversub(new_df)
                        [initial_oversubbed_players.append(ops) for ops in oversubbed_players if ops not in initial_oversubbed_players]
                        if len(oversubbed_players) == 0:
                            subs_needed = False
                        if oversubbed_player not in oversubbed_players:
                            break
                    else:
                        continue
        return self




def run_iter(iter, dk_merge, NoL=22):
    dkRoster = pd.DataFrame(columns=("G1", "G2", "G3", "G4", "G5", "G6", "Salary", "TotalPoints"))
    with alive_bar(iter) as bar:
        for _ in range(iter):
            highest_points = 0
            rnd_lineup = random.sample(range(len(dk_merge)), 6)
            lineup = LineUp(Golfer(dk_merge.iloc[rnd_lineup[0]]), 
                            Golfer(dk_merge.iloc[rnd_lineup[1]]),
                            Golfer(dk_merge.iloc[rnd_lineup[2]]),
                            Golfer(dk_merge.iloc[rnd_lineup[3]]),
                            Golfer(dk_merge.iloc[rnd_lineup[4]]),
                            Golfer(dk_merge.iloc[rnd_lineup[5]]))
            if  lineup.is_valid() and (lineup.get_total() > highest_points):
                dkRoster.loc[len(dkRoster)] = lineup.to_dict()
                dkRoster.sort_values(by="TotalPoints", ascending=False, inplace=True, ignore_index=True)
                dkRoster = dkRoster.iloc[0:NoL]
                if len(dkRoster) == (NoL + 1):
                    highest_points = float(dkRoster.iloc[NoL]["TotalPoints"])
            bar()
    
    dkRoster = LineUps(dkRoster)
    dkRoster.optimize_lineups(dk_merge)
    dkRoster.optimize_ownership(dk_merge)
    dkRoster.lineups.sort_values(by="TotalPoints", ascending=False, inplace=True, ignore_index=True)

    return dkRoster


def main(iter=500000):
    odds_df = pd.read_csv(f'2024/{TOURNEY}/odds.csv')
    odds_df["Name"] = odds_df["Name"].apply(lambda x: fix_names(x))
    odds_headers = list(odds_df.columns.values)[1:]
    for header in odds_headers:
        odds_df[header] = odds_df[header].apply(lambda x: odds_to_score(x, header, w=1, t5=0, t10=0, t20=0))
    odds_df["Total"] = odds_df[odds_headers].sum(axis=1)
    odds_df["Total"] = odds_df["Total"].apply(lambda x: round(x, 3))
    odds_df.to_csv(f'2024/{TOURNEY}/dk_odds_score.csv')

    dk_df = pd.read_csv(f'2024/{TOURNEY}/DKSalaries.csv')
    dk_df = dk_df[dk_df["Salary"] > 6400]
    dk_df["Name"] = dk_df["Name"].apply(lambda x: fix_names(x))
    print(odds_df.head())
    print(dk_df.head())
    dk_merge = pd.merge(dk_df, odds_df, on="Name", how="left")
    dk_merge["Value"] = dk_merge["Total"] / dk_merge["Salary"] * 1000
    dk_merge.to_csv(f'2024/{TOURNEY}/dk_final.csv')

    if len(dk_merge[dk_merge.isna().any(axis=1)]["Name"]) == 0:
        print("All players merged successfully")
    else:
        print("The following players were not merged")
        print(dk_merge[dk_merge.isna().any(axis=1)]["Name"])

    g1 = Golfer(dk_merge.iloc[0])
    
    dkRoster = run_iter(iter, dk_merge)
    dkRoster.lineups.to_csv(f"2024/{TOURNEY}/dk_lineups.csv")



if __name__ == "__main__":
    main()