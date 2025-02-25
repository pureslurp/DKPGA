import pandas as pd
import pulp
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from typing import Dict
from utils import TOURNAMENT_LIST_2025, fix_names

def load_data(tournament: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load salary and results data for the specified tournament
    """
    # Get tournament ID
    tournament_id = TOURNAMENT_LIST_2025[tournament]["ID"]
    
    # Load DraftKings salaries
    salary_path = f"2025/{tournament}/DKSalaries.csv"
    if not os.path.exists(salary_path):
        raise FileNotFoundError(f"Salary file not found: {salary_path}")
    
    dk_salaries = pd.read_csv(salary_path)
    
    # Load results
    results_path = f"past_results/2025/dk_points_id_{tournament_id}.csv"
    if not os.path.exists(results_path):
        raise FileNotFoundError(f"Results file not found: {results_path}")
        
    results = pd.read_csv(results_path)
    
    return dk_salaries, results

def merge_data(dk_salaries: pd.DataFrame, results: pd.DataFrame) -> pd.DataFrame:
    """
    Merge salary and results data, cleaning names for matching
    """
    # Clean names in both dataframes
    dk_salaries['Name'] = dk_salaries['Name'].apply(fix_names)
    results['Name'] = results['Name'].apply(fix_names)
    
    # Merge datasets
    merged = pd.merge(
        dk_salaries,
        results[['Name', 'DK Score']],
        on='Name',
        how='inner'
    )
    
    print(f"\nFound {len(merged)} players with both salary and result data")
    return merged

def optimize_lineup(data: pd.DataFrame, salary_cap: int = 50000) -> Dict:
    """
    Generate optimal lineup using Integer Linear Programming
    """
    # Create the optimization model
    prob = pulp.LpProblem("DraftKings_Historical_Lineup", pulp.LpMaximize)
    
    # Decision variables - whether to include each player
    decisions = pulp.LpVariable.dicts("players",
                                    ((p['Name + ID']) for _, p in data.iterrows()),
                                    cat='Binary')
    
    # Objective: Maximize total DK points
    prob += pulp.lpSum([decisions[p['Name + ID']] * p['DK Score'] for _, p in data.iterrows()])
    
    # Constraint 1: Must select exactly 6 players
    prob += pulp.lpSum([decisions[p['Name + ID']] for _, p in data.iterrows()]) == 6
    
    # Constraint 2: Must not exceed salary cap
    prob += pulp.lpSum([decisions[p['Name + ID']] * p['Salary'] for _, p in data.iterrows()]) <= salary_cap
    
    # Constraint 3: Must meet minimum salary requirement
    prob += pulp.lpSum([decisions[p['Name + ID']] * p['Salary'] for _, p in data.iterrows()]) >= 49000
    
    # Solve the optimization problem
    prob.solve(pulp.PULP_CBC_CMD(msg=False))
    
    if pulp.LpStatus[prob.status] != 'Optimal':
        raise Exception("Could not find optimal solution")
    
    # Extract the optimal lineup
    lineup = {
        'Players': [],
        'Salary': 0,
        'TotalPoints': 0
    }
    
    for _, player in data.iterrows():
        if decisions[player['Name + ID']].value() == 1:
            lineup['Players'].append({
                'Name': player['Name'],
                'Salary': player['Salary'],
                'Points': player['DK Score']
            })
            lineup['Salary'] += player['Salary']
            lineup['TotalPoints'] += player['DK Score']
    
    return lineup

def save_results(tournament: str, lineup: Dict):
    """
    Save the optimal lineup to a CSV file
    """
    # Create DataFrame from lineup data
    rows = []
    for i, player in enumerate(lineup['Players'], 1):
        rows.append({
            'Position': f'G{i}',
            'Name': player['Name'],
            'Salary': player['Salary'],
            'Points': player['Points']
        })
    
    df = pd.DataFrame(rows)
    df.loc['Total'] = ['', '', lineup['Salary'], lineup['TotalPoints']]
    
    # Save to CSV
    output_path = f"2025/{tournament}/optimal_historical_lineup.csv"
    df.to_csv(output_path, index=False)
    print(f"\nSaved optimal lineup to: {output_path}")
    
    # Print lineup details
    print("\nOptimal Historical Lineup:")
    print("-" * 50)
    for player in lineup['Players']:
        print(f"{player['Name']:<30} ${player['Salary']:,}  {player['Points']:.1f} pts")
    print("-" * 50)
    print(f"Total Salary: ${lineup['Salary']:,}")
    print(f"Total Points: {lineup['TotalPoints']:.1f}")

def main(tournament: str):
    """
    Main function to find the best historical lineup for a tournament
    """
    print(f"\nFinding optimal historical lineup for: {tournament}")
    
    # Load and merge data
    dk_salaries, results = load_data(tournament)
    merged_data = merge_data(dk_salaries, results)
    
    # Find optimal lineup
    optimal_lineup = optimize_lineup(merged_data)
    
    # Save and display results
    save_results(tournament, optimal_lineup)

if __name__ == "__main__":
    # if len(sys.argv) != 2:
    #     print("Usage: python dk_find_best_finish.py <tournament_name>")
    #     print("Example: python dk_find_best_finish.py Mexico_Open_at_VidantaWorld")
    #     sys.exit(1)
        
    tournament_name = "Mexico_Open_at_VidantaWorld"
    if tournament_name not in TOURNAMENT_LIST_2025:
        print(f"Error: Tournament '{tournament_name}' not found in TOURNAMENT_LIST_2025")
        sys.exit(1)
        
    main(tournament_name)
