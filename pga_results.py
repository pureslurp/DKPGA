from utils import TOURNAMENT_LIST_2025
from backtest.pga_dk_scoring import dk_points_df
from pga_v5 import fix_names
import pandas as pd

tournament = "3M_Open"

# Get results data
try:
    results_df = pd.read_csv(f'past_results/2025/dk_points_id_{TOURNAMENT_LIST_2025[tournament]["ID"]}.csv')
    results_df = results_df[['Name', 'DK Score']]
except FileNotFoundError:
    print(f"Results not found for {tournament}, generating them...")
    dk_points_df(TOURNAMENT_LIST_2025[tournament]["ID"])
    results_df = pd.read_csv(f'past_results/2025/dk_points_id_{TOURNAMENT_LIST_2025[tournament]["ID"]}.csv')
    results_df = results_df[['Name', 'DK Score']]

# Clean names in results
results_df['Name'] = results_df['Name'].apply(fix_names)

# Read lineups file
lineups_df = pd.read_csv(f'2025/{tournament}/dk_lineups_optimized.csv')

# Function to extract player name from lineup column
def extract_name(player_str):
    return fix_names(player_str.split('(')[0].strip())

# Process each golfer column
for col in ['G1', 'G2', 'G3', 'G4', 'G5', 'G6']:
    # Create new column name
    score_col = f'{col}_Score'
    # Extract name and lookup score
    lineups_df[score_col] = lineups_df[col].apply(extract_name).map(results_df.set_index('Name')['DK Score'])

# Add total actual score column
lineups_df['ActualPoints'] = lineups_df[[f'G{i}_Score' for i in range(1,7)]].sum(axis=1)

# Print summary statistics
print(f"\nLineup Score Summary:")
print(f"Highest Score: {lineups_df['ActualPoints'].max():.2f}")
print(f"Lowest Score: {lineups_df['ActualPoints'].min():.2f}")
print(f"Mean Score: {lineups_df['ActualPoints'].mean():.2f}")

# Save updated lineups
lineups_df.to_csv(f'2025/{tournament}/dk_lineups_optimized_results.csv', index=False)


