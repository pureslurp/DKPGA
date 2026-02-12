from utils import TOURNAMENT_LIST_2026
from backtest.pga_dk_scoring import dk_points_df
from pga_v5 import fix_names
import pandas as pd

tournament = "WM_Phoenix_Open"

# Get results data
results_path = f'past_results/2026/dk_points_id_{TOURNAMENT_LIST_2026[tournament]["ID"]}.csv'
try:
    results_df = pd.read_csv(results_path)
    results_df = results_df[['Name', 'DK Score']]
except FileNotFoundError:
    print(f"Results not found for {tournament}, generating them...")
    try:
        dk_points_df(TOURNAMENT_LIST_2026[tournament]["ID"])
        results_df = pd.read_csv(results_path)
        results_df = results_df[['Name', 'DK Score']]
    except Exception as e:
        print(f"Error generating results file: {e}")
        raise FileNotFoundError(f"Could not generate or load results file: {results_path}")

# Clean names in results
results_df['Name'] = results_df['Name'].apply(fix_names)

# Read lineups file
lineups_df = pd.read_csv(f'2026/{tournament}/dk_lineups_optimized.csv')

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
lineups_df.to_csv(f'2026/{tournament}/dk_lineups_optimized_results.csv', index=False)


