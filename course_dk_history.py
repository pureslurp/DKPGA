import pandas as pd
import os
from datetime import datetime
from utils import TOURNAMENT_LIST_2024
from Legacy.pga_v4 import fix_names
import numpy as np

'''
Script that creates a player course history dataframe from the 2024 tournaments.

Output:
- golfers/player_course_history.csv: the player course history dataframe
'''

def get_player_salary_stats(player_name):
    """Get salary statistics for a player across all tournaments"""
    salary_stats = []
    
    # Look through all tournament folders in 2024
    for tournament in TOURNAMENT_LIST_2024.keys():
        dk_final_path = f'2024/{tournament}/dk_final.csv'
        if os.path.exists(dk_final_path):
            df = pd.read_csv(dk_final_path)
            if 'Name' in df.columns and 'Salary' in df.columns:
                player_data = df[df['Name'].apply(fix_names) == player_name]
                if not player_data.empty:
                    salary_stats.append(player_data['Salary'].iloc[0])
    
    if salary_stats:
        return {
            'min_salary': min(salary_stats),
            'max_salary': max(salary_stats),
            'avg_salary': np.mean(salary_stats),
            'salary_volatility': np.std(salary_stats),
            'num_tournaments_priced': len(salary_stats)
        }
    return {
        'min_salary': np.nan,
        'max_salary': np.nan,
        'avg_salary': np.nan,
        'salary_volatility': np.nan,
        'num_tournaments_priced': 0
    }

def create_player_course_history():
    # Initialize empty list to store all results
    all_results = []
    
    # Read each past results file
    results_dir = 'past_results/2024'
    for filename in os.listdir(results_dir):
        if filename.endswith('.csv'):
            # Extract tournament ID from filename
            tournament_id = int(filename.split('_')[-1].replace('.csv', ''))
            
            # Find corresponding tournament and course
            tournament_info = None
            for tournament, info in TOURNAMENT_LIST_2024.items():
                if info['ID'] == tournament_id:
                    tournament_info = info
                    tournament_name = tournament
                    break
            
            if tournament_info:
                # Read the results file
                df = pd.read_csv(os.path.join(results_dir, filename))
                
                # Clean player names
                df['Name'] = df['Name'].apply(fix_names)
                
                # Add tournament and course info
                df['Tournament'] = tournament_name
                df['Course'] = tournament_info['Course']
                df['Course_Date'] = f"{tournament_info['Course']}_{datetime.now().year}"
                
                # Keep only relevant columns
                df = df[['Name', 'Course', 'Course_Date', 'DK Score', 'Tournament']]
                
                all_results.append(df)
    
    # Combine all results
    combined_results = pd.concat(all_results, ignore_index=True)
    
    # Calculate player statistics
    player_stats = combined_results.groupby('Name').agg({
        'DK Score': ['count', 'mean', 'std', 'min', 'max', 
                    lambda x: np.percentile(x, 25),  # Q1
                    lambda x: np.percentile(x, 75)], # Q3
        'Tournament': 'nunique'
    }).reset_index()
    
    # Rename columns
    player_stats.columns = ['Name', 'total_rounds', 'avg_score', 'score_volatility', 
                          'min_score', 'max_score', 'score_q1', 'score_q3', 
                          'unique_tournaments']
    
    # Calculate scoring consistency metrics
    player_stats['score_iqr'] = player_stats['score_q3'] - player_stats['score_q1']
    player_stats['score_range'] = player_stats['max_score'] - player_stats['min_score']
    
    # Add salary statistics
    salary_stats = []
    for player in player_stats['Name']:
        stats = get_player_salary_stats(player)
        salary_stats.append(stats)
    
    salary_df = pd.DataFrame(salary_stats)
    player_stats = pd.concat([player_stats, salary_df], axis=1)
    
    # Create course history pivot table
    course_history = combined_results.pivot_table(
        index='Name',
        columns='Course_Date',
        values='DK Score',
        aggfunc='last'
    )
    
    # Merge player stats with course history
    final_df = pd.merge(
        player_stats,
        course_history,
        left_on='Name',
        right_index=True
    )
    
    # Calculate additional metrics
    final_df['made_cut_rate'] = final_df.apply(
        lambda x: sum(x[col] >= 0 for col in course_history.columns) / len(course_history.columns)
        if len(course_history.columns) > 0 else 0, axis=1
    )
    
    final_df['salary_per_point'] = final_df['avg_salary'] / final_df['avg_score']
    final_df['value_volatility'] = final_df['salary_volatility'] / final_df['score_volatility']
    
    # Add performance trend (linear regression slope of recent scores)
    def calculate_trend(row):
        scores = [row[col] for col in course_history.columns if pd.notna(row[col])]
        if len(scores) > 1:
            return np.polyfit(range(len(scores)), scores, 1)[0]
        return 0
    
    final_df['performance_trend'] = final_df.apply(calculate_trend, axis=1)
    
    # Sort by average score (descending) and reset index
    final_df.sort_values('avg_score', ascending=False, inplace=True)
    
    # Save to CSV
    final_df.to_csv('player_course_history.csv', index=False)
    print(f"Successfully created player_course_history.csv with {len(final_df)} players")
    
    # Print some summary statistics
    print("\nDataset Summary:")
    print(f"Total unique courses: {len(course_history.columns)}")
    print(f"Total players: {len(final_df)}")
    print(f"Average tournaments per player: {final_df['unique_tournaments'].mean():.1f}")
    print(f"Average score: {final_df['avg_score'].mean():.1f}")
    
    return final_df

if __name__ == "__main__":
    create_player_course_history()