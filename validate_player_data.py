import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt
import seaborn as sns

class PlayerDataValidator:
    def __init__(self, file_path: str):
        """Initialize validator with path to player_data.csv"""
        self.df = pd.read_csv(file_path)
        self.original_file = file_path
        self.validation_results: List[Dict] = []
        
    def check_column_outliers(self, column: str, z_score_threshold: float = 3.0) -> List[Dict]:
        """
        Check for outliers in a column using z-score method
        Returns list of outlier records with their values and z-scores
        """
        if self.df[column].dtype in ['object', 'string']:
            return []
            
        z_scores = np.abs((self.df[column] - self.df[column].mean()) / self.df[column].std())
        outliers = self.df[z_scores > z_score_threshold]
        
        return [
            {
                'name': row['Name'],
                'column': column,
                'value': row[column],
                'z_score': z_score,
                'column_mean': self.df[column].mean(),
                'column_std': self.df[column].std()
            }
            for (_, row), z_score in zip(outliers.iterrows(), z_scores[z_scores > z_score_threshold])
        ]

    def check_zero_values(self) -> List[Dict]:
        """
        Check for suspicious zero values where other related columns have data
        """
        zero_issues = []
        
        # Define related column groups
        column_groups = {
            'odds': ['Odds Total', 'Normalized Odds'],
            'fit': ['Fit Score', 'Normalized Fit'],
            'history': ['History Score', 'Normalized History'],
            'form': ['Form Score', 'Normalized Form']
        }
        
        for group_name, columns in column_groups.items():
            base_col, norm_col = columns
            suspicious = self.df[
                ((self.df[base_col] == 0) & (self.df[norm_col] != 0)) |
                ((self.df[base_col] != 0) & (self.df[norm_col] == 0))
            ]
            
            for _, row in suspicious.iterrows():
                zero_issues.append({
                    'name': row['Name'],
                    'group': group_name,
                    'base_value': row[base_col],
                    'normalized_value': row[norm_col]
                })
                
        return zero_issues

    def validate_normalization(self) -> List[Dict]:
        """
        Check if normalized columns are properly scaled between 0 and 1
        """
        normalized_columns = [
            'Normalized Odds',
            'Normalized Fit',
            'Normalized History',
            'Normalized Form'
        ]
        
        normalization_issues = []
        
        for col in normalized_columns:
            min_val = self.df[col].min()
            max_val = self.df[col].max()
            
            if min_val < 0 or max_val > 1:
                normalization_issues.append({
                    'column': col,
                    'min_value': min_val,
                    'max_value': max_val,
                    'affected_players': self.df[
                        (self.df[col] < 0) | (self.df[col] > 1)
                    ]['Name'].tolist()
                })
                
        return normalization_issues

    def plot_distributions(self, output_dir: str):
        """
        Create distribution plots for all numeric columns
        """
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            plt.figure(figsize=(10, 6))
            sns.histplot(data=self.df, x=col, kde=True)
            plt.title(f'Distribution of {col}')
            plt.savefig(f'{output_dir}/{col}_distribution.png')
            plt.close()

    def run_validation(self) -> Tuple[bool, str]:
        """
        Run all validation checks and return results
        """
        all_valid = True
        report = []
        
        # Check outliers
        for col in self.df.select_dtypes(include=[np.number]).columns:
            outliers = self.check_column_outliers(col)
            if outliers:
                all_valid = False
                report.append(f"\nOutliers found in {col}:")
                for outlier in outliers:
                    report.append(
                        f"  - {outlier['name']}: value={outlier['value']:.2f}, "
                        f"z-score={outlier['z_score']:.2f} "
                        f"(mean={outlier['column_mean']:.2f}, std={outlier['column_std']:.2f})"
                    )

        # Check zero values
        zero_issues = self.check_zero_values()
        if zero_issues:
            all_valid = False
            report.append("\nSuspicious zero values found:")
            for issue in zero_issues:
                report.append(
                    f"  - {issue['name']} ({issue['group']}): "
                    f"base={issue['base_value']:.2f}, normalized={issue['normalized_value']:.2f}"
                )

        # Check normalization
        norm_issues = self.validate_normalization()
        if norm_issues:
            all_valid = False
            report.append("\nNormalization issues found:")
            for issue in norm_issues:
                report.append(
                    f"  - {issue['column']}: range [{issue['min_value']:.2f}, {issue['max_value']:.2f}]"
                )
                if issue['affected_players']:
                    report.append(f"    Affected players: {', '.join(issue['affected_players'])}")

        return all_valid, "\n".join(report)

def main(tourney: str):
    """
    Main function to validate player data for a tournament
    """
    file_path = f"2026/{tourney}/player_data.csv"
    output_dir = f"2026/{tourney}/validation"
    
    # Create output directory if it doesn't exist
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    validator = PlayerDataValidator(file_path)
    
    # Run validation
    is_valid, report = validator.run_validation()
    
    # Generate plots
    validator.plot_distributions(output_dir)
    
    # Save report
    report_path = f"{output_dir}/validation_report.txt"
    with open(report_path, 'w') as f:
        f.write(report)
    
    print(f"Validation {'passed' if is_valid else 'failed'}")
    print(f"Full report saved to: {report_path}")
    print(f"Distribution plots saved to: {output_dir}")
    
    return is_valid

if __name__ == "__main__":
    from pga_v5 import TOURNEY
    main(TOURNEY) 