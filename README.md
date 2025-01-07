# DraftKings PGA Tour Optimization Model

## Overview
A comprehensive model for optimizing DraftKings PGA Tour lineups using betting odds, historical data, course attributes, and player statistics.

## Data Sources & Features

### 1. Tournament History
- Sources tournament-specific performance data from PGATour.com
- Includes past 5 event finishes for each tournament
- Data stored in: `{tournament_name}/tournament_history.csv`

### 2. Course-Player Analysis
#### Course Attributes
- Reference data stored in: `dg_course_table.csv`
- Mapped against player performance metrics
- Weights optimized through backtesting against 2024 tournament results
- Optimized weights loaded from: `optimization_results/final_weights.csv`

#### Player Statistics
Data sourced from: `golfers/pga_stats_{date}.csv` or `golfers/current_form_{date}.csv`

Key Metrics:
- **Strokes Gained Categories**
  - Off the Tee
  - Approach
  - Around the Green
  - Putting
- **Additional Performance Metrics**
  - Driving Accuracy
  - Green in Regulation
  - Scrambling from Sand

### 3. Performance Weight Optimization
- Results stored in: `weight_optimization_results.csv`
- Optimizes for multiple finish positions:
  - Tournament Winner
  - Top 5 Finish
  - Top 10 Finish
  - Top 20 Finish

#### Optimized Weights
The following weights were derived from backtesting 2024 tournament results:

| Finish Position | Weight |
|----------------|--------|
| Winner         | 0.6    |
| Top 5          | 0.5    |
| Top 10         | 0.8    |
| Top 20         | 0.4    |

### 4. Lineup Optimization
Utilizes DKLineupOptimizer with customizable parameters:
- Exposure limits per player
- Lineup overlap restrictions
- Salary cap compliance
- Lineup size requirements
- Multiple lineup generation capability

## Output
- Generates optimized lineups exported to CSV format
- Includes all necessary lineup constraints and diversification rules

## Quick Start
- Run `parse_odds.py` to get the odds for each golfer for each stat, this will also create the `2025/{tournament_name}` folder and populate it with the odds.csv file. There needs to be odds available for a tournament or it will fail.
- Run `tournament_history.py` to get the tournament history for each golfer, this will create the `2025/{tournament_name}/tournament_history.csv` file. The url will need to be updated to the correct url for the tournament.
- Run `current_form.py` to get the current form for each golfer, this will create the `2025/{tournament_name}/current_form.csv` file. The url will need to be updated to the correct url for the tournament.
- Run `pga_course_fit.py` to get the course fit for each golfer, this will create the `2025/{tournament_name}/course_fit.csv` file. The url will need to be updated to the correct url for the tournament.
- Run `pga_stats.py` to get the pga stats for each golfer, this will create the `2025/{tournament_name}/pga_stats.csv` file. The url will need to be updated to the correct url for the tournament.
- Add the DKSalaries.csv file to the `2025/{tournament_name}/` folder, this file is from DraftKings.
- Run `pga_v5.py` with the tournament name to get the optimized lineups for each tournament, it will also create the player_data.csv and fit_details.csv files.
