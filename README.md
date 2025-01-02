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
