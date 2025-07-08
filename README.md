# DraftKings PGA Tour Optimization Model

## Overview
A comprehensive model for optimizing DraftKings PGA Tour lineups using betting odds, historical data, course attributes, and player statistics. The main optimization engine is `pga_v5.py`, which combines multiple data sources to generate optimal DraftKings lineups.

## Quick Start

### Prerequisites
1. **Python 3.8+** with pip
2. **Firefox browser** (for Selenium web scraping)
3. **GeckoDriver** for Firefox (automatically managed by Selenium)

### Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd DKPGA
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify installation:**
   ```bash
   python -c "import pandas, pulp, selenium; print('Dependencies installed successfully!')"
   ```

## Main Script: `pga_v5.py`

The core optimization engine that generates DraftKings lineups. This script:

- **Combines multiple data sources** (odds, course fit, tournament history, player form)
- **Uses Integer Linear Programming** for optimal lineup generation
- **Applies exposure and overlap constraints** for lineup diversification
- **Generates multiple lineups** with customizable parameters

### Key Features

#### 1. Data Integration
- **Betting Odds**: Tournament winner, top 5, top 10, top 20 finishes
- **Course Fit Analysis**: Player performance on similar courses
- **Tournament History**: Past performance at the specific tournament
- **Current Form**: Recent player statistics and performance trends

#### 2. Optimization Engine
- **Salary Cap Compliance**: $50,000 DraftKings limit
- **Lineup Size**: 6 players per lineup
- **Exposure Limits**: Configurable player exposure (default 66%)
- **Overlap Restrictions**: Maximum 4 players between lineups
- **Salary Distribution**: Smart salary tier management

#### 3. Scoring System
The model calculates player scores using weighted components:

| Component | Default Weight | Description |
|-----------|---------------|-------------|
| Odds | 60% | Betting odds converted to projected points |
| Form | 40% | Recent player performance and statistics |
| Course Fit | 0% | Player performance on similar courses |
| History | 0% | Past tournament performance |

### Usage

#### Basic Usage
```python
from pga_v5 import main

# Run optimization for a tournament
main("John_Deere_Classic", num_lineups=20)
```

#### Advanced Usage
```python
# Custom weights and exclusions
weights = {
    'odds': {
        'winner': 0.35,
        'top5': 0.15,
        'top10': 0.20,
        'top20': 0.30
    },
    'form': {
        'current': 0.7,
        'long': 0.3
    },
    'components': {
        'odds': 0.6,
        'fit': 0.0,
        'history': 0.0,
        'form': 0.4
    }
}

exclude_golfers = ["Tiger Woods", "Rory McIlroy"]

main("John_Deere_Classic", 
     num_lineups=20, 
     weights=weights, 
     exclude_golfers=exclude_golfers)
```

### Required Data Files

Before running `pga_v5.py`, ensure these files exist in `2025/{tournament_name}/`:

| File | Description | Source |
|------|-------------|--------|
| `odds.csv` | Betting odds for all finish positions | `parse_odds.py` |
| `DKSalaries.csv` | DraftKings player salaries | DraftKings export |
| `course_fit.csv` | Course fit analysis | `pga_course_fit.py` |
| `tournament_history.csv` | Past tournament performance | `tournament_history.py` |
| `pga_stats.csv` | Player statistics | `pga_stats.py` |

### Data Collection Scripts

Run these scripts in order to collect required data:

1. **`parse_odds.py`** - Scrapes betting odds from scoresandodds.com
2. **`tournament_history.py`** - Gets tournament-specific performance data
3. **`current_form.py`** - Collects current player form data
4. **`pga_course_fit.py`** - Analyzes course fit for players
5. **`pga_stats.py`** - Gathers comprehensive player statistics

### Output Files

The script generates several output files:

| File | Description |
|------|-------------|
| `dk_lineups_optimized.csv` | Generated DraftKings lineups |
| `player_data.csv` | Detailed player analysis and scores |
| `fit_details.csv` | Course fit analysis details |

## Dependencies

### Core Dependencies
- **pandas** (≥2.0.0): Data manipulation and analysis
- **numpy** (≥1.24.0): Numerical computations
- **pulp** (≥2.7.0): Integer Linear Programming optimization
- **selenium** (≥4.11.0): Web scraping for data collection

### Optional Dependencies
- **streamlit** (≥1.31.0): Web dashboard interface
- **python-dotenv** (≥1.0.0): Environment variable management
- **openai** (≥1.0.0): AI-powered analysis features

### System Requirements
- **Firefox browser**: Required for Selenium web scraping
- **GeckoDriver**: Automatically managed by Selenium
- **Internet connection**: For real-time data collection

## Configuration

### Tournament Setup
1. Create tournament folder: `2025/{tournament_name}/`
2. Add DraftKings salaries file: `DKSalaries.csv`
3. Update tournament URLs in data collection scripts
4. Run data collection scripts in order

### Optimization Parameters
- **Salary Cap**: $50,000 (DraftKings standard)
- **Lineup Size**: 6 players
- **Minimum Salary**: $49,000 (configurable)
- **Exposure Limit**: 66% (configurable)
- **Overlap Limit**: 4 players between lineups

## Troubleshooting

### Common Issues

1. **Selenium/Firefox errors**:
   ```bash
   # Install Firefox if not present
   brew install firefox  # macOS
   sudo apt-get install firefox  # Ubuntu
   ```

2. **Missing data files**:
   - Ensure all required CSV files exist in tournament folder
   - Check file permissions and paths

3. **Optimization failures**:
   - Verify salary cap compliance
   - Check for sufficient player pool
   - Review exclusion lists

### Performance Tips
- Use `force_refresh=False` for faster subsequent runs
- Limit `num_lineups` for quicker optimization
- Pre-filter player pool for large tournaments

## Advanced Features

### Custom Weighting
Modify the scoring weights to emphasize different factors:

```python
weights = {
    'odds': {'winner': 0.4, 'top5': 0.2, 'top10': 0.3, 'top20': 0.1},
    'form': {'current': 0.8, 'long': 0.2},
    'components': {'odds': 0.5, 'fit': 0.2, 'history': 0.2, 'form': 0.1}
}
```

### Parallel Processing
The script uses parallel processing for score calculations to improve performance on large player pools.

### Caching
Data is cached to avoid redundant web scraping. Force refresh with `force_refresh=True` when needed.

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Verify all dependencies are installed correctly
3. Ensure data files are properly formatted
4. Review console output for error messages
