#!/usr/bin/env python3
"""
Batch script to prepare tournament data for pga_v5.py

This script runs the following steps in sequence:
1. dg_fit.py - Scrapes DataGolf course fit data
2. pga_stats.py - Fetches PGA Tour season stats
3. current_form.py - Scrapes current form data
4. tournament_history.py - Scrapes tournament history data

Usage:
    python prepare_tournament.py <tournament_name>
    
Example:
    python prepare_tournament.py Farmers_Insurance_Open
"""

import sys
import os
import time
from utils import TOURNAMENT_LIST_2026

# Import functions from each script
from dg_fit import main as run_dg_fit
from pga_stats import create_pga_stats
from current_form import get_current_form
from tournament_history import get_tournament_history


def run_dg_fit_step(tourney: str):
    """Step 1: Run DataGolf course fit scraper"""
    print("\n" + "="*60)
    print(f"Step 1/4: Running DataGolf course fit scraper for {tourney}")
    print("="*60)
    try:
        run_dg_fit(tourney)
        print("✓ DataGolf course fit data saved successfully")
        return True
    except Exception as e:
        print(f"✗ Error running dg_fit.py: {e}")
        return False


def run_pga_stats_step(tourney: str):
    """Step 2: Create PGA stats file"""
    print("\n" + "="*60)
    print(f"Step 2/4: Fetching PGA Tour stats for {tourney}")
    print("="*60)
    try:
        create_pga_stats(tourney)
        print("✓ PGA stats file created successfully")
        return True
    except Exception as e:
        print(f"✗ Error running pga_stats.py: {e}")
        return False


def run_current_form_step(tourney: str):
    """Step 3: Scrape current form data"""
    print("\n" + "="*60)
    print(f"Step 3/4: Scraping current form data for {tourney}")
    print("="*60)
    try:
        # Check if tournament exists in TOURNAMENT_LIST_2026
        if tourney not in TOURNAMENT_LIST_2026:
            print(f"✗ Tournament '{tourney}' not found in TOURNAMENT_LIST_2026")
            return False
        
        if 'pga-url' not in TOURNAMENT_LIST_2026[tourney]:
            print(f"✗ Tournament '{tourney}' missing 'pga-url' in TOURNAMENT_LIST_2026")
            print("  Please add the pga-url to utils.py first")
            return False
        
        # Construct URL
        pga_url = TOURNAMENT_LIST_2026[tourney]["pga-url"]
        url = f"https://www.pgatour.com/tournaments/2026/{pga_url}/field/current-form"
        
        # Scrape data
        df = get_current_form(url)
        
        if df is not None:
            # Ensure directory exists
            os.makedirs(f'2026/{tourney}', exist_ok=True)
            # Save to CSV
            df.to_csv(f'2026/{tourney}/current_form.csv', index=False)
            print("✓ Current form data saved successfully")
            return True
        else:
            print("✗ Failed to scrape current form data")
            return False
    except Exception as e:
        print(f"✗ Error running current_form.py: {e}")
        return False


def run_tournament_history_step(tourney: str):
    """Step 4: Scrape tournament history data"""
    print("\n" + "="*60)
    print(f"Step 4/4: Scraping tournament history data for {tourney}")
    print("="*60)
    try:
        # Check if tournament exists in TOURNAMENT_LIST_2026
        if tourney not in TOURNAMENT_LIST_2026:
            print(f"✗ Tournament '{tourney}' not found in TOURNAMENT_LIST_2026")
            return False
        
        if 'pga-url' not in TOURNAMENT_LIST_2026[tourney]:
            print(f"✗ Tournament '{tourney}' missing 'pga-url' in TOURNAMENT_LIST_2026")
            print("  Please add the pga-url to utils.py first")
            return False
        
        # Construct URL
        pga_url = TOURNAMENT_LIST_2026[tourney]['pga-url']
        url = f"https://www.pgatour.com/tournaments/2026/{pga_url}/field/tournament-history"
        
        # Scrape data
        df = get_tournament_history(url)
        
        if df is not None:
            # Get tournament name from the scraped data
            tournament_name = df['tournament'].iloc[0] if 'tournament' in df.columns else tourney
            # Ensure directory exists
            os.makedirs(f'2026/{tournament_name}', exist_ok=True)
            # Save to CSV
            df.to_csv(f'2026/{tournament_name}/tournament_history.csv', index=False)
            print("✓ Tournament history data saved successfully")
            return True
        else:
            print("✗ Failed to scrape tournament history data")
            return False
    except Exception as e:
        print(f"✗ Error running tournament_history.py: {e}")
        return False


def main():
    """Main function to run all preparation steps"""
    if len(sys.argv) < 2:
        print("Usage: python prepare_tournament.py <tournament_name>")
        print("\nExample:")
        print("  python prepare_tournament.py Farmers_Insurance_Open")
        sys.exit(1)
    
    tourney = sys.argv[1]
    
    print("\n" + "="*60)
    print(f"Preparing tournament data for: {tourney}")
    print("="*60)
    
    # Verify tournament exists in TOURNAMENT_LIST_2026
    if tourney not in TOURNAMENT_LIST_2026:
        print(f"\n✗ Error: Tournament '{tourney}' not found in TOURNAMENT_LIST_2026")
        print("  Please add it to utils.py first")
        sys.exit(1)
    
    # Run all steps
    results = {}
    
    results['dg_fit'] = run_dg_fit_step(tourney)
    time.sleep(2)  # Brief pause between steps
    
    results['pga_stats'] = run_pga_stats_step(tourney)
    time.sleep(2)
    
    results['current_form'] = run_current_form_step(tourney)
    time.sleep(2)
    
    results['tournament_history'] = run_tournament_history_step(tourney)
    
    # Summary
    print("\n" + "="*60)
    print("PREPARATION SUMMARY")
    print("="*60)
    print(f"Tournament: {tourney}")
    print(f"\nSteps completed:")
    print(f"  1. DataGolf course fit:     {'✓' if results['dg_fit'] else '✗'}")
    print(f"  2. PGA Tour stats:          {'✓' if results['pga_stats'] else '✗'}")
    print(f"  3. Current form:            {'✓' if results['current_form'] else '✗'}")
    print(f"  4. Tournament history:      {'✓' if results['tournament_history'] else '✗'}")
    
    all_success = all(results.values())
    
    if all_success:
        print("\n✓ All steps completed successfully!")
        print(f"\nNext steps:")
        print(f"  1. Download DKSalaries.csv from DraftKings")
        print(f"  2. Run parse_odds.py to get odds data (if not done already)")
        print(f"  3. Run pga_v5.py to generate optimized lineups")
    else:
        print("\n⚠ Some steps failed. Please review the errors above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
