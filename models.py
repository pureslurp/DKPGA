from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Optional
import pandas as pd
import numpy as np
from typing import List

from utils import fix_names

@dataclass
class StrokesGained:
    """Class to hold strokes gained statistics"""
    off_tee: float = 0.0
    approach: float = 0.0
    around_green: float = 0.0
    putting: float = 0.0

    @property
    def tee_to_green(self) -> float:
        """Calculate strokes gained tee to green"""
        return self.off_tee + self.approach + self.around_green
    
    @property
    def total(self) -> float:
        """Calculate total strokes gained"""
        return self.off_tee + self.approach + self.around_green + self.putting


class Golfer:
    def __init__(self, golfer: pd.DataFrame):
        # Original DraftKings attributes
        try:
            self.name = golfer["Name + ID"].iloc[0]
            self.salary = golfer["Salary"].iloc[0]
        except:
            try:
                self.name = golfer["Name + ID"]
                self.salary = golfer["Salary"]
            except:
                raise Exception("Unable to assign ", golfer)
        
        # Try to get total from dk_final.csv, default to 0 if not found
        try:
            self._total = float(golfer["Total"].iloc[0] if isinstance(golfer, pd.DataFrame) else golfer["Total"])
        except (KeyError, AttributeError, ValueError):
            self._total = 0.0

        # Try to get odds total, default to 0 if not found
        try:
            self._odds_total = float(golfer["Odds Total"].iloc[0] if isinstance(golfer, pd.DataFrame) else golfer["Odds Total"])
        except (KeyError, AttributeError, ValueError):
            self._odds_total = 0.0
                
        # New statistics attributes
        self.stats = {
            'current': self._initialize_stats(),
            'historical': {}  # Will hold stats by date
        }
        self.fit_score = None
        

        
    @property
    def total(self) -> float:
        """Get total projected score"""
        return self._total
        
    @total.setter
    def total(self, value: float):
        """Set total projected score"""
        self._total = value
        
    @property
    def value(self) -> float:
        """Calculate value (total points per salary)"""
        if self.salary == 0:
            return 0.0
        return self.total / self.salary * 1000
        
    @property
    def get_clean_name(self) -> str:
        """Returns cleaned name without DK ID and standardized format"""
        # Remove DK ID if present and clean the name
        return fix_names(self.name.split('(')[0].strip())
    
    def get_stats_summary(self) -> Dict:
        """Returns a clean summary of golfer's current stats"""
        stats = self.stats['current']
        sg = stats['strokes_gained']
        
        return {
            'name': self.get_clean_name,
            'driving_distance': stats['driving_distance'],
            'driving_accuracy': stats['driving_accuracy'],
            'gir': stats['gir'],
            'scrambling_sand': stats['scrambling_sand'],
            'strokes_gained': {
                'off_the_tee': sg.off_tee,
                'approach': sg.approach,
                'around_green': sg.around_green,
                'putting': sg.putting,
                'total': sg.total
            },
            'last_updated': stats['last_updated']
        }

    def print_stats(self):
        """Prints a formatted summary of golfer's current stats"""
        stats = self.get_stats_summary()
        print(f"\nStats for {stats['name']}:")
        print(f"Driving Distance: {stats['driving_distance']:.1f}")
        print(f"Driving Accuracy: {stats['driving_accuracy']:.1%}")
        print(f"GIR: {stats['gir']:.1%}")
        print(f"Sand Save: {stats['scrambling_sand']:.1%}")
        print("\nStrokes Gained:")
        for key, value in stats['strokes_gained'].items():
            print(f"  {key.replace('_', ' ').title()}: {value:.3f}")
        
    def _initialize_stats(self) -> Dict:
        """Initialize the stats dictionary with default values"""
        return {
            'strokes_gained': StrokesGained(),
            'driving_distance': 0.0,
            'driving_accuracy': 0.0,
            'gir': 0.0,
            'scrambling_sand': 0.0,
            'scoring_average': 0.0,
            'last_updated': None
        }

    def update_stats(self, 
                    strokes_gained: Optional[StrokesGained] = None,
                    driving_distance: Optional[float] = None,
                    driving_accuracy: Optional[float] = None,
                    gir: Optional[float] = None,
                    scrambling_sand: Optional[float] = None,
                    scoring_average: Optional[float] = None,
                    date: Optional[datetime] = None) -> None:
        """
        Update golfer statistics. If date is provided, stores in historical data.
        Otherwise updates current stats.
        """
        stats_dict = self.stats['historical'].setdefault(date.strftime('%Y-%m-%d'), 
                                                        self._initialize_stats()) if date else self.stats['current']
        
        if strokes_gained:
            stats_dict['strokes_gained'] = strokes_gained
        if driving_distance is not None:
            stats_dict['driving_distance'] = driving_distance
        if driving_accuracy is not None:
            stats_dict['driving_accuracy'] = driving_accuracy / 100  # Convert to decimal
        if gir is not None:
            stats_dict['gir'] = gir / 100
        if scrambling_sand is not None:
            stats_dict['scrambling_sand'] = scrambling_sand / 100
        if scoring_average is not None:
            stats_dict['scoring_average'] = scoring_average
        
        stats_dict['last_updated'] = datetime.now()

    def get_stats_trend(self, stat_name: str, weeks: int = 12) -> List[float]:
        """Get trend data for a specific statistic over the last n weeks"""
        if not self.stats['historical']:
            return []

        dates = sorted(self.stats['historical'].keys())[-weeks:]
        
        if '.' in stat_name:
            main_stat, sub_stat = stat_name.split('.')
            return [self.stats['historical'][date][main_stat].__dict__[sub_stat] 
                   for date in dates]
        
        return [self.stats['historical'][date][stat_name] for date in dates]

    def get_stats_average(self, stat_name: str, weeks: int = 12) -> float:
        """Calculate average for a specific statistic over the last n weeks"""
        trend_data = self.get_stats_trend(stat_name, weeks)
        return np.mean(trend_data) if trend_data else 0.0

    def get_current_form(self) -> float:
        """Calculate golfer's current form based on recent performance"""
        sg_trend = self.get_stats_trend('strokes_gained.total', weeks=4)
        if not sg_trend:
            return 0.0
        
        weights = np.array([0.1, 0.2, 0.3, 0.4])
        weights = weights[-len(sg_trend):]
        weights = weights / weights.sum()
        
        return max(0.0, min(1.0, np.average(sg_trend, weights=weights) / 2.0 + 0.5))

    def adjust_total_for_stats(self) -> float:
        """
        Adjust the total projected points based on current statistics.
        This method can be customized based on how you want to incorporate
        stats into your projections.
        """
        base_total = self.total
        
        # Example adjustment based on current form and recent strokes gained
        form_factor = self.get_current_form()
        recent_sg = self.get_stats_average('strokes_gained.total', weeks=4)
        
        # Simple adjustment formula - can be modified based on your needs
        adjustment = (form_factor - 0.5) * 5  # +/- 2.5 points based on form
        adjustment += recent_sg  # Add recent strokes gained directly
        
        return base_total + adjustment

    def __str__(self):
        return self.name
    
    def __repr__(self):
        return f"{self.name}"
    
    def __eq__(self, other):
        return self.name == other.name

    def set_fit_score(self, score: float) -> None:
        """Set the course fit score for this golfer"""
        self.fit_score = score
        
    def set_odds_total(self, total: float) -> None:
        """Set the odds total score for this golfer"""
        self._odds_total = total
        
    def calculate_total(self, odds_weight: float, fit_weight: float) -> None:
        """Calculate total score using provided weights"""
        if self.odds_total is not None and self.fit_score is not None:
            self._total = (
                odds_weight * self._odds_total +
                fit_weight * self.fit_score
            )
        else:
            self._total = 0.0

