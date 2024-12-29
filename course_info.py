import pandas as pd
import re
from dataclasses import dataclass
from typing import Optional, Dict, Tuple, List
from utils import TOURNAMENT_LIST_2024

'''
Script that creates a course database and tournament course mapping from dg_course_table.csv

Output: 
- course_database.csv: Complete stats for all courses
- tournament_courses.csv: Tournament-specific course stats
'''

@dataclass
class CourseStats:
    # Basic Info
    name: str
    par: int
    yardage: int
    
    # Distance Stats
    adj_driving_distance: float
    adj_sd_distance: float
    
    # Accuracy Stats
    adj_driving_accuracy: float
    fw_width: float
    
    # Scoring Stats
    adj_score_to_par: float
    adj_par_3_score: float
    adj_par_4_score: float
    adj_par_5_score: float
    
    # Course Difficulty Metrics
    fw_diff: float
    rgh_diff: Optional[float]
    non_rgh_diff: float
    miss_fw_pen_frac: float
    
    # Strokes Gained Stats
    putt_sg: float
    arg_sg: float
    app_sg: float
    ott_sg: float
    
    # Distance-based Performance
    less_150_sg: float
    greater_150_sg: float
    
    # Lie-specific Performance
    arg_fairway_sg: float
    arg_rough_sg: Optional[float]
    arg_bunker_sg: float
    
    # Putting Performance
    less_5_ft_sg: float
    greater_5_less_15_sg: float
    greater_15_sg: float
    
    # Penalty Stats
    adj_penalties: float
    adj_ob: float

def get_course_name_variations(name: str) -> list:
    """Generate possible variations of course names"""
    name = name.lower()
    variations = [name]
    
    # Handle common replacements
    replacements = [
        ("and", "&"),
        ("&", "and"),
        ("golf club", "gc"),
        ("golf course", "gc"),
        ("country club", "cc"),
        ("golf and country club", "gcc"),
        ("golf & country club", "gcc"),
        ("tournament course", ""),
        ("stadium course", ""),
        ("championship course", ""),
        ("the course at", ""),
        ("resort and spa", "resort"),
        ("resort & spa", "resort")
    ]
    
    # Create variations with each replacement
    for old, new in replacements:
        if old in name:
            variations.append(name.replace(old, new).strip())
            
    # Remove special characters and extra spaces
    variations = [re.sub(r'[^\w\s&]', '', v).strip() for v in variations]
    variations = [re.sub(r'\s+', ' ', v).strip() for v in variations]
    
    # Add variations without common words
    skip_words = ['golf', 'club', 'course', 'resort', 'spa', 'the']
    base_words = name.split()
    filtered = ' '.join(word for word in base_words if word.lower() not in skip_words)
    variations.append(filtered)
    
    return list(set(variations))  # Remove duplicates

def clean_course_name(name: str) -> str:
    """Clean course name to match between different data sources"""
    # Convert to lowercase
    name = name.lower()
    
    # Handle special HTML entities
    replacements = {
        "amp;": "",  # HTML ampersand
        "&": "and",  # Regular ampersand
        "'s": "s",   # Possessives
        "'": "",     # Single quotes
        "_": " ",    # Underscores
        "(": " ",    # Opening parentheses
        ")": " ",    # Closing parentheses
        ".": " ",    # Periods
        ",": " ",    # Commas
        "-": " ",    # Hyphens
        "  ": " "    # Double spaces
    }
    
    for old, new in replacements.items():
        name = name.replace(old, new)
    
    # Remove any remaining special characters
    name = re.sub(r'[^a-z0-9\s]', '', name)
    
    # Remove extra whitespace
    name = ' '.join(name.split())
    
    return name

def get_course_name_variations(name: str) -> list:
    """Generate possible variations of course names"""
    # Clean the name first
    name = clean_course_name(name)
    variations = [name]
    
    # Handle common replacements
    replacements = [
        ("and", "&"),
        ("golf club", "gc"),
        ("golf course", "gc"),
        ("country club", "cc"),
        ("golf and country club", "gcc"),
        ("tournament course", ""),
        ("stadium course", ""),
        ("championship course", ""),
        ("the course at", ""),
        ("resort and spa", "resort"),
        ("golf club and resort", "gc"),
        ("number", "no"),
        ("course no", "no")
    ]
    
    # Create variations with each replacement
    for old, new in replacements:
        if old in name:
            variations.append(name.replace(old, new).strip())
            
    # Add variations without common words
    skip_words = ['golf', 'club', 'course', 'resort', 'spa', 'the', 'at', 'and']
    base_words = name.split()
    filtered = ' '.join(word for word in base_words if word.lower() not in skip_words)
    if filtered:
        variations.append(filtered)
    
    return list(set(variations))  # Remove duplicates

def create_course_mapping(df: pd.DataFrame) -> Tuple[Dict[str, str], Dict[str, str]]:
    """Create mapping between DataGolf and TOURNAMENT_LIST_2024 course names"""
    mapping = {}
    reverse_mapping = {}
    
    # Clean DataGolf course names
    df['clean_course'] = df['course'].apply(clean_course_name)
    
    # Custom mapping for specific courses
    custom_mapping = {
        "Kapalua_Resort_(Plantation_Course)": "plantation course at kapalua",
        "Arnold_Palmer's_Bay_Hill_Club_&_Lodge": "arnold palmers bay hill club and lodge",
        "Pinehurst_No._2": "pinehurst resort and country club course no 2",
        "Hamilton_Golf_&_Country_Club": "hamilton golf and country club",
        "Dunes_Golf_&_Beach_Club": "the dunes golf and beach club",
        "Torrey_Pines_(North_Course)": "torrey pines golf course south course",  # Note: mapping to South
        "Spyglass_Hill_GC": "pebble beach golf links",  # Note: mapping to Pebble
    }
    
    # Update custom mappings first
    for tournament_name, clean_name in custom_mapping.items():
        matching_courses = df[df['clean_course'] == clean_name]
        if not matching_courses.empty:
            mapping[tournament_name] = matching_courses.iloc[0]['course']
            reverse_mapping[matching_courses.iloc[0]['course']] = tournament_name
    
    # Create variations for all DataGolf courses
    datagolf_variations = {}
    for _, row in df.iterrows():
        course = row['course']
        clean_course = row['clean_course']
        variations = get_course_name_variations(clean_course)
        for variation in variations:
            datagolf_variations[variation] = course
    
    # Try to match remaining courses
    for tournament, info in TOURNAMENT_LIST_2024.items():
        if info['Course'] not in mapping:
            course_name = clean_course_name(info['Course'])
            found_match = False
            
            # Try direct match first
            if course_name in datagolf_variations:
                mapping[info['Course']] = datagolf_variations[course_name]
                reverse_mapping[datagolf_variations[course_name]] = info['Course']
                found_match = True
            else:
                # Try variations
                for variation in get_course_name_variations(course_name):
                    if variation in datagolf_variations:
                        mapping[info['Course']] = datagolf_variations[variation]
                        reverse_mapping[datagolf_variations[variation]] = info['Course']
                        found_match = True
                        break
            
            if not found_match:
                print(f"\nNo match found for {info['Course']}")
                print(f"Cleaned name: {course_name}")
                print(f"Tried variations: {get_course_name_variations(course_name)}")
                
                # Print potential close matches
                close_matches = []
                for dgolf_course in df['course'].unique():
                    clean_dgolf = clean_course_name(dgolf_course)
                    if any(word in clean_dgolf for word in course_name.split()):
                        close_matches.append(dgolf_course)
                if close_matches:
                    print(f"Potential close matches in DataGolf data: {close_matches}")
    
    return mapping, reverse_mapping

def create_course_database():
    """Create and populate course database"""
    # Read DataGolf stats
    df = pd.read_csv('courses/dg_course_table.csv')
    
    # Create database
    course_database = {}
    
    # Process each course in the DataGolf data
    for _, row in df.iterrows():
        course_stats = CourseStats(
            name=row['course'],
            par=int(row['par']),
            yardage=int(row['yardage']),
            adj_driving_distance=row['adj_driving_distance'],
            adj_sd_distance=row['adj_sd_distance'],
            adj_driving_accuracy=row['adj_driving_accuracy'],
            fw_width=row['fw_width'],
            adj_score_to_par=row['adj_score_to_par'],
            adj_par_3_score=row['adj_par_3_score'],
            adj_par_4_score=row['adj_par_4_score'],
            adj_par_5_score=row['adj_par_5_score'],
            fw_diff=row['fw_diff'],
            rgh_diff=row['rgh_diff'] if pd.notna(row['rgh_diff']) else None,
            non_rgh_diff=row['non_rgh_diff'],
            miss_fw_pen_frac=row['miss_fw_pen_frac'],
            putt_sg=row['putt_sg'],
            arg_sg=row['arg_sg'],
            app_sg=row['app_sg'],
            ott_sg=row['ott_sg'],
            less_150_sg=row['less_150_sg'],
            greater_150_sg=row['greater_150_sg'],
            arg_fairway_sg=row['arg_fairway_sg'],
            arg_rough_sg=row['arg_rough_sg'] if pd.notna(row['arg_rough_sg']) else None,
            arg_bunker_sg=row['arg_bunker_sg'],
            less_5_ft_sg=row['less_5_ft_sg'],
            greater_5_less_15_sg=row['greater_5_less_15_sg'],
            greater_15_sg=row['greater_15_sg'],
            adj_penalties=row['adj_penalties'],
            adj_ob=row['adj_ob']
        )
        course_database[row['course']] = course_stats
    
    # Create course name mapping
    course_mapping, reverse_mapping = create_course_mapping(df)
    
    # Map current tournament courses
    tournament_courses = {}
    for tournament, info in TOURNAMENT_LIST_2024.items():
        if info['Course'] in course_mapping:
            datagolf_name = course_mapping[info['Course']]
            tournament_courses[tournament] = course_database[datagolf_name]
        else:
            print(f"Warning: No data for {tournament} course: {info['Course']}")
            
    # Print mapping summary
    print("\nCourse Mapping Summary:")
    print(f"Successfully mapped: {len(tournament_courses)}/{len(TOURNAMENT_LIST_2024)} courses")
    print("\nUnmapped courses:")
    for tournament, info in TOURNAMENT_LIST_2024.items():
        if tournament not in tournament_courses:
            print(f"{tournament}: {info['Course']}")
    
    # Calculate some interesting correlations
    stats_df = pd.DataFrame([{
        'course': name,
        'score_to_par': stats.adj_score_to_par,
        'driving_distance': stats.adj_driving_distance,
        'driving_accuracy': stats.adj_driving_accuracy,
        'fw_width': stats.fw_width,
        'total_sg': stats.putt_sg + stats.arg_sg + stats.app_sg + stats.ott_sg
    } for name, stats in course_database.items()])
    
    print("\nKey Correlations:")
    numeric_columns = ['score_to_par', 'driving_distance', 'driving_accuracy', 'fw_width', 'total_sg']
    correlations = stats_df[numeric_columns].corr()['score_to_par'].sort_values(ascending=False)
    print(correlations)
    
    return course_database, tournament_courses

def main():
    course_db, tournament_courses = create_course_database()
    
    # Save full course database
    course_records = []
    for course_name, stats in course_db.items():
        course_records.append({
            'name': course_name,
            **{k: v for k, v in stats.__dict__.items() if k != 'name'}
        })
    
    pd.DataFrame(course_records).to_csv('courses/course_database.csv', index=False)
    
    # Save tournament course mapping with all stats
    tournament_records = []
    for tournament, stats in tournament_courses.items():
        # Use the course name from TOURNAMENT_LIST_2024 instead of stats.name
        tournament_records.append({
            'tournament': tournament,
            'course': TOURNAMENT_LIST_2024[tournament]['Course'],  # Use this instead of stats.name
            **{k: v for k, v in stats.__dict__.items() if k != 'name'}
        })
    
    tournament_df = pd.DataFrame(tournament_records)
    
    # Reorder columns to match course_database.csv format
    course_db_df = pd.DataFrame(course_records)
    tournament_df = tournament_df[['tournament', 'course'] + list(course_db_df.columns[1:])]
    
    tournament_df.to_csv('courses/tournament_courses.csv', index=False)
    
    print("\nDatabase Summary:")
    print(f"Total courses in database: {len(course_records)}")
    print(f"Mapped tournaments: {len(tournament_records)}")
    print("\nSaved files:")
    print("- course_database.csv: Complete stats for all courses")
    print("- tournament_courses.csv: Tournament-specific course stats")

def format_tournament_name(name: str) -> str:
    """Convert tournament name to match TOURNAMENT_LIST_2024 format"""
    # Handle special cases first
    special_cases = {
        "THE PLAYERS Championship": "THE_PLAYERS_Championship",
        "RBC Canadian Open": "RBC_Canadian_Open",
        "the Memorial Tournament": "the_Memorial_Tournament_pres._by_Workday"
        # Add more special cases as needed
    }
    
    for old, new in special_cases.items():
        if name.replace('_', ' ').lower() == old.lower():
            return new
            
    # For other cases, ensure format matches TOURNAMENT_LIST_2024
    if name in TOURNAMENT_LIST_2024:
        return name
        
    # Try to find match ignoring case and underscores
    for tournament in TOURNAMENT_LIST_2024:
        if name.replace('_', ' ').lower() == tournament.replace('_', ' ').lower():
            return tournament
            
    return name

def load_courses_from_csv(filepath: str) -> Dict[str, CourseStats]:
    """
    Load course data from CSV and return a dictionary of CourseStats objects.
    For tournament_courses.csv, key will be tournament name.
    For course_database.csv, key will be course name.
    """
    df = pd.read_csv(filepath)
    courses = {}
    
    # Determine if this is tournament data or course database
    is_tournament_data = 'tournament' in df.columns
    
    for _, row in df.iterrows():
        # Create CourseStats object from row data
        course_stats = CourseStats(
            name=row['course'] if is_tournament_data else row['name'],
            par=int(row['par']),
            yardage=int(row['yardage']),
            adj_driving_distance=row['adj_driving_distance'],
            adj_sd_distance=row['adj_sd_distance'],
            adj_driving_accuracy=row['adj_driving_accuracy'],
            fw_width=row['fw_width'],
            adj_score_to_par=row['adj_score_to_par'],
            adj_par_3_score=row['adj_par_3_score'],
            adj_par_4_score=row['adj_par_4_score'],
            adj_par_5_score=row['adj_par_5_score'],
            fw_diff=row['fw_diff'],
            rgh_diff=row['rgh_diff'] if pd.notna(row['rgh_diff']) else None,
            non_rgh_diff=row['non_rgh_diff'],
            miss_fw_pen_frac=row['miss_fw_pen_frac'],
            putt_sg=row['putt_sg'],
            arg_sg=row['arg_sg'],
            app_sg=row['app_sg'],
            ott_sg=row['ott_sg'],
            less_150_sg=row['less_150_sg'],
            greater_150_sg=row['greater_150_sg'],
            arg_fairway_sg=row['arg_fairway_sg'],
            arg_rough_sg=row['arg_rough_sg'] if pd.notna(row['arg_rough_sg']) else None,
            arg_bunker_sg=row['arg_bunker_sg'],
            less_5_ft_sg=row['less_5_ft_sg'],
            greater_5_less_15_sg=row['greater_5_less_15_sg'],
            greater_15_sg=row['greater_15_sg'],
            adj_penalties=row['adj_penalties'],
            adj_ob=row['adj_ob']
        )
        
        # Use tournament name as key if tournament data, otherwise use course name
        key = row['tournament'] if is_tournament_data else row['name']

            
        courses[key] = course_stats
    
    return courses

def load_course_stats(tournament_name: str, filepath: str = 'courses/tournament_courses.csv') -> CourseStats:
    """
    Load course stats for a single tournament from CSV file.
    
    Args:
        tournament_name: Name of the tournament (e.g., "The_Sentry")
        filepath: Path to the tournament courses CSV file
    
    Returns:
        CourseStats object for the specified tournament
    
    Raises:
        ValueError: If tournament not found or name format doesn't match
    """
    
    # Load courses
    courses = load_courses_from_csv(filepath)
    
    # Try to find the course
    if tournament_name not in courses:
        raise ValueError(
            f"No data found for tournament: {tournament_name}\n"
            f"Available tournaments: {sorted(courses.keys())}"
        )
    
    return courses[tournament_name]

if __name__ == "__main__":
    main()