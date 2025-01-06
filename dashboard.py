import streamlit as st
import pandas as pd
import os
from course_info import load_course_stats
from pga_v5 import StrokesGained, main as run_pga_model, fix_names, CoursePlayerFit, Golfer
from typing import List

class DataManager:
    """Manages data and session state for the dashboard"""
    
    def __init__(self):
        # Initialize session state if needed
        if 'custom_weights' not in st.session_state:
            st.session_state.custom_weights = CoursePlayerFit.DEFAULT_WEIGHTS_DICT.copy()
        if 'player_data_adjustments' not in st.session_state:
            st.session_state.player_data_adjustments = {}

    def load_tournament_data(self, tournament: str) -> tuple:
        """Load and process tournament data with any user adjustments"""
        try:
            # Load base data
            player_data = pd.read_csv(f"2025/{tournament}/player_data.csv")
            lineups = pd.read_csv(f"2025/{tournament}/dk_lineups_optimized.csv")
            
            # Apply any stored adjustments
            if tournament in st.session_state.player_data_adjustments:
                adjustments = st.session_state.player_data_adjustments[tournament]
                for player_name, values in adjustments.items():
                    if player_name in player_data['Name'].values:
                        for column, value in values.items():
                            player_data.loc[player_data['Name'] == player_name, column] = value
            
            return player_data, lineups
        except FileNotFoundError:
            return None, None
    
    def update_player_adjustment(self, tournament: str, player_name: str, column: str, value: float):
        """Store a player data adjustment"""
        if tournament not in st.session_state.player_data_adjustments:
            st.session_state.player_data_adjustments[tournament] = {}
        if player_name not in st.session_state.player_data_adjustments[tournament]:
            st.session_state.player_data_adjustments[tournament][player_name] = {}
        st.session_state.player_data_adjustments[tournament][player_name][column] = value
    
    def reset_player_adjustments(self, tournament: str = None):
        """Reset player adjustments for a specific tournament or all tournaments"""
        if tournament:
            st.session_state.player_data_adjustments[tournament] = {}
        else:
            st.session_state.player_data_adjustments = {}
    
    def get_custom_weights(self):
        """Get current custom weights"""
        return st.session_state.custom_weights
    
    def update_custom_weight(self, stat_name: str, value: float):
        """Update a specific custom weight"""
        st.session_state.custom_weights[stat_name] = value
    
    def reset_custom_weights(self):
        """Reset custom weights to defaults"""
        st.session_state.custom_weights = CoursePlayerFit.DEFAULT_WEIGHTS_DICT.copy()

def get_available_tournaments():
    """Get list of tournaments that have data in the 2025 folder"""
    try:
        # Get all subdirectories in the 2025 folder
        tournaments = [d for d in os.listdir("2025") if os.path.isdir(os.path.join("2025", d))]
        return sorted(tournaments) if tournaments else ["No tournaments available"]
    except FileNotFoundError:
        return ["No tournaments available"]

def create_golfers_from_fit_details(filtered_data: pd.DataFrame, fit_details: pd.DataFrame) -> List[Golfer]:
    """
    Create list of Golfer objects from filtered data and fit details
    
    Args:
        filtered_data: DataFrame containing player data including names and salaries
        fit_details: DataFrame containing detailed fit statistics for each player
        
    Returns:
        List[Golfer]: List of initialized Golfer objects with their stats
    """
    golfers = []
    fit_details_player = fit_details.pivot(
        index='Name',
        columns='Course Stat',
        values='Player Value'
    ).reset_index()
    
    for _, row in filtered_data.iterrows():
        if len(fit_details_player[fit_details_player['Name'] == row['Name']]) > 0:
            player_stats = fit_details_player[fit_details_player['Name'] == row['Name']].iloc[0]
            player = Golfer({
                'Name + ID': row['Name'],
                'Salary': row['Salary']
            })
            player.stats = {
                'current': {
                    'driving_distance': player_stats['adj_driving_distance'],
                    'driving_accuracy': player_stats['adj_driving_accuracy'],
                    'scrambling_sand': player_stats['arg_bunker_sg'],
                    'strokes_gained': StrokesGained(
                        off_tee=player_stats['ott_sg'],
                        approach=player_stats['app_sg'],
                        around_green=player_stats['arg_sg'],
                        putting=player_stats['putt_sg']
                    )
                }
            }
            golfers.append(player)
    
    return golfers

def main():
    st.set_page_config(layout="wide", page_title="PGA DFS Dashboard", page_icon=":golf:")
    
    # Initialize the DataManager at the start
    data_manager = DataManager()
    
    # Main title
    st.title(f":golf: PGA DFS Dashboard")
    st.write("This dashboard is designed to help you make informed decisions about which golfers to draft for your PGA DFS lineup. It also has a built in optimizer to help build your lineups.")
    st.write("**Quick Start:** Select a tournament, use the sliders to adjust the weights (both in the left sidebar and in the Course Fit section), and run the model. The Optimized Lineups at the bottom will update automatically.")

    # Add mobile warning with CSS media query
    st.markdown(
        """
        <style>
            #mobile-warning {
                display: none;
                padding: 10px;
                background-color: #ff9800;
                color: white;
                border-radius: 5px;
                margin-bottom: 20px;
                font-weight: bold;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
            
            @media screen and (max-width: 768px) {
                #mobile-warning {
                    display: block;
                }
            }
        </style>
        <div id="mobile-warning">
            ⚠️ This dashboard is optimized for desktop viewing. Mobile users may experience limited functionality.
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # Sidebar
    st.sidebar.title("PGA DFS Dashboard")
    
    # Tournament selection using available tournaments
    selected_tournament = st.sidebar.selectbox(
        "Select Tournament",
        get_available_tournaments()
    )
    
    # Run model button (now data_manager is available)
    if st.sidebar.button("Run Model"):
        with st.spinner(f"Running model for {selected_tournament}..."):
            run_pga_model(selected_tournament, num_lineups=20, tournament_history=True, 
                         custom_weights=data_manager.get_custom_weights())
        st.sidebar.success("Model run complete!")
    
    # Main content
    st.title(f"{selected_tournament}")
    
    # Load data using DataManager
    player_data, lineups = data_manager.load_tournament_data(selected_tournament)
    
    if player_data is not None:
        # Load fit details at the start
        try:
            fit_details = pd.read_csv(f"2025/{selected_tournament}/fit_details.csv")
        except FileNotFoundError:
            st.error("No fit details data available for this tournament.")
            return
        
        # Weight adjustment section
        st.sidebar.subheader("Adjust Weights")
        
        # Default weights based on whether tournament history exists
        has_history = 'Normalized History' in player_data.columns and player_data['Normalized History'].sum() > 0
        
        default_odds = 0.5 if has_history else 0.5
        default_fit = 0.3 if has_history else 0.5
        default_history = 0.2 if has_history else 0.0
        
        # Weight sliders
        odds_weight = st.sidebar.slider("Odds Weight", 0.0, 1.0, default_odds, 0.1)
        fit_weight = st.sidebar.slider("Course Fit Weight", 0.0, 1.0, default_fit, 0.1)
        history_weight = st.sidebar.slider("Tournament History Weight", 0.0, 1.0, default_history, 0.1)
        
        # Normalize weights to sum to 1
        total_weight = odds_weight + fit_weight + history_weight
        if total_weight > 0:
            odds_weight = odds_weight / total_weight
            fit_weight = fit_weight / total_weight
            history_weight = history_weight / total_weight
            
        # Display normalized weights
        st.sidebar.write(f"Normalized Weights:")
        st.sidebar.write(f"Odds: {odds_weight:.2f}")
        st.sidebar.write(f"Fit: {fit_weight:.2f}")
        st.sidebar.write(f"History: {history_weight:.2f}")
        
        # Recalculate Total based on weights
        filtered_data = player_data.copy()
        
        # Use the function in the main code
        golfers = create_golfers_from_fit_details(filtered_data, fit_details)
        course_stats = load_course_stats(selected_tournament)
        analyzer = CoursePlayerFit(course_stats, golfers, custom_weights=data_manager.get_custom_weights())
        
        # Calculate fit scores for all players
        new_fit_scores = []
        for player in golfers:
            fit_score = analyzer.calculate_fit_score(player)
            new_fit_scores.append({
                'Name': player.name,
                'Normalized Fit': fit_score['overall_fit'] / 100  # Convert to 0-1 scale
            })
        
        # Update filtered_data with new fit scores
        new_fit_scores_df = pd.DataFrame(new_fit_scores)
        filtered_data = pd.merge(filtered_data, new_fit_scores_df, on='Name', how='left', suffixes=('_old', ''))

        # Recalculate Total based on weights
        filtered_data['Total'] = (
            filtered_data['Normalized Odds'] * odds_weight +
            filtered_data['Normalized Fit'] * fit_weight +
            filtered_data['Normalized History'] * history_weight
        )
        
        # Recalculate Value based on new Total
        filtered_data['Value'] = filtered_data['Total'] / filtered_data['Salary'] * 100000
        
        # Player Analysis section
        st.subheader("Player Analysis")
        st.write("This section summarizes each golfer's odds, fit, and history scores. This section updates dynamically as you make adjustements in the dashboard. It is the source-of-truth for running the model.")
        
        # Add search/filter box
        search = st.text_input("Search Players")
        if search:
            filtered_data = filtered_data[filtered_data['Name'].str.contains(search, case=False)]
        
        # Display player data with formatting
        st.dataframe(
            filtered_data[['Name', 'Salary', 'Normalized Odds', 'Normalized Fit', 'Normalized History', 'Total', 'Value']].style.format({
                'Salary': '${:,.0f}',
                'Normalized Odds': '{:.2f}',
                'Normalized Fit': '{:.2f}',
                'Normalized History': '{:.2f}',
                'Total': '{:.2f}',
                'Value': '{:.2f}'
            }),
            height=400,
            use_container_width=True
        )
        
        # New Player Odds section
        st.subheader("Player Odds")
        st.write("This section summarizes each golfer's odds for the tournament. The odds are based on the lines from https://www.scoresandodds.com/golf and are used to calculate the normalized odds score.")
        st.write("TODO: Add adjustment sliders for weighting each line.")
        try:
            odds_data = pd.read_csv(f"2025/{selected_tournament}/odds.csv")
            # Format the odds columns to show plus sign for positive values
            odds_columns = ['Tournament Winner', 'Top 5 Finish', 'Top 10 Finish', 'Top 20 Finish']
            for col in odds_columns:
                odds_data[col] = odds_data[col].apply(lambda x: f"+{x}" if x > 0 else str(x))
            
            # Display the odds data
            st.dataframe(
                odds_data,
                height=400,
                use_container_width=True
            )
        except FileNotFoundError:
            st.warning("No odds data available for this tournament.")
        
        # After odds section but before optimized lineups
        st.subheader("Course Fit Analysis")
        st.write("This section details the course and player fit, and combines them into a score that is shown in the anlaysis table in the beginning of the dashboard.")
        
        # After the Course Fit Analysis header but before displaying the data
        st.markdown("#### Adjust Course Fit Weights")
        st.write("These sliders allow you to adjust the weights of the course fit stats. The weights are used to calculate the fit score for each player. The values are by default set to what worked the best last year.")

        # Create columns for the sliders
        col1, col2 = st.columns(2)

        # Define callback function for each slider
        def update_weight(stat_name):
            data_manager.update_custom_weight(stat_name, st.session_state[f"{stat_name.lower().replace(' ', '_')}_weight"])

        # Add sliders for each weight in the first column
        with col1:
            st.slider(
                "Driving Distance Weight", 0.0, 2.0, 
                data_manager.get_custom_weights()['Driving Distance'], 0.1,
                key='driving_distance_weight',
                on_change=update_weight,
                args=('Driving Distance',)
            )
            st.slider(
                "Driving Accuracy Weight", 0.0, 2.0, 
                data_manager.get_custom_weights()['Driving Accuracy'], 0.1,
                key='driving_accuracy_weight',
                on_change=update_weight,
                args=('Driving Accuracy',)
            )
            st.slider(
                "Fairway Width Weight", 0.0, 2.0, 
                data_manager.get_custom_weights()['Fairway Width'], 0.1,
                key='fairway_width_weight',
                on_change=update_weight,
                args=('Fairway Width',)
            )
            st.slider(
                "Off the Tee SG Weight", 0.0, 2.0, 
                data_manager.get_custom_weights()['Off the Tee SG'], 0.1,
                key='off_the_tee_sg_weight',
                on_change=update_weight,
                args=('Off the Tee SG',)
            )

        # Add sliders for remaining weights in the second column
        with col2:
            st.slider(
                "Approach SG Weight", 0.0, 2.0, 
                data_manager.get_custom_weights()['Approach SG'], 0.1,
                key='approach_sg_weight',
                on_change=update_weight,
                args=('Approach SG',)
            )
            st.slider(
                "Around Green SG Weight", 0.0, 2.0, 
                data_manager.get_custom_weights()['Around Green SG'], 0.1,
                key='around_green_sg_weight',
                on_change=update_weight,
                args=('Around Green SG',)
            )
            st.slider(
                "Putting SG Weight", 0.0, 2.0, 
                data_manager.get_custom_weights()['Putting SG'], 0.1,
                key='putting_sg_weight',
                on_change=update_weight,
                args=('Putting SG',)
            )
            st.slider(
                "Sand Save Weight", 0.0, 2.0, 
                data_manager.get_custom_weights()['Sand Save'], 0.1,
                key='sand_save_weight',
                on_change=update_weight,
                args=('Sand Save',)
            )

        # Add a button to reset weights to defaults
        if st.button("Reset Weights to Defaults"):
            data_manager.reset_custom_weights()
            st.rerun()

        # After the weight adjustment section but before the next section
        try:
            # Define column mapping once
            column_mapping = {
                'adj_driving_distance': 'Driving Distance',
                'adj_driving_accuracy': 'Driving Accuracy',
                'fw_width': 'Fairway Width',
                'ott_sg': 'Off the Tee SG',
                'app_sg': 'Approach SG',
                'arg_sg': 'Around Green SG',
                'putt_sg': 'Putting SG',
                'arg_bunker_sg': 'Sand Save %'
            }
            
            # Course Stats Table
            course_stats_df = fit_details[['Course Stat', 'Course Value', 'Base Weight']].drop_duplicates()
            course_stats_df['Course Stat'] = course_stats_df['Course Stat'].map(column_mapping)
            course_stats_df = course_stats_df.drop_duplicates(subset=['Course Stat'])

            # Update Base Weight with current slider values
            reverse_mapping = {v: k for k, v in column_mapping.items()}
            for stat, weight in data_manager.get_custom_weights().items():
                original_stat = reverse_mapping.get(stat)
                if original_stat:
                    course_stats_df.loc[course_stats_df['Course Stat'] == stat, 'Base Weight'] = weight
            
            # Display course stats
            st.markdown("#### Course Statistics")
            st.write("This table shows the course stats and their base weights from the sliders above.")
            st.dataframe(
                course_stats_df.style.format({
                    'Course Value': '{:.3f}',
                    'Base Weight': '{:.3f}'
                }),
                height=200,
                use_container_width=True
            )
            
            st.markdown("---")  # Add a separator between tables
            st.markdown("#### Player Statistics")
            st.write("This table shows golfer's stats which is used to calculate the fit score based on the course stats and weights.")
            st.write("TODO: Allow user's to adjust short-term stats vs. long-term stats.\nCurrent weight is 0.7 short, 0.3 long.")
            # Player Fit Details
            display_df = fit_details.pivot(
                index='Name',
                columns='Course Stat',
                values='Player Value'
            ).reset_index()
            
            # Rename columns to be more readable
            display_df.columns = [column_mapping.get(col, col) for col in display_df.columns]

            # Display the data
            st.dataframe(
                display_df.style.format({
                    'Driving Distance': '{:.1f}',
                    'Driving Accuracy': '{:.3f}',
                    'Fairway Width': '{:.3f}',
                    'Off the Tee SG': '{:.3f}',
                    'Approach SG': '{:.3f}',
                    'Around Green SG': '{:.3f}',
                    'Putting SG': '{:.3f}',
                    'Sand Save %': '{:.3f}',
                }),
                height=400,
                use_container_width=True
            )
            
        except FileNotFoundError:
            st.warning("No course fit data available for this tournament.")
        
        # Tournament History section
        st.subheader("Tournament History")
        st.write("This section shows the golfer's history at this specific tournament. The history is used to calculate the normalized history score.")
        st.write("TODO: Clean up 'make cuts' column to be more accurate, currently uses T65 or better.")
        try:
            history_data = pd.read_csv(f"2025/{selected_tournament}/tournament_history.csv")
            
            # Format the columns for better display
            display_columns = ['Name', '24', '2022-23', '2021-22', '2020-21', '2019-20', 
                             'sg_ott', 'sg_app', 'sg_atg', 'sg_putting', 'sg_total',
                             'rounds', 'avg_finish', 'measured_years', 'made_cuts_pct']
            
            # Format numeric columns to show fewer decimal places
            numeric_cols = ['sg_ott', 'sg_app', 'sg_atg', 'sg_putting', 'sg_total', 
                          'avg_finish', 'made_cuts_pct']
            
            # Create formatted dataframe
            display_df = history_data[display_columns].copy()
            for col in numeric_cols:
                display_df[col] = display_df[col].apply(lambda x: f"{x:.3f}" if pd.notna(x) else "")
            
            # Sort by most recent year's finish
            display_df = display_df.sort_values('24', na_position='last')
            
            # Display the history data
            st.dataframe(
                display_df,
                height=400,
                use_container_width=True
            )
            
        except FileNotFoundError:
            st.warning("No tournament history data available for this tournament.")
        
        # Optimized Lineups section
        st.subheader("Optimized Lineups")
        
        # Display top lineups
        for i in range(min(5, len(lineups))):
            with st.expander(f"Lineup {i+1}"):
                lineup = lineups.iloc[i]
                total_salary = 0
                total_points = 0
                
                # Create a temporary dataframe for this lineup
                lineup_details = []
                for j in range(6):
                    player = lineup[f'G{j+1}']
                    # Use fix_names function for consistent name handling
                    player_name = fix_names(player.split('(')[0].strip())
                    player_data_row = filtered_data[filtered_data['Name'] == player_name]
                    if not player_data_row.empty:
                        salary = player_data_row['Salary'].iloc[0]
                        points = player_data_row['Total'].iloc[0]
                        total_salary += salary
                        total_points += points
                        lineup_details.append({
                            'Player': player,
                            'Salary': salary,
                            'Total': points
                        })
                
                # Display lineup details
                lineup_df = pd.DataFrame(lineup_details)
                st.dataframe(
                    lineup_df.style.format({
                        'Salary': '${:,.0f}',
                        'Total': '{:.2f}'
                    }),
                    use_container_width=True
                )
                st.write(f"Total Salary: ${total_salary:,.0f}")
                st.write(f"Total Points: {total_points:.2f}")
        
        # After the lineups section, add Info section
        st.subheader("Understanding the Scores") 
        
        with st.expander("Normalized Odds"):
            st.markdown("""
            **Odds Score** combines multiple betting market predictions:
            - Tournament Winner (Weight: 0.6)
            - Top 5 Finish (Weight: 0.5)
            - Top 10 Finish (Weight: 0.8)
            - Top 20 Finish (Weight: 0.4)
            
            The score is normalized to a 0-1 scale, where higher values indicate better betting market expectations.
            Points awarded:
            - Win: 30 pts
            - Top 5: 14 pts
            - Top 10: 7 pts
            - Top 20: 5 pts
            """)
            
        with st.expander("Normalized Fit"):
            st.markdown("""
            **Course Fit Score** analyzes how well a player's attributes match the course characteristics. 
            Includes multiple factors:
            
            1. **Distance Analysis**
            - Driving Distance vs Course Length
            - Adjusted for course-specific requirements
            
            2. **Accuracy Metrics**
            - Driving Accuracy
            - Fairway Width considerations
            - Green in Regulation
            
            3. **Strokes Gained Categories**
            - Off the Tee
            - Approach
            - Around the Green
            - Putting
            
            4. **Specific Situations**
            - Sand Save percentage
            - Course-specific challenges
            
            All metrics are normalized to a 0-1 scale, with higher values indicating better course fit.
            """)
            
        with st.expander("Normalized History"):
            st.markdown("""
            **Tournament History Score** evaluates past performance at this specific event.
            
            Factors considered:
            - Recent performance (last 3 years weighted more heavily)
            - Finish positions
            - Strokes Gained data from previous appearances
            - Cut percentage
            
            Scoring system:
            - More recent results have higher weights
            - Consistency bonus for multiple good performances
            - Penalty for missed cuts
            - Normalized to 0-1 scale
            
            *Note: This score is only available for tournaments with historical data.*
            """)
            
        with st.expander("Total Score & Value"):
            st.markdown("""
            **Total Score** is calculated using the weighted combination of the three normalized scores:
            - Odds Score × Odds Weight
            - Fit Score × Fit Weight
            - History Score × History Weight
            
            **Value** is calculated as: `(Total Score / Salary) × 100,000`
            
            Use the sliders in the sidebar to adjust the weights and optimize for different strategies.
            """)
    
    else:
        st.warning("No data available for this tournament. Please run the model first.")

if __name__ == "__main__":
    main()
