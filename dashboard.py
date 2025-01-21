import streamlit as st
import pandas as pd
import os
from pga_v5 import main as run_pga_model
from utils import fix_names

class DataManager:
    """Manages data and session state for the dashboard"""
    
    def __init__(self):
        # Initialize session state if needed
        if 'weights' not in st.session_state:
            st.session_state.weights = {
                'components': {
                    'odds': 0.25,
                    'fit': 0.25,
                    'history': 0.25,
                    'form': 0.25
                },
                'form': {
                    'current': 0.7,
                    'long': 0.3
                },
                'odds': {
                    'winner': 0.6,
                    'top5': 0.5,
                    'top10': 0.8,
                    'top20': 0.4
                }
            }
        
        # # Initialize component weight sliders
        # if 'odds_weight' not in st.session_state:
        #     st.session_state.odds_weight = st.session_state.weights['components']['odds']
        # if 'fit_weight' not in st.session_state:
        #     st.session_state.fit_weight = st.session_state.weights['components']['fit']
        # if 'history_weight' not in st.session_state:
        #     st.session_state.history_weight = st.session_state.weights['components']['history']
        # if 'form_weight' not in st.session_state:
        #     st.session_state.form_weight = st.session_state.weights['components']['form']
            
        # Initialize form weight sliders if needed
        if 'current_form_weight' not in st.session_state:
            st.session_state.current_form_weight = st.session_state.weights['form']['current']
        if 'long_form_weight' not in st.session_state:
            st.session_state.long_form_weight = st.session_state.weights['form']['long']
            
        # Add reference to session state weights
        self.weights = st.session_state.weights
        if 'player_data' not in st.session_state:
            st.session_state.player_data = {}
        self.player_data = st.session_state.player_data

        # Add reset callback to initialization
        if 'reset_component_weights_clicked' not in st.session_state:
            st.session_state.reset_component_weights_clicked = False
        if 'reset_form_weights_clicked' not in st.session_state:
            st.session_state.reset_form_weights_clicked = False

    def load_tournament_data(self, tournament: str) -> tuple:
        """Load and process tournament data with any user adjustments"""
        player_data = None
        lineups = None
        
        # Try to load player data
        try:
            player_data = pd.read_csv(f"2025/{tournament}/player_data.csv")
            # Apply any stored adjustments
            if tournament in st.session_state.player_data:
                adjustments = st.session_state.player_data[tournament]
                for player_name, values in adjustments.items():
                    if player_name in player_data['Name'].values:
                        for column, value in values.items():
                            player_data.loc[player_data['Name'] == player_name, column] = value
        except FileNotFoundError:
            pass
        
        # Try to load lineup data independently
        try:
            lineups = pd.read_csv(f"2025/{tournament}/dk_lineups_optimized.csv")
        except FileNotFoundError:
            pass
        
        return player_data, lineups
    
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
    
    def update_component_weight(self, component: str):
        """Update a component weight without normalization"""
        # Get the value from session state using the component's key
        value = st.session_state[f"{component}_weight"]
        
        # Update the weight
        st.session_state.weights['components'][component] = value
        
        # Update the session state value
        st.session_state[f"{component}_weight"] = value

    def update_form_weight(self, form_type: str):
        """Update form weights ensuring they sum to 1"""
        # Get the new value from session state
        value = st.session_state[f"{form_type}_form_weight"]
        
        # Update both session state values
        if form_type == 'current':
            st.session_state.current_form_weight = value
            st.session_state.long_form_weight = 1 - value
        else:  # form_type == 'long'
            st.session_state.long_form_weight = value
            st.session_state.current_form_weight = 1 - value
        
        # Also update the weights dictionary
        st.session_state.weights['form']['current'] = st.session_state.current_form_weight
        st.session_state.weights['form']['long'] = st.session_state.long_form_weight

    def on_reset_component_weights(self):
        """Callback for resetting component weights"""
        st.session_state.reset_component_weights_clicked = True
        st.session_state.odds_weight = 0.4
        st.session_state.fit_weight = 0.2
        st.session_state.history_weight = 0.2
        st.session_state.form_weight = 0.2
        st.session_state.weights['components'] = {
            'odds': 0.4,
            'fit': 0.2,
            'history': 0.2,
            'form': 0.2
        }

    def on_reset_form_weights(self):
        """Callback for resetting form weights"""
        st.session_state.reset_form_weights_clicked = True
        st.session_state.current_form_weight = 0.7
        st.session_state.long_form_weight = 0.3
        st.session_state.weights['form'] = {
            'current': 0.7,
            'long': 0.3
        }

def get_available_tournaments(featured_tournament: str = None):
    """Get list of tournaments that have data in the 2025 folder"""
    try:
        # Get all subdirectories in the 2025 folder
        tournaments = [d for d in os.listdir("2025") if os.path.isdir(os.path.join("2025", d))]
        
        # Sort alphabetically first
        tournaments.sort()
        
        # If featured tournament is specified and exists, move it to front
        if featured_tournament and featured_tournament in tournaments:
            tournaments.remove(featured_tournament)
            tournaments.insert(0, featured_tournament)
            
        return tournaments if tournaments else ["No tournaments available"]
    except FileNotFoundError:
        return ["No tournaments available"]

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
        get_available_tournaments(featured_tournament="Farmers_Insurance_Open")
    )
    
    # Add weight validation
    total_weight = sum(data_manager.weights['components'].values())
    if not abs(total_weight - 1.0) < 0.01:  # Allow for small floating point differences
        st.sidebar.error(f"⚠️ Component weights must sum to 1.0 (Current total: {total_weight:.2f})")
    
    # Add number input for lineup count (just before Run Model button)
    num_lineups = st.sidebar.number_input(
        "Number of Lineups",
        min_value=1,
        max_value=150,
        value=20,
        step=1,
        help="""Specify how many lineups to generate.

Note: The more lineups you generate, the longer it will take to run the model."""
    )
    
    # Run model button with num_lineups parameter
    if st.sidebar.button("Run Model"):
        with st.spinner(f"Running model for {selected_tournament}..."):
            run_pga_model(selected_tournament,
                         weights=data_manager.weights,
                         num_lineups=num_lineups)
            # Reload the data after running the model
            _, lineups = data_manager.load_tournament_data(selected_tournament)
            st.sidebar.success("Model run complete!")
            
            # Add export button right after model completion
            st.sidebar.download_button(
                label="Export Lineups to CSV",
                data=lineups.to_csv(index=False),
                file_name=f"{selected_tournament}_lineups.csv",
                mime="text/csv"
            )
    
    # Main content
    st.title(f"{selected_tournament}")
    
    # Load data using DataManager
    player_data, lineups = data_manager.load_tournament_data(selected_tournament)
    
    if player_data is None:
        st.warning("No player data available for this tournament.")
        return
    
    # Weight adjustment section
    st.sidebar.subheader("Adjust Weights")
    
    # Component weights
    odds_weight = st.sidebar.slider("Odds Weight", 0.0, 1.0, 
                                    value=data_manager.weights['components']['odds'],
                                    step=0.05,
                                    key='odds_weight',
                                    on_change=data_manager.update_component_weight,
                                    args=('odds',),
                                    help="""Weight given to betting market predictions.

Higher values place more emphasis on odds from sportsbooks for tournament winner, top 5, top 10, and top 20 finishes.""")
    
    fit_weight = st.sidebar.slider("Course Fit Weight", 0.0, 1.0,
                                    value=data_manager.weights['components']['fit'],
                                    step=0.05,
                                    key='fit_weight',
                                    on_change=data_manager.update_component_weight,
                                    args=('fit',),
                                    help="""Weight given to course fit metrics.

Higher values emphasize how well a player's attributes match the course characteristics, including distance, accuracy, and specific course challenges.""")
    
    history_weight = st.sidebar.slider("Tournament History Weight", 0.0, 1.0,
                                    value=data_manager.weights['components']['history'],
                                    step=0.05,
                                    key='history_weight',
                                    on_change=data_manager.update_component_weight,
                                    args=('history',),
                                    help="""Weight given to tournament history.

Higher values emphasize past performance at this specific event, including finish positions and strokes gained data.""")
    
    form_weight = st.sidebar.slider("Form Weight", 0.0, 1.0,
                                    value=data_manager.weights['components']['form'],
                                    step=0.05,
                                    key='form_weight',
                                    on_change=data_manager.update_component_weight,
                                    args=('form',),
                                    help="""Weight given to player form.

Higher values emphasize recent performance metrics, combining both short-term (last 5 starts) and long-term (season-long) statistics.""")
    
    if st.sidebar.button("Reset Component Weights", 
                        on_click=data_manager.on_reset_component_weights):
        st.sidebar.success("Component weights reset to default values!")
    
    # Recalculate Total based on weights
    filtered_data = player_data.copy()
        
        
    
    # Recalculate Total based on weights
    filtered_data = player_data.copy()
    
    

    
    # Update filtered_data with new fit scores

    # Recalculate Total based on weights
    filtered_data['Total'] = (
        filtered_data['Normalized Odds'] * odds_weight +
        filtered_data['Normalized Fit'] * fit_weight +
        filtered_data['Normalized History'] * history_weight +
        filtered_data['Normalized Form'] * form_weight
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
        filtered_data[['Name', 'Salary', 'Normalized Odds', 'Normalized Fit', 'Normalized History', 'Normalized Form', 'Total', 'Value']].style.format({
            'Salary': '${:,.0f}',
            'Normalized Odds': '{:.2f}',
            'Normalized Fit': '{:.2f}',
            'Normalized History': '{:.2f}',
            'Normalized Form': '{:.2f}',
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
    
                # Course Fit section (simplified)
    st.subheader("Course Fit")
    try:
        course_fit = pd.read_csv(f"2025/{selected_tournament}/course_fit.csv")
        st.dataframe(
            course_fit.sort_values('projected_course_fit'),
            height=400,
            use_container_width=True
        )
    except FileNotFoundError:
        st.warning("No course fit data available for this tournament.")

    # Player Form section
    st.subheader("Player Form")
    st.write("Short-term form represents last 5 starts, Long-term form represents the entire season.")
    
    col1, col2 = st.columns(2)

    if st.button("Reset Form Weights",
                 on_click=data_manager.on_reset_form_weights):
        st.success("Form weights reset to default values!")
    
    with col1:
        st.markdown("#### Short-term Form")
        st.slider(
            "Current Form Weight", 
            0.0, 
            1.0,
            key="current_form_weight",
            on_change=data_manager.update_form_weight,
            args=('current',)
        )
        
        try:
            current_form = pd.read_csv(f"2025/{selected_tournament}/current_form.csv")
            st.dataframe(current_form, height=400, use_container_width=True)
        except FileNotFoundError:
            st.warning("No current form data available.")
            
    with col2:
        st.markdown("#### Long-term Form")
        st.slider(
            "Long-term Form Weight", 
            0.0, 
            1.0,
            key="long_form_weight",
            on_change=data_manager.update_form_weight,
            args=('long',)
        )
        try:
            long_form = pd.read_csv(f"2025/{selected_tournament}/pga_stats.csv")
            st.dataframe(long_form, height=400, use_container_width=True)
        except FileNotFoundError:
            st.warning("No long-term form data available.")
            
        # Tournament History section
    st.subheader("Tournament History")
    st.write("This section shows the golfer's history at this specific tournament. The history is used to calculate the normalized history score.")
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
    
    if lineups is None:
        st.warning("No lineup data available. Please run the model first to generate optimized lineups.")
    else:
        
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
        
    with st.expander("Normalized Form"):
        st.markdown("""
        **Form Score** combines both short-term and long-term performance metrics:
        
        **Short-term Form (Current Weight: {:.1f})**
        - Recent tournament performances
        - Current strokes gained statistics
        - Last 5-10 tournaments
        
        **Long-term Form (Current Weight: {:.1f})**
        - Season-long statistics
        - Historical performance metrics
        - Overall strokes gained data
        
        The final form score is a weighted combination of both components, normalized to a 0-1 scale.
        """.format(data_manager.weights['form']['current'],
                    data_manager.weights['form']['long']))
        
    with st.expander("Total Score & Value"):
        st.markdown("""
        **Total Score** is calculated using the weighted combination of the three normalized scores:
        - Odds Score × Odds Weight
        - Fit Score × Fit Weight
        - History Score × History Weight
        
        **Value** is calculated as: `(Total Score / Salary) × 100,000`
        
        Use the sliders in the sidebar to adjust the weights and optimize for different strategies.
        """)

    # Handle form weight reset
    if 'reset_form' in st.session_state and st.session_state.reset_form:
        st.session_state.current_form_weight = 0.7
        st.session_state.long_form_weight = 0.3
        st.session_state.reset_form = False
        st.success("Form weights reset to default values!")

if __name__ == "__main__":
    main()
