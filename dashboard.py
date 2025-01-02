import streamlit as st
import pandas as pd
import os
from pga_v5 import main as run_pga_model, fix_names
from utils import TOURNAMENT_LIST_2024

def load_tournament_data(tournament: str) -> tuple:
    """Load data for a specific tournament"""
    try:
        # Load player data
        player_data = pd.read_csv(f"2025/{tournament}/player_data.csv")
        # Load lineup data
        lineups = pd.read_csv(f"2025/{tournament}/dk_lineups_optimized.csv")
        return player_data, lineups
    except FileNotFoundError:
        return None, None

def main():
    st.set_page_config(layout="wide", page_title="PGA DFS Dashboard")
    
    # Sidebar
    st.sidebar.title("PGA DFS Dashboard")
    
    # Tournament selection
    selected_tournament = st.sidebar.selectbox(
        "Select Tournament",
        TOURNAMENT_LIST_2024
    )
    
    # Run model button
    if st.sidebar.button("Run Model"):
        with st.spinner(f"Running model for {selected_tournament}..."):
            run_pga_model(selected_tournament, num_lineups=20, tournament_history=True)
        st.sidebar.success("Model run complete!")
    
    # Main content
    st.title(f"PGA DFS Analysis: {selected_tournament}")
    
    # Load data
    player_data, lineups = load_tournament_data(selected_tournament)
    
    if player_data is not None and lineups is not None:
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
        filtered_data['Total'] = (
            filtered_data['Normalized Odds'] * odds_weight +
            filtered_data['Normalized Fit'] * fit_weight +
            filtered_data['Normalized History'] * history_weight
        )
        
        # Recalculate Value based on new Total
        filtered_data['Value'] = filtered_data['Total'] / filtered_data['Salary'] * 100000
        
        # Player Analysis section
        st.subheader("Player Analysis")
        
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
        
        # # Add some basic stats
        # st.subheader("Value Plays")
        # value_threshold = st.slider("Value Threshold", 
        #                          min_value=1.0, 
        #                          max_value=3.0, 
        #                          value=2.0,
        #                          step=0.1)
        # value_plays = player_data[player_data['Value'] >= value_threshold]
        # st.dataframe(
        #     value_plays[['Name', 'Salary', 'Total', 'Value']].style.format({
        #         'Salary': '${:,.0f}',
        #         'Total': '{:.2f}',
        #         'Value': '{:.2f}'
        #     })
        # )
        
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
