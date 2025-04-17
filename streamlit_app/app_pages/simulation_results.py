import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import simulation
import data_handlers

def display_simulation_results(model_data=None):
    """Display the full simulation results page"""
    st.title("NHL Playoff Simulation Results")
    
    # Check if we have simulation results
    sim_results = simulation.get_simulation_results()
    
    if not sim_results or 'team_advancement' not in sim_results:
        st.error("No simulation results available. Try running a simulation first.")
        
        # Add a button to force run simulations
        if st.button("Run Simulations Now", key="run_sims_results"):
            with st.spinner("Running playoff simulations..."):
                success = simulation.update_daily_simulations(force=True)
                if success:
                    st.success("Simulations completed successfully!")
                    # Refresh the page to show new results
                    st.experimental_rerun()
                else:
                    st.error("Failed to run simulations. Please check logs for details.")
        return
    
    # Show when the simulations were last run
    if 'last_simulation_refresh' in st.session_state:
        last_refresh = st.session_state.last_simulation_refresh
        st.caption(f"Simulations last run: {last_refresh.strftime('%Y-%m-%d %H:%M')}")
    
    # Get the formatted results
    results_df = sim_results['team_advancement']
    
    # Create tabs for different result views
    tab1, tab2, tab3 = st.tabs(["Championship Odds", "Round-by-Round", "Potential Matchups"])
    
    with tab1:
        st.subheader("Stanley Cup Championship Odds")
        
        # Format the data for display
        display_df = results_df.copy()
        display_df['champion'] = (display_df['champion'] * 100).round(1)
        display_df = display_df.sort_values('champion', ascending=False)
        
        # Create a horizontal bar chart
        fig, ax = plt.subplots(figsize=(10, 8))
        teams = display_df['teamName'].values[:10]  # Top 10 teams
        probs = display_df['champion'].values[:10]
        y_pos = np.arange(len(teams))
        
        # Create bars with team colors (placeholder colors)
        bars = ax.barh(y_pos, probs, color='royalblue')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(teams)
        ax.invert_yaxis()  # Teams listed from top to bottom
        ax.set_xlabel('Championship Probability (%)')
        ax.set_title('Stanley Cup Championship Odds')
        
        # Add percentage labels to bars
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax.text(width + 0.5, bar.get_y() + bar.get_height()/2, f"{width}%", 
                    ha='left', va='center')
        
        st.pyplot(fig)
        
        # Display the full table
        st.subheader("Full Championship Odds Table")
        table_df = display_df[['teamName', 'champion']].copy()
        table_df.columns = ['Team', 'Championship Probability (%)']
        st.dataframe(table_df, use_container_width=True)
    
    with tab2:
        st.subheader("Round-by-Round Advancement Probabilities")
        
        # Show visualization first, by default
        # Create chart showing progression through rounds
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Prepare data
        teams = results_df['teamName'].values[:16]  # Top 16 teams
        rounds = ['round_1', 'round_2', 'conf_final', 'final', 'champion']
        round_labels = ['First Round', 'Second Round', 'Conf Finals', 'Cup Final', 'Champion']
        
        # Sort teams by champion probability
        team_order = results_df.sort_values('champion', ascending=False)['teamName'].values[:16]
        
        # Get data in correct order
        data = []
        for team in team_order:
            team_data = results_df[results_df['teamName'] == team]
            if not team_data.empty:
                data.append([team_data[round].values[0] * 100 for round in rounds])
        
        # Create stacked bar chart with improved colors
        data_array = np.array(data)
        bottoms = np.zeros(len(team_order))
        
        # Using a better color palette
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        
        for i, round_label in enumerate(round_labels):
            ax.barh(team_order, data_array[:, i], left=bottoms, label=round_label, 
                   color=colors[i], alpha=1.0)
            bottoms += data_array[:, i]
        
        ax.set_xlabel('Probability (%)')
        ax.set_title('Round-by-Round Playoff Advancement Probabilities')
        
        # Move legend to top right and make it more appealing
        ax.legend(loc='upper right', framealpha=0.9, frameon=True, 
                 fancybox=True, shadow=True, fontsize=10)
        
        # Add a grid for better readability
        ax.grid(axis='x', alpha=0.3, linestyle='--')
        
        # Improve overall appearance
        plt.tight_layout()
        plt.xticks(np.arange(0, 101, 20))  # Set x-axis ticks at 0%, 20%, 40%, etc.
        
        st.pyplot(fig)
        
        # Format the data for display table (below the visualization)
        round_df = results_df.copy()
        for col in ['round_1', 'round_2', 'conf_final', 'final', 'champion']:
            round_df[col] = (round_df[col] * 100).round(1).astype(str) + '%'
        round_df['avg_games_played'] = round_df['avg_games_played'].round(1)
        
        # Rename columns for display
        display_cols = {
            'teamName': 'Team',
            'round_1': 'First Round',
            'round_2': 'Second Round', 
            'conf_final': 'Conf. Finals',
            'final': 'Cup Final',
            'champion': 'Win Cup',
            'avg_games_played': 'Avg. Games'
        }
        round_df = round_df.rename(columns=display_cols)
        
        # Sort by championship probability (removing % for sorting)
        sort_order = results_df['champion'].values
        round_df['sort_key'] = sort_order
        round_df = round_df.sort_values('sort_key', ascending=False).drop('sort_key', axis=1)
        
        # Display the round-by-round table
        st.subheader("Detailed Probability Table")
        st.dataframe(round_df[display_cols.values()], use_container_width=True)
    
    with tab3:
        st.subheader("Potential Playoff Matchups")
        
        # Display potential second round matchups if available
        if 'round2_matchups' in sim_results and not sim_results['round2_matchups'].empty:
            st.subheader("Most Likely Second Round Matchups")
            
            r2_df = sim_results['round2_matchups'].copy()
            r2_df['probability'] = r2_df['probability'].round(1).astype(str) + '%'
            r2_df['top_seed_win_pct'] = r2_df['top_seed_win_pct'].round(1).astype(str) + '%'
            
            r2_cols = {
                'matchup': 'Matchup',
                'conference': 'Conference',
                'probability': 'Probability',
                'top_seed_win_pct': 'Higher Seed Win %'
            }
            
            st.dataframe(r2_df[r2_cols.keys()].rename(columns=r2_cols), use_container_width=True)
        
        # Display potential conference finals matchups
        if 'conf_final_matchups' in sim_results and not sim_results['conf_final_matchups'].empty:
            st.subheader("Most Likely Conference Finals Matchups")
            
            cf_df = sim_results['conf_final_matchups'].copy()
            cf_df['probability'] = cf_df['probability'].round(1).astype(str) + '%'
            cf_df['top_seed_win_pct'] = cf_df['top_seed_win_pct'].round(1).astype(str) + '%'
            
            cf_cols = {
                'matchup': 'Matchup',
                'conference': 'Conference',
                'probability': 'Probability',
                'top_seed_win_pct': 'Higher Seed Win %'
            }
            
            st.dataframe(cf_df[cf_cols.keys()].rename(columns=cf_cols), use_container_width=True)
        
        # Display potential Stanley Cup Finals matchups
        if 'final_matchups' in sim_results and not sim_results['final_matchups'].empty:
            st.subheader("Most Likely Stanley Cup Finals Matchups")
            
            f_df = sim_results['final_matchups'].copy()
            f_df['probability'] = f_df['probability'].round(1).astype(str) + '%'
            f_df['top_seed_win_pct'] = f_df['top_seed_win_pct'].round(1).astype(str) + '%'
            
            f_cols = {
                'matchup': 'Matchup',
                'probability': 'Probability',
                'top_seed_win_pct': 'Higher Seed Win %'
            }
            
            st.dataframe(f_df[f_cols.keys()].rename(columns=f_cols), use_container_width=True)
    
    # Add a refresh button
    st.sidebar.subheader("Refresh Simulations")
    if st.sidebar.button("Run New Simulations"):
        with st.spinner("Running playoff simulations..."):
            success = simulation.update_daily_simulations(force=True)
            if success:
                st.sidebar.success("Simulations completed successfully!")
                st.experimental_rerun()
            else:
                st.sidebar.error("Failed to run simulations. Please check logs for details.")

if __name__ == "__main__":
    # This allows the page to be run directly for testing
    display_simulation_results()
