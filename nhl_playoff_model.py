import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import requests
import json
import random
from tqdm import tqdm

class NHLPlayoffSimulator:
    def __init__(self):
        self.current_season = self.get_current_season()
        self.teams_data = None
        self.playoff_tree = None
        self.sim_results = {}

    def get_current_season(self):
        """Determine the current NHL season based on the date"""
        today = datetime.now()
        if today.month < 9:  # If current month is before September
            return f"{today.year - 1}{today.year}"
        else:
            return f"{today.year}{today.year + 1}"

    def fetch_team_data(self, season=None):
        """Fetch team data for a specific season"""
        if season is None:
            season = self.current_season
        
        print(f"Fetching team data for season {season}...")
        
        # NHL API endpoint for team stats
        url = f"https://api-web.nhle.com/v1/standings/{season}"
        
        try:
            response = requests.get(url)
            response.raise_for_status()  # Raise exception for HTTP errors
            
            # Parse the JSON response
            data = response.json()
            
            # Extract team stats from the standings data
            teams_list = []
            for conference in data.get('standings', []):
                for team in conference:
                    team_info = {
                        'team_id': team.get('teamAbbrev', {}).get('default', ''),
                        'team_name': team.get('teamName', {}).get('default', ''),
                        'conference': team.get('conferenceName', ''),
                        'division': team.get('divisionName', ''),
                        'gp': team.get('gamesPlayed', 0),
                        'wins': team.get('wins', 0),
                        'losses': team.get('losses', 0),
                        'ot_losses': team.get('otLosses', 0),
                        'points': team.get('points', 0),
                        'point_pct': team.get('pointPctg', 0),
                        'goals_for': team.get('goalFor', 0),
                        'goals_against': team.get('goalAgainst', 0),
                        'goal_diff': team.get('goalDifferential', 0)
                    }
                    teams_list.append(team_info)
            
            self.teams_data = pd.DataFrame(teams_list)
            return self.teams_data
        
        except requests.exceptions.RequestException as e:
            print(f"Error fetching team data: {e}")
            return None

    def create_playoff_bracket(self):
        """Create the initial playoff bracket based on current standings"""
        if self.teams_data is None:
            self.fetch_team_data()
        
        # Sort teams by conference, division, and points to determine playoff teams
        teams_by_conference = {}
        for conference in self.teams_data['conference'].unique():
            conf_teams = self.teams_data[self.teams_data['conference'] == conference].sort_values(
                by=['division', 'points', 'wins'], ascending=[True, False, False]
            ).reset_index(drop=True)
            
            # Get top 3 teams from each division
            divisions = conf_teams['division'].unique()
            playoff_teams = []
            
            for division in divisions:
                div_teams = conf_teams[conf_teams['division'] == division]
                playoff_teams.extend(div_teams.head(3).to_dict('records'))
            
            # Get wildcard teams (next 2 highest point totals regardless of division)
            remaining_teams = conf_teams[~conf_teams['team_id'].isin([t['team_id'] for t in playoff_teams])]
            wildcard_teams = remaining_teams.sort_values(by=['points', 'wins'], ascending=[False, False]).head(2).to_dict('records')
            playoff_teams.extend(wildcard_teams)
            
            teams_by_conference[conference] = playoff_teams
        
        # Create the playoff bracket structure
        self.playoff_tree = {
            "round1": [],  # First round matchups
            "round2": [],  # Conference semifinals
            "round3": [],  # Conference finals
            "round4": []   # Stanley Cup Final
        }
        
        # Set up first-round matchups
        for conference, teams in teams_by_conference.items():
            # Sort divisions
            div1_teams = [t for t in teams if t['division'] == teams[0]['division']]
            div2_teams = [t for t in teams if t['division'] == teams[3]['division']]
            
            # Sort by position within division
            div1_teams = sorted(div1_teams, key=lambda x: x['points'], reverse=True)
            div2_teams = sorted(div2_teams, key=lambda x: x['points'], reverse=True)
            
            # Get wildcard teams (last two teams)
            wildcard_teams = sorted(
                [t for t in teams if t not in div1_teams[:3] and t not in div2_teams[:3]],
                key=lambda x: x['points'],
                reverse=True
            )
            
            # Create first round matchups
            # Division winners vs wildcards
            div1_winner = div1_teams[0]
            div2_winner = div2_teams[0]
            
            # Assign wildcards to division winners (higher seed plays lower wildcard)
            if div1_winner['points'] >= div2_winner['points']:
                self.playoff_tree["round1"].append({
                    "team1": div1_winner,
                    "team2": wildcard_teams[1],  # Lower wildcard
                    "conference": conference
                })
                self.playoff_tree["round1"].append({
                    "team1": div2_winner,
                    "team2": wildcard_teams[0],  # Higher wildcard
                    "conference": conference
                })
            else:
                self.playoff_tree["round1"].append({
                    "team1": div2_winner,
                    "team2": wildcard_teams[1],  # Lower wildcard
                    "conference": conference
                })
                self.playoff_tree["round1"].append({
                    "team1": div1_winner, 
                    "team2": wildcard_teams[0],  # Higher wildcard
                    "conference": conference
                })
            
            # 2nd vs 3rd matchups in each division
            self.playoff_tree["round1"].append({
                "team1": div1_teams[1],  # 2nd in division 1
                "team2": div1_teams[2],  # 3rd in division 1
                "conference": conference
            })
            
            self.playoff_tree["round1"].append({
                "team1": div2_teams[1],  # 2nd in division 2
                "team2": div2_teams[2],  # 3rd in division 2
                "conference": conference
            })
        
        return self.playoff_tree

    def simulate_series(self, team1, team2, round_num=1):
        """Simulate a best-of-7 playoff series between two teams"""
        # Calculate win probability based on point percentage, adjusted for home ice
        point_diff = team1['point_pct'] - team2['point_pct']
        
        # Base probability adjustment based on round number (later rounds have more randomness)
        if round_num == 1:
            base_prob = 0.55  # More predictable
        elif round_num == 2:
            base_prob = 0.53
        elif round_num == 3:
            base_prob = 0.52
        else:  # Stanley Cup Final
            base_prob = 0.51  # Most unpredictable
        
        # Calculate win probability with diminishing returns for point percentage difference
        win_prob = base_prob + (point_diff * 0.5)
        
        # Keep probability within reasonable bounds
        win_prob = max(0.35, min(win_prob, 0.65))
        
        # Simulate the series
        team1_wins = 0
        team2_wins = 0
        games = []
        
        # Best-of-7 series continues until one team has 4 wins
        while team1_wins < 4 and team2_wins < 4:
            game_num = team1_wins + team2_wins + 1
            
            # Home ice alternates based on 2-2-1-1-1 format
            if game_num in [1, 2, 5, 7]:
                team1_home_advantage = 0.05  # Additional probability for home team
                game_win_prob = win_prob + team1_home_advantage
            else:
                team2_home_advantage = 0.05  # Additional probability for home team
                game_win_prob = win_prob - team2_home_advantage
            
            # Simulate game result
            if random.random() < game_win_prob:
                team1_wins += 1
                winner = team1['team_id']
            else:
                team2_wins += 1
                winner = team2['team_id']
            
            games.append({
                'game': game_num,
                'winner': winner,
                'score': f"{team1_wins}-{team2_wins}"
            })
        
        # Return series result
        winner = team1 if team1_wins > team2_wins else team2
        return {
            'team1_id': team1['team_id'],
            'team2_id': team2['team_id'],
            'team1_wins': team1_wins,
            'team2_wins': team2_wins,
            'winner': winner,
            'games': games
        }

    def simulate_playoff_round(self, round_num):
        """Simulate an entire playoff round and advance teams to the next round"""
        current_round = f"round{round_num}"
        next_round = f"round{round_num + 1}"
        
        # Initialize next round if it doesn't exist
        if next_round not in self.playoff_tree:
            self.playoff_tree[next_round] = []
        
        # Simulate each series in the current round
        round_results = []
        
        for matchup in self.playoff_tree[current_round]:
            series_result = self.simulate_series(
                matchup['team1'], 
                matchup['team2'],
                round_num
            )
            
            round_results.append({
                'matchup': f"{series_result['team1_id']} vs {series_result['team2_id']}",
                'winner': series_result['winner']['team_id'],
                'series_score': f"{series_result['team1_wins']}-{series_result['team2_wins']}"
            })
            
            # Store the conference for next round matchups
            if 'conference' in matchup:
                conference = matchup['conference']
            else:
                conference = None
        
        # Organize winners for the next round
        if round_num < 3:  # Rounds 1 and 2 are within conference
            winners_by_conference = {}
            
            for idx, matchup in enumerate(self.playoff_tree[current_round]):
                winner = round_results[idx]['winner']
                winner_team = matchup['team1'] if matchup['team1']['team_id'] == winner else matchup['team2']
                
                conf = matchup['conference']
                if conf not in winners_by_conference:
                    winners_by_conference[conf] = []
                winners_by_conference[conf].append(winner_team)
            
            # Create matchups for next round
            for conf, winners in winners_by_conference.items():
                if round_num == 1:
                    # For round 2, match winners based on original position
                    # Teams are already in the correct order from create_playoff_bracket
                    self.playoff_tree[next_round].append({
                        'team1': winners[0],
                        'team2': winners[2],
                        'conference': conf
                    })
                    
                    self.playoff_tree[next_round].append({
                        'team1': winners[1],
                        'team2': winners[3],
                        'conference': conf
                    })
                elif round_num == 2:
                    # For conference finals, just match the two remaining teams
                    self.playoff_tree[next_round].append({
                        'team1': winners[0],
                        'team2': winners[1],
                        'conference': conf
                    })
        
        elif round_num == 3:
            # For Stanley Cup Final, match conference champions
            if len(round_results) == 2:
                # Get the winner from each conference final
                east_winner = None
                west_winner = None
                
                for idx, matchup in enumerate(self.playoff_tree[current_round]):
                    winner_id = round_results[idx]['winner']
                    winner_team = matchup['team1'] if matchup['team1']['team_id'] == winner_id else matchup['team2']
                    
                    if matchup['conference'] == 'Eastern':
                        east_winner = winner_team
                    else:
                        west_winner = winner_team
                
                if east_winner and west_winner:
                    # Higher seed (points) gets home ice advantage
                    if east_winner['points'] >= west_winner['points']:
                        self.playoff_tree[next_round].append({
                            'team1': east_winner,
                            'team2': west_winner
                        })
                    else:
                        self.playoff_tree[next_round].append({
                            'team1': west_winner,
                            'team2': east_winner
                        })
        
        return round_results

    def simulate_playoffs(self):
        """Simulate the entire playoff tournament"""
        if self.playoff_tree is None:
            self.create_playoff_bracket()
        
        # Dictionary to store results of each round
        all_results = {}
        
        # Simulate each round
        for round_num in range(1, 5):
            round_key = f"round{round_num}"
            if round_key in self.playoff_tree and len(self.playoff_tree[round_key]) > 0:
                print(f"Simulating playoff round {round_num}...")
                round_results = self.simulate_playoff_round(round_num)
                all_results[round_key] = round_results
        
        # Get the Stanley Cup champion
        if "round4" in self.playoff_tree and len(self.playoff_tree["round4"]) > 0:
            final_result = all_results.get("round4", [{}])[0]
            champion_id = final_result.get("winner")
            final_matchup = self.playoff_tree["round4"][0]
            
            if champion_id == final_matchup['team1']['team_id']:
                champion = final_matchup['team1']
            else:
                champion = final_matchup['team2']
            
            all_results["champion"] = {
                "team_id": champion['team_id'],
                "team_name": champion['team_name']
            }
        
        self.sim_results = all_results
        return all_results

    def run_multiple_simulations(self, num_sims=1000):
        """Run multiple playoff simulations to calculate probabilities"""
        print(f"Running {num_sims} playoff simulations...")
        
        if self.teams_data is None:
            self.fetch_team_data()
        
        results = {
            'cup_wins': {},
            'finals_appearances': {},
            'conference_finals': {},
            'round2_appearances': {}
        }
        
        # Initialize counters for each team
        team_ids = self.teams_data['team_id'].unique()
        for team_id in team_ids:
            results['cup_wins'][team_id] = 0
            results['finals_appearances'][team_id] = 0
            results['conference_finals'][team_id] = 0
            results['round2_appearances'][team_id] = 0
        
        for i in tqdm(range(num_sims)):
            # Create fresh playoff bracket for each simulation
            self.playoff_tree = None
            self.create_playoff_bracket()
            sim_result = self.simulate_playoffs()
            
            # Record Stanley Cup winner
            if 'champion' in sim_result:
                champion_id = sim_result['champion']['team_id']
                results['cup_wins'][champion_id] = results['cup_wins'].get(champion_id, 0) + 1
            
            # Record Finals appearances
            if 'round4' in sim_result:
                for matchup in sim_result['round4']:
                    team1_id = matchup['matchup'].split(' vs ')[0]
                    team2_id = matchup['matchup'].split(' vs ')[1]
                    results['finals_appearances'][team1_id] = results['finals_appearances'].get(team1_id, 0) + 1
                    results['finals_appearances'][team2_id] = results['finals_appearances'].get(team2_id, 0) + 1
            
            # Record Conference Finals appearances
            if 'round3' in sim_result:
                for matchup in sim_result['round3']:
                    team1_id = matchup['matchup'].split(' vs ')[0]
                    team2_id = matchup['matchup'].split(' vs ')[1]
                    results['conference_finals'][team1_id] = results['conference_finals'].get(team1_id, 0) + 1
                    results['conference_finals'][team2_id] = results['conference_finals'].get(team2_id, 0) + 1
            
            # Record Round 2 appearances
            if 'round2' in sim_result:
                for matchup in sim_result['round2']:
                    team1_id = matchup['matchup'].split(' vs ')[0]
                    team2_id = matchup['matchup'].split(' vs ')[1]
                    results['round2_appearances'][team1_id] = results['round2_appearances'].get(team1_id, 0) + 1
                    results['round2_appearances'][team2_id] = results['round2_appearances'].get(team2_id, 0) + 1
        
        # Convert counts to percentages
        for category in results:
            for team_id in results[category]:
                results[category][team_id] = results[category][team_id] / num_sims * 100
        
        self.simulation_results = results
        return results

    def display_results(self, results=None):
        """Display simulation results in a readable format"""
        if results is None:
            results = self.simulation_results
        
        if results is None:
            print("No simulation results available. Run simulations first.")
            return
        
        # Get team names from team IDs
        team_names = {}
        for _, row in self.teams_data.iterrows():
            team_names[row['team_id']] = row['team_name']
        
        # Create DataFrame with all results
        data = []
        for team_id in results['cup_wins']:
            if team_id in team_names:
                data.append({
                    'Team': team_names[team_id],
                    'Stanley Cup %': results['cup_wins'].get(team_id, 0),
                    'Finals %': results['finals_appearances'].get(team_id, 0),
                    'Conf Finals %': results['conference_finals'].get(team_id, 0),
                    'Round 2 %': results['round2_appearances'].get(team_id, 0)
                })
        
        results_df = pd.DataFrame(data)
        
        # Sort by Stanley Cup win percentage
        results_df = results_df.sort_values(by='Stanley Cup %', ascending=False)
        
        return results_df

    def plot_results(self, results_df=None, top_n=10):
        """Plot simulation results"""
        if results_df is None:
            # Generate results DataFrame if not provided
            results_df = self.display_results()
        
        if results_df is None or len(results_df) == 0:
            print("No results to plot. Run simulations first.")
            return
        
        # Select top N teams for display
        plot_df = results_df.head(top_n)
        
        # Set up the plot
        plt.figure(figsize=(12, 8))
        sns.set_style("whitegrid")
        
        # Create grouped bar chart
        bar_width = 0.2
        index = np.arange(len(plot_df))
        
        plt.bar(index, plot_df['Stanley Cup %'], bar_width, 
                label='Stanley Cup Win', color='gold')
        plt.bar(index + bar_width, plot_df['Finals %'], bar_width,
                label='Finals Appearance', color='silver')
        plt.bar(index + 2*bar_width, plot_df['Conf Finals %'], bar_width,
                label='Conference Finals', color='#CD7F32')  # Bronze color
        plt.bar(index + 3*bar_width, plot_df['Round 2 %'], bar_width,
                label='Round 2', color='blue')
        
        # Add labels and title
        plt.xlabel('Team')
        plt.ylabel('Probability (%)')
        plt.title(f'NHL Playoff Simulation Results - Top {top_n} Teams')
        plt.xticks(index + 1.5*bar_width, plot_df['Team'], rotation=45, ha='right')
        plt.legend()
        plt.tight_layout()
        
        return plt

def main():
    simulator = NHLPlayoffSimulator()
    simulator.fetch_team_data()
    simulator.create_playoff_bracket()
    
    # Run a single simulation and display results
    print("Running a single playoff simulation...")
    single_sim = simulator.simulate_playoffs()
    
    # Run multiple simulations
    print("\nRunning multiple simulations to calculate probabilities...")
    simulator.run_multiple_simulations(num_sims=1000)
    
    # Display and plot results
    results_df = simulator.display_results()
    print("\nPlayoff Probabilities:")
    print(results_df)
    
    # Plot results
    plt = simulator.plot_results(results_df)
    plt.savefig('nhl_playoff_probabilities.png')
    plt.show()

if __name__ == "__main__":
    main()
