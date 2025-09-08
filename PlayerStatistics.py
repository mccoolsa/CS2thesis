import os
import pandas as pd
from demoparser2 import DemoParser
import numpy as np
from collections import defaultdict
import glob
import argparse
import gc
import math

# ---------------------------
# Resume capability functions
# ---------------------------

def get_processed_demos(output_file):
    """Read existing CSV to determine which demos have already been processed."""
    processed_demos = set()
    
    if not os.path.exists(output_file):
        return processed_demos
    
    try:
        existing_df = pd.read_csv(output_file)
        if 'demo_file' in existing_df.columns:
            processed_demos = set(existing_df['demo_file'].unique())
            print(f"Found {len(processed_demos)} already processed demo files")
    except Exception as e:
        print(f"Warning: Could not read existing CSV: {e}")
    
    return processed_demos

def save_progress(stats_list, output_file, append_mode=False):
    """Save current progress to CSV."""
    if not stats_list:
        return
        
    df = pd.DataFrame(stats_list)
    
    if append_mode and os.path.exists(output_file):
        df.to_csv(output_file, mode='a', header=False, index=False)
        print(f"✅ Appended {len(df)} player records to {output_file}")
    else:
        df.to_csv(output_file, index=False)
        print(f"✅ Saved {len(df)} player records to {output_file}")

# ---------------------------
# Swing Rating Calculation Functions
# ---------------------------

def calculate_equipment_value(inventory_str, armor_value=0):
    """Calculate total equipment value from inventory string and armor."""
    equipment_values = {
        # Rifles
        'ak47': 2700, 'awp': 4750, 'm4a4': 3100, 'm4a1_silencer': 2900,
        'famas': 2250, 'galil': 1800, 'aug': 3300, 'sg553': 3000,
        'scar20': 5000, 'g3sg1': 5000, 'ssg08': 1700,
        # SMGs
        'mp9': 1250, 'mac10': 1050, 'mp7': 1500, 'ump45': 1200,
        'p90': 2350, 'bizon': 1400, 'mp5sd': 1500,
        # Pistols  
        'glock': 200, 'usp_silencer': 200, 'p250': 300, 'fiveseven': 500,
        'tec9': 500, 'cz75a': 500, 'deagle': 700, 'revolver': 600,
        'dualberettas': 300,
        # Shotguns
        'nova': 1050, 'xm1014': 2000, 'sawedoff': 1100, 'mag7': 1300,
        # Grenades
        'he_grenade': 300, 'flashbang': 200, 'smokegrenade': 300,
        'incgrenade': 600, 'molotov': 400, 'decoy': 50,
        # Equipment
        'defuser': 400
    }
    
    total_value = 0
    
    # Add armor value
    if armor_value > 0:
        total_value += 650 if armor_value < 100 else 1000
    
    # Parse inventory if provided
    if inventory_str and isinstance(inventory_str, str):
        inventory_str = inventory_str.lower()
        for weapon, value in equipment_values.items():
            if weapon in inventory_str:
                total_value += value
    
    return total_value

def estimate_win_probability(team_alive_count, enemy_alive_count, team_equipment_avg, enemy_equipment_avg, 
                           side, map_name, time_remaining_pct=0.5):
    """
    Estimate win probability based on round state.
    Returns probability that the team wins the round (0.0 to 1.0).
    """
    
    # Base probability from player count advantage
    if team_alive_count == 0:
        return 0.0
    if enemy_alive_count == 0:
        return 1.0
    
    # Player advantage factor (exponential scaling)
    player_ratio = team_alive_count / enemy_alive_count
    player_advantage = (player_ratio - 1) * 0.3  # 30% swing per player advantage
    
    # Equipment advantage (normalized by typical values)
    equipment_diff = (team_equipment_avg - enemy_equipment_avg) / 1000  # Per $1000 difference
    equipment_advantage = equipment_diff * 0.1  # 10% per $1000 advantage
    
    # Side advantage (CT usually favored on most maps)
    side_advantage = 0.0
    if side == 'CT':
        side_advantage = 0.05  # 5% CT advantage
    elif side == 'T':
        side_advantage = -0.05  # 5% T disadvantage
    
    # Time factor (time pressure on T side)
    time_pressure = 0.0
    if side == 'T' and time_remaining_pct < 0.3:
        time_pressure = -0.1 * (0.3 - time_remaining_pct) / 0.3  # Up to -10% penalty
    elif side == 'CT' and time_remaining_pct < 0.3:
        time_pressure = 0.1 * (0.3 - time_remaining_pct) / 0.3   # Up to +10% bonus
    
    # Base probability (50/50)
    base_prob = 0.5
    
    # Combine all factors
    total_advantage = player_advantage + equipment_advantage + side_advantage + time_pressure
    
    # Apply sigmoid function to keep probability in [0, 1] range
    win_probability = 1 / (1 + math.exp(-total_advantage * 5))  # Sigmoid scaling
    
    # Clamp to reasonable bounds
    return max(0.05, min(0.95, win_probability))

def calculate_round_swing(events_df, ticks_df, player_teams, round_num, map_name, round_winner, 
                         player_stats, opening_kills_this_round, opening_deaths_this_round, multikill_players_this_round, total_rounds):
    """
    Calculate swing values for all players in a single round.
    Returns dict of player_id -> swing_value.
    """
    player_swings = defaultdict(float)
    
    # Filter events for this round
    round_events = events_df[events_df['round'] == round_num].copy()
    if round_events.empty:
        return player_swings
    
    # Sort events chronologically
    round_events = round_events.sort_values('tick')
    
    # Initialize round state
    team_alive = {2: set(), 3: set()}  # T=2, CT=3
    team_equipment = {2: [], 3: []}
    
    # Get all players and their teams for this round
    for steamid, team in player_teams.items():
        if team in [2, 3]:
            team_alive[team].add(steamid)
            
            # Get equipment value from ticks near round start
            player_ticks = ticks_df[
                (ticks_df['steamid'] == int(steamid)) & 
                (ticks_df['tick'] >= round_events['tick'].min() - 1000) &
                (ticks_df['tick'] <= round_events['tick'].min() + 1000)
            ]
            
            if not player_ticks.empty:
                tick_sample = player_ticks.iloc[0]
                inventory = tick_sample.get('inventory', '')
                armor = tick_sample.get('armor', 0)
                equipment_value = calculate_equipment_value(inventory, armor)
                team_equipment[team].append(equipment_value)
    
    # Calculate initial team equipment averages
    team_equipment_avg = {}
    for team in [2, 3]:
        team_equipment_avg[team] = np.mean(team_equipment[team]) if team_equipment[team] else 1000
    
    # Track probability changes throughout the round
    previous_prob = {}
    for team in [2, 3]:
        side = 'T' if team == 2 else 'CT'
        enemy_team = 3 if team == 2 else 2
        prob = estimate_win_probability(
            len(team_alive[team]), len(team_alive[enemy_team]),
            team_equipment_avg[team], team_equipment_avg[enemy_team],
            side, map_name, 1.0  # Start of round
        )
        previous_prob[team] = prob
    
    # Process each event in chronological order
    for _, event in round_events.iterrows():
        event_type = event.get('event_type', '')
        
        if event_type == 'player_death':
            victim_id = str(event.get('user_steamid', ''))
            attacker_id = str(event.get('attacker_steamid', ''))
            assister_id = str(event.get('assister_steamid', ''))
            
            # Remove victim from alive count
            victim_team = player_teams.get(victim_id, 0)
            if victim_team in team_alive:
                team_alive[victim_team].discard(victim_id)
            
            # Calculate new probabilities after this death
            new_prob = {}
            for team in [2, 3]:
                side = 'T' if team == 2 else 'CT'
                enemy_team = 3 if team == 2 else 2
                prob = estimate_win_probability(
                    len(team_alive[team]), len(team_alive[enemy_team]),
                    team_equipment_avg[team], team_equipment_avg[enemy_team],
                    side, map_name, 0.5  # Mid-round assumption
                )
                new_prob[team] = prob
            
            # Calculate swing for each team
            for team in [2, 3]:
                prob_change = new_prob[team] - previous_prob[team]
                
                # Attribute swing to players involved in this kill
                if attacker_id and attacker_id in player_teams and player_teams[attacker_id] == team:
                    # Positive swing for attacker's team
                    attacker_swing = prob_change * 100  # Convert to percentage
                    player_swings[attacker_id] += attacker_swing * 0.8  # 80% credit to attacker
                    
                    # Partial credit to assister
                    if assister_id and assister_id in player_teams and player_teams[assister_id] == team:
                        assist_swing = prob_change * 100 * 0.3  # 30% credit to assister
                        player_swings[assister_id] += assist_swing
                
                elif victim_id and victim_id in player_teams and player_teams[victim_id] == team:
                    # Negative swing for victim's team (victim gets heavily penalized)
                    victim_swing = prob_change * 100  # This will be negative
                    player_swings[victim_id] += victim_swing * 1.5  # 150% penalty to victim (much more severe)
            
            # Update previous probabilities
            previous_prob = new_prob
    
    # Add opening kill/death bonuses and penalties (with more severe death penalty)
    for player_id in opening_kills_this_round:
        if player_id in player_swings or player_id in player_stats:
            player_swings[player_id] += 50.0  # +50% swing for first kill
    
    for player_id in opening_deaths_this_round:
        if player_id in player_swings or player_id in player_stats:
            player_swings[player_id] -= 50.0  # -50% swing for first death
    
    # Add multi-kill bonuses
    for player_id in multikill_players_this_round:
        if player_id in player_swings or player_id in player_stats:
            player_swings[player_id] += 25.0  # +25% swing for multi-kill round

    # Apply K/D ratio penalty for players with poor death ratios
    # This ensures players with many deaths get properly penalized
    for player_id in player_stats:
        if player_id in player_swings:
            player_kills = player_stats[player_id]["kills"]
            player_deaths = player_stats[player_id]["deaths"]
            
            if player_deaths > 0:
                kd_ratio = player_kills / player_deaths
                if kd_ratio < 1.0:  # Negative K/D ratio
                    # Apply additional death penalty: -20% per round for each death over kills
                    excess_deaths = player_deaths - player_kills
                    death_penalty_per_round = -20.0 * (excess_deaths / max(30, 1))  # Assume max 30 rounds
                    player_swings[player_id] += death_penalty_per_round
    
    # Final adjustment based on actual round outcome
    actual_winner = round_winner
    if actual_winner in [2, 3]:
        # Adjust swings based on whether predictions were correct
        for team in [2, 3]:
            final_prob = previous_prob.get(team, 0.5)
            
            # If team won but had low probability, boost their players
            # If team lost but had high probability, penalize their players
            outcome_adjustment = 0
            if team == actual_winner:
                # Team won - reward based on how unlikely it was
                outcome_adjustment = (1.0 - final_prob) * 10  # Up to 10% bonus
            else:
                # Team lost - penalize based on how likely they were to win
                outcome_adjustment = -final_prob * 10  # Up to -10% penalty
            
            # Distribute adjustment among team members
            team_players = [pid for pid, pteam in player_teams.items() if pteam == team]
            if team_players:
                adjustment_per_player = outcome_adjustment / len(team_players)
                for player_id in team_players:
                    player_swings[player_id] += adjustment_per_player
    
    return dict(player_swings)

# ---------------------------
# Original parsing functions (enhanced with swing calculation)
# ---------------------------

def parse_demo_file(demo_path):
    """Parse a single demo file and extract player statistics."""
    print(f"Parsing: {demo_path}")
    
    try:
        parser = DemoParser(demo_path)
        
        # Parse events we need (including equipment/economy data)
        events = parser.parse_events([
            "player_death",
            "round_end", 
            "player_hurt",
            "weapon_fire",
            "flashbang_detonate",
            "item_pickup",
            "item_purchase"
        ])
        
        # Get ticks and player info (including equipment data)
        ticks = parser.parse_ticks(["X", "Y", "Z", "health", "armor", "active_weapon_name", "team_name", "name", "eye_angle_x", "eye_angle_y", "inventory"])
        
        # Debug: Print basic info about what we parsed
        print(f"  Events type: {type(events)}")
        print(f"  Ticks type: {type(ticks)}")
        
        # Handle different return types
        if isinstance(events, dict):
            print(f"  Events parsed: {len(events)}")
            for event_name, event_data in events.items():
                if hasattr(event_data, '__len__'):
                    print(f"    {event_name}: {len(event_data)} records")
        elif hasattr(events, '__len__'):
            print(f"  Events parsed: {len(events)}")
        
        if isinstance(ticks, dict):
            print(f"  Players found: {len(ticks)}")
        elif hasattr(ticks, '__len__'):
            print(f"  Ticks parsed: {len(ticks)}")
        
        # Get header info for map name
        header = parser.parse_header()
        map_name = header.get("map_name", "unknown")
        
        result = {
            "events": events,
            "ticks": ticks,
            "map_name": map_name,
            "demo_path": demo_path
        }
        
        # Clean up parser to help with memory
        del parser
        gc.collect()
        
        return result
        
    except Exception as e:
        print(f"Error parsing {demo_path}: {e}")
        import traceback
        traceback.print_exc()
        return None
    

def calculate_player_stats_single_demo(demo_data):
    """Calculate comprehensive player statistics from a single parsed demo."""
    
    def check_position_change(position_data, flash_tick, death_tick):
        """Check if player's position or view angle changed significantly after flash."""
        if len(position_data) < 2:
            return False
        
            
        # Find positions before and after flash
        pre_flash = None
        post_flash = None
        
        for tick, x, y, z, eye_x, eye_y in position_data:
            if tick <= flash_tick and (pre_flash is None or tick > pre_flash[0]):
                pre_flash = (tick, x, y, z, eye_x, eye_y)
            elif tick > flash_tick and tick <= death_tick and post_flash is None:
                post_flash = (tick, x, y, z, eye_x, eye_y)
                break
        
        if not pre_flash or not post_flash:
            return True  # Assume affected if we can't determine
            
        # Calculate position difference
        pos_diff = ((post_flash[1] - pre_flash[1])**2 + 
                   (post_flash[2] - pre_flash[2])**2 + 
                   (post_flash[3] - pre_flash[3])**2)**0.5
        
        # Calculate view angle difference 
        angle_diff = abs(post_flash[4] - pre_flash[4]) + abs(post_flash[5] - pre_flash[5])
        
        # Consider affected if moved >50 units or view changed >30 degrees
        return pos_diff > 50 or angle_diff > 30
    

    stats_list = []
    
    if demo_data is None:
        return stats_list
        
    events = demo_data["events"]
    ticks = demo_data["ticks"]
    map_name = demo_data["map_name"]
    demo_path = demo_data["demo_path"]
    
    print(f"\nProcessing {os.path.basename(demo_path)} on {map_name}")
    
    # Convert events list to DataFrames - fix the parsing order
    events_dict = {}
    
    for event_tuple in events:
        if len(event_tuple) >= 2:
            event_name = event_tuple[0]  # First element is the event name
            event_df = event_tuple[1]    # Second element is the DataFrame
            events_dict[event_name] = event_df
            print(f"  {event_name}: {len(event_df)} events")
    
    print(f"  Available events: {list(events_dict.keys())}")
    
    # Calculate rounds from round_end events FIRST - but fix for actual game rounds
    rounds_count = 22  # Default
    if "round_end" in events_dict and not events_dict["round_end"].empty:
        detected_rounds = len(events_dict["round_end"])
        # For competitive matches, the actual rounds played = max(round wins) where one team reaches 13
        # If we detect 22 round_end events, the game likely ended 13-8 = 21 actual rounds
        if detected_rounds == 22:
            rounds_count = 21  # Competitive match that ended 13-8 or similar
        else:
            rounds_count = detected_rounds
        print(f"  Detected {detected_rounds} round_end events -> Using {rounds_count} actual rounds")
    
    # Initialize player stats tracking
    player_stats = defaultdict(lambda: {
        "name": "",
        "kills": 0,
        "deaths": 0,
        "assists": 0,
        "flash_assists": 0,
        "damage_dealt": 0,
        "headshots": 0,
        "opening_kills": 0,
        "opening_deaths": 0,
        "rounds_with_kill": set(),
        "rounds_with_assist": set(),
        "rounds_survived": set(),
        "rounds_traded": set(),
        "multi_kills": 0,
        "clutches_won": 0,
        "clutches_lost": 0,  # Track clutch failures for negative impact
        "kast_rounds": 0,
        "equipment_value_destroyed": 0,  # Total equipment value of enemies killed
        "kills_for_equipment": 0,  # Number of kills for equipment calculation
        "equipment_value_lost": 0,  # Total equipment value lost when dying
        "deaths_with_equipment": 0,  # Number of deaths where equipment was lost
        "swing_total": 0.0,  # Total swing rating
        "rounds_won": 0,  # ADD THIS LINE
        "rounds_lost": 0,
        "map_name": map_name,
        "demo_file": os.path.basename(demo_path)
    })
    
    # Get unique players from ticks
    unique_players = ticks[['steamid', 'name']].drop_duplicates()
    for _, player in unique_players.iterrows():
        steamid = str(player['steamid'])  # Convert to string for consistency
        name = player['name']
        if pd.notna(name) and name.strip():
            player_stats[steamid]["name"] = name
    
    print(f"  Found {len(player_stats)} players")
    print(f"  Sample player Steam IDs: {list(player_stats.keys())[:3]}")
    
    # Get player teams EARLY (before processing deaths) - MOVED UP
    player_teams = {}
    for steamid in player_stats.keys():
        player_ticks_subset = ticks[ticks['steamid'] == int(steamid)]
        if not player_ticks_subset.empty:
            # Get most common team for this player
            team_counts = player_ticks_subset['team_name'].value_counts()
            if len(team_counts) > 0:
                most_common_team = team_counts.index[0]
                # Convert team names to winner codes
                if 'TERRORIST' in str(most_common_team).upper():
                    player_teams[steamid] = 2
                elif 'CT' in str(most_common_team).upper():
                    player_teams[steamid] = 3
                else:
                    player_teams[steamid] = 0
    
    print(f"  Player teams detected: {len(player_teams)}")
    
    # Process player_death events
    if "player_death" in events_dict and not events_dict["player_death"].empty:
        death_events = events_dict["player_death"]
        print(f"  Processing {len(death_events)} death events")
        print(f"  Death event columns: {list(death_events.columns)}")
        
        # Debug: Show sample Steam IDs from death events
        sample_attackers = death_events['attacker_steamid'].dropna().unique()[:3]
        sample_victims = death_events['user_steamid'].dropna().unique()[:3]
        print(f"  Sample attacker Steam IDs: {sample_attackers}")
        print(f"  Sample victim Steam IDs: {sample_victims}")
        
        # Check if Steam IDs match (convert both to strings)
        all_tick_steamids = set(str(sid) for sid in ticks['steamid'].unique())
        all_death_steamids = set(str(sid) for sid in death_events['attacker_steamid'].dropna()) | set(str(sid) for sid in death_events['user_steamid'].dropna())
        matching_steamids = all_tick_steamids & all_death_steamids
        print(f"  Matching Steam IDs between ticks and deaths: {len(matching_steamids)}")
        
        # Check if we have round data
        if 'round' in death_events.columns:
            print(f"  Round column found. Sample rounds: {death_events['round'].unique()[:10]}")
            print(f"  Round range: {death_events['round'].min()} to {death_events['round'].max()}")
            use_round_col = True
        else:
            print("  No round column found, will estimate from ticks")
            use_round_col = False
        
        # Better round estimation if needed
        if not use_round_col:
            # Get round_end events to map ticks to rounds
            if "round_end" in events_dict and not events_dict["round_end"].empty:
                round_end_events = events_dict["round_end"]
                if 'tick' in round_end_events.columns:
                    round_end_ticks = round_end_events[['tick']].copy()
                    round_end_ticks['round'] = range(len(round_end_ticks))
                    round_end_ticks = round_end_ticks.sort_values('tick')
                    
                    # Assign rounds based on tick ranges
                    death_events = death_events.copy()
                    death_events['round'] = pd.cut(death_events['tick'], 
                                                 bins=[-float('inf')] + round_end_ticks['tick'].tolist() + [float('inf')], 
                                                 labels=list(range(len(round_end_ticks) + 1)),
                                                 include_lowest=True).astype(int)
                    use_round_col = True
                    print(f"  Estimated rounds from round_end ticks. Sample rounds: {death_events['round'].unique()[:10]}")
        
        if not use_round_col:
            # Fallback: simple tick-based estimation
            death_events = death_events.copy()
            tick_range = death_events['tick'].max() - death_events['tick'].min()
            tick_per_round = tick_range / rounds_count
            death_events['round'] = ((death_events['tick'] - death_events['tick'].min()) / tick_per_round).astype(int)
            print(f"  Fallback round estimation. Sample rounds: {death_events['round'].unique()[:10]}")
        
        # Get round winners from round_end events
        round_winners = {}  # round -> winner (2=T, 3=CT)
        if "round_end" in events_dict and not events_dict["round_end"].empty:
            round_end_events = events_dict["round_end"]
            print(f"  Round end columns: {list(round_end_events.columns)}")
            
            # Check different possible column names for winner
            winner_col = None
            if 'winner' in round_end_events.columns:
                winner_col = 'winner'
            elif 'reason' in round_end_events.columns:
                winner_col = 'reason'
            
            if winner_col:
                for idx, round_end in round_end_events.iterrows():
                    round_num = idx  # Assuming round_end events are in order
                    winner_raw = round_end.get(winner_col, 0)
                    
                    # Convert string winners to integer codes
                    if winner_raw == 'T' or winner_raw == 'TERRORIST' or winner_raw == 2:
                        round_winners[round_num] = 2
                    elif winner_raw == 'CT' or winner_raw == 'Counter-Terrorist' or winner_raw == 3:
                        round_winners[round_num] = 3
                    else:
                        # Try to parse as integer
                        try:
                            winner_int = int(winner_raw) if winner_raw else 0
                            if winner_int in [2, 3]:
                                round_winners[round_num] = winner_int
                        except:
                            pass
            else:
                print(f"  Warning: No winner column found in round_end events")
        
        # Group deaths by round to calculate opening kills/deaths and multi-kills
        deaths_by_round = death_events.groupby('round')
        opening_kills_count = 0
        opening_deaths_count = 0
        
        # Track opening kills/deaths and multi-kills per round for swing calculation
        opening_kills_by_round = {}  # round -> [player_ids]
        opening_deaths_by_round = {}  # round -> [player_ids]
        multikill_by_round = {}  # round -> [player_ids]
        
        # Calculate opening kills/deaths (first kill of each round)
        for round_num, round_deaths in deaths_by_round:
            opening_kills_by_round[round_num] = []
            opening_deaths_by_round[round_num] = []
            
            if len(round_deaths) > 0:
                # Sort by tick time to get first death
                round_deaths_sorted = round_deaths.sort_values('tick')
                first_death = round_deaths_sorted.iloc[0]
                
                first_attacker = str(first_death.get("attacker_steamid")) if pd.notna(first_death.get("attacker_steamid")) else None
                first_victim = str(first_death.get("user_steamid")) if pd.notna(first_death.get("user_steamid")) else None
                
                if first_attacker and first_attacker in player_stats:
                    player_stats[first_attacker]["opening_kills"] += 1
                    opening_kills_count += 1
                    opening_kills_by_round[round_num].append(first_attacker)
                if first_victim and first_victim in player_stats:
                    player_stats[first_victim]["opening_deaths"] += 1
                    opening_deaths_count += 1
                    opening_deaths_by_round[round_num].append(first_victim)
        
        print(f"  Opening kills detected: {opening_kills_count}, Opening deaths: {opening_deaths_count}")
        
        # Calculate multi-kills per round
        kill_counts_per_round = defaultdict(lambda: defaultdict(int))
        total_multikills = 0
        
        total_flash_assists = 0
        equipment_values = {
            # Weapons
            'ak47': 2700, 'awp': 4750, 'm4a4': 3100, 'm4a1_silencer': 2900,
            'famas': 2250, 'galil': 1800, 'aug': 3300, 'sg553': 3000,
            'scar20': 5000, 'g3sg1': 5000, 'ssg08': 1700,
            # SMGs
            'mp9': 1250, 'mac10': 1050, 'mp7': 1500, 'ump45': 1200,
            'p90': 2350, 'bizon': 1400, 'mp5sd': 1500,
            # Pistols  
            'glock': 200, 'usp_silencer': 200, 'p250': 300, 'fiveseven': 500,
            'tec9': 500, 'cz75a': 500, 'deagle': 700, 'revolver': 600,
            'dualberettas': 300,
            # Shotguns
            'nova': 1050, 'xm1014': 2000, 'sawedoff': 1100, 'mag7': 1300,
            # Equipment
            'kevlar': 650, 'assaultsuit': 1000, 'defuser': 400,
            'he_grenade': 300, 'flashbang': 200, 'smokegrenade': 300,
            'incgrenade': 600, 'molotov': 400, 'decoy': 50
        }
        
        def get_weapon_value(weapon_name):
            """Get the equipment value for a weapon."""
            if not weapon_name:
                return 0
                
            # Clean weapon name
            clean_name = str(weapon_name).lower().replace('weapon_', '')
            return equipment_values.get(clean_name, 0)
        
        # Process all deaths for basic stats and equipment values
        for _, death in death_events.iterrows():
            attacker_id = str(death.get("attacker_steamid")) if pd.notna(death.get("attacker_steamid")) else None
            victim_id = str(death.get("user_steamid")) if pd.notna(death.get("user_steamid")) else None
            assister_id = str(death.get("assister_steamid")) if pd.notna(death.get("assister_steamid")) else None
            headshot = death.get("headshot", False)  # Use actual headshot column
            round_num = death.get("round", 0)
            tick = death.get("tick", 0)
            weapon_used = death.get("weapon", "")
            
            # Victim stats
            if victim_id and victim_id in player_stats:
                player_stats[victim_id]["deaths"] += 1
                
                # Track equipment value lost when dying
                victim_ticks = ticks[ticks['steamid'] == int(victim_id)]
                if not victim_ticks.empty:
                    # Find tick closest to death
                    tick_diff = abs(victim_ticks['tick'] - tick)
                    closest_tick_idx = tick_diff.idxmin()
                    closest_tick = victim_ticks.loc[closest_tick_idx]
                    
                    victim_weapon = closest_tick.get('active_weapon_name', '')
                    victim_armor = closest_tick.get('armor', 0)
                    victim_inventory = closest_tick.get('inventory', '')
                    
                    victim_equipment_value = calculate_equipment_value(victim_inventory, victim_armor)
                    if victim_equipment_value == 0:
                        # Fallback: just weapon and armor
                        victim_equipment_value = get_weapon_value(victim_weapon)
                        if victim_armor > 0:
                            victim_equipment_value += 650 if victim_armor < 100 else 1000
                    
                    # Only count significant equipment losses (>$500)
                    if victim_equipment_value > 500:
                        player_stats[victim_id]["equipment_value_lost"] += victim_equipment_value
                        player_stats[victim_id]["deaths_with_equipment"] += 1
            
            # Attacker stats
            if attacker_id and attacker_id != victim_id and attacker_id in player_stats:
                player_stats[attacker_id]["kills"] += 1
                player_stats[attacker_id]["rounds_with_kill"].add(round_num)
                kill_counts_per_round[round_num][attacker_id] += 1
                
                if headshot:
                    player_stats[attacker_id]["headshots"] += 1
                
                # Calculate equipment value destroyed - NOW player_teams is defined
                if victim_id and victim_id in player_teams and attacker_id in player_teams:
                    victim_team = player_teams.get(victim_id, 0)
                    attacker_team = player_teams.get(attacker_id, 0)
                    
                    # Only count equipment value for enemy kills
                    if victim_team != attacker_team and victim_team in [2, 3] and attacker_team in [2, 3]:
                        # Estimate victim's equipment value
                        # Try to get victim's weapon from tick data around death time
                        victim_equipment_value = 0
                        
                        # Look for victim's weapon in nearby tick data
                        victim_ticks = ticks[ticks['steamid'] == int(victim_id)]
                        if not victim_ticks.empty:
                            # Find tick closest to death
                            tick_diff = abs(victim_ticks['tick'] - tick)
                            closest_tick_idx = tick_diff.idxmin()
                            closest_tick = victim_ticks.loc[closest_tick_idx]
                            
                            victim_weapon = closest_tick.get('active_weapon_name', '')
                            victim_equipment_value += get_weapon_value(victim_weapon)
                            
                            # Add armor value if they had armor
                            armor = closest_tick.get('armor', 0)
                            if armor > 0:
                                victim_equipment_value += 650 if armor < 100 else 1000
                        
                        # If we couldn't determine equipment, use average values
                        if victim_equipment_value == 0:
                            victim_equipment_value = 2000  # Average equipment value estimate
                        
                        player_stats[attacker_id]["equipment_value_destroyed"] += victim_equipment_value
                        player_stats[attacker_id]["kills_for_equipment"] += 1
            
            # Assister stats
            if assister_id and assister_id != attacker_id and assister_id in player_stats:
                player_stats[assister_id]["assists"] += 1
                player_stats[assister_id]["rounds_with_assist"].add(round_num)
                
                # Check for flash assists - use the assistedflash flag
                if death.get("assistedflash", False):
                    player_stats[assister_id]["flash_assists"] += 1
                    total_flash_assists += 1
        
        print(f"  Total flash assists detected: {total_flash_assists}")
        print(f"  Equipment value calculation included")
        
        # Calculate multi-kills (2+ kills in same round)
        for round_num, round_kills in kill_counts_per_round.items():
            multikill_by_round[round_num] = []
            for player_id, kills in round_kills.items():
                if kills >= 2 and player_id in player_stats:
                    player_stats[player_id]["multi_kills"] += 1
                    total_multikills += 1
                    multikill_by_round[round_num].append(player_id)
        
        # ============================================================================
        # CALCULATE SWING RATINGS
        # ============================================================================
        print(f"  Calculating swing ratings for {len(player_stats)} players...")
        
        # Prepare combined events DataFrame with event types for swing calculation
        combined_events = []
        
        # Add death events
        death_events_copy = death_events.copy()
        death_events_copy['event_type'] = 'player_death'
        combined_events.append(death_events_copy)
        
        # Combine all events
        if combined_events:
            all_events_df = pd.concat(combined_events, ignore_index=True)
            
            # Calculate swing for each round
            total_swing_calculated = 0
            for round_num in range(rounds_count):
                if round_num in round_winners:
                    # Get round-specific data for swing calculation
                    opening_kills_this_round = opening_kills_by_round.get(round_num, [])
                    opening_deaths_this_round = opening_deaths_by_round.get(round_num, [])
                    multikill_players_this_round = multikill_by_round.get(round_num, [])
                    
                    #Overall calculate swing method
                    round_swings = calculate_round_swing(
                    all_events_df, ticks, player_teams, round_num, map_name, round_winners[round_num],
                    player_stats, opening_kills_this_round, opening_deaths_this_round, multikill_players_this_round, rounds_count
                    )
                    
                    # Add to player totals
                    for player_id, swing_value in round_swings.items():
                        if player_id in player_stats:
                            player_stats[player_id]["swing_total"] += swing_value
                            total_swing_calculated += abs(swing_value)
            
            print(f"  Swing calculation complete. Total swing magnitude: {total_swing_calculated:.1f}")
            
            # Apply additional swing modifiers based on round outcomes and flash assists
            print(f"  Applying additional swing modifiers...")
            
            for steamid, stats in player_stats.items():
                if stats["name"]:
                    # Flash assists bonus: +25% swing per flash assist per round
                    flash_assists_per_round = stats["flash_assists"] / rounds_count if rounds_count > 0 else 0
                    flash_bonus = flash_assists_per_round * 25.0  # +10% per flash assist per round
                    stats["swing_total"] += flash_bonus

                      # Clutch bonuses/penalties: equal value to opening kills/deaths (+50/-50)
                    clutch_won_bonus = stats["clutches_won"] * 50.0  # +50% per clutch won
                    clutch_lost_penalty = stats["clutches_lost"] * -50.0  # -50% per clutch lost
                    stats["swing_total"] += clutch_won_bonus + clutch_lost_penalty
        
                    # Traded death bonus: +33% per traded death (helps team momentum)
                    traded_death_count = len(stats["rounds_traded"])
                    traded_bonus = traded_death_count * 33.3  # +33.3% swing per traded death (proportional change)
                    stats["swing_total"] += traded_bonus
                    
                    # Kills on rounds won bonus and deaths on rounds lost penalty
                    kills_on_won_rounds = 0
                    deaths_on_lost_rounds = 0
                    
                    player_team = player_teams.get(steamid, 0)
                    if player_team in [2, 3]:
                        # Count kills on rounds this player's team won
                        for round_num, round_winner in round_winners.items():
                            if round_winner == player_team:
                                # This player's team won this round
                                kills_this_round = 0
                                for _, death in death_events.iterrows():
                                    if (death.get("round", -1) == round_num and 
                                        str(death.get("attacker_steamid")) == steamid):
                                        kills_this_round += 1
                                kills_on_won_rounds += kills_this_round
                            else:
                                # This player's team lost this round
                                deaths_this_round = 0
                                for _, death in death_events.iterrows():
                                    if (death.get("round", -1) == round_num and 
                                        str(death.get("user_steamid")) == steamid):
                                        deaths_this_round += 1
                                deaths_on_lost_rounds += deaths_this_round
                    
                    # Apply bonuses/penalties
                    kills_won_bonus = (kills_on_won_rounds / rounds_count) * 25.0 if rounds_count > 0 else 0  # +25% per kill on won round per round
                    deaths_lost_penalty = (deaths_on_lost_rounds / rounds_count) * -30.0 if rounds_count > 0 else 0  # -30% per death on lost round per round
                    
                    stats["swing_total"] += kills_won_bonus + deaths_lost_penalty
                    
                    # Debug for first few players
                    if list(player_stats.keys()).index(steamid) < 2:
                        print(f"    {stats['name']}: flash_bonus={flash_bonus:.1f}%, kills_won_bonus={kills_won_bonus:.1f}%, deaths_lost_penalty={deaths_lost_penalty:.1f}%")
            
            print(f"  Additional swing modifiers applied")
            
            # Debug: Show swing values for top players
            swing_values = [(pid, stats["swing_total"]) for pid, stats in player_stats.items() if stats["name"]]
            swing_values.sort(key=lambda x: abs(x[1]), reverse=True)
            for i, (pid, swing) in enumerate(swing_values[:3]):
                player_name = player_stats[pid]["name"]
                print(f"    {player_name}: {swing:.1f}% swing")
        
        # Continue with KAST calculation and other existing code...
        
        # Calculate KAST properly - track survival and trades with correct rules
        print(f"  Calculating KAST (Kill, Assist, Survive+Win, Trade)...")
        
        print(f"  Round winners detected: {len(round_winners)}")
        print(f"  Sample round winners (converted): {dict(list(round_winners.items())[:5])}")
        
        # Simplified flash assist detection - use the basic method but improve it
        print(f"  Using improved flash assist detection...")
        
        # Reset flash assists and use a combination of basic + enhanced detection
        for steamid in player_stats:
            player_stats[steamid]["flash_assists"] = 0
        
        # Method 1: Use the assistedflash flag from death events (basic)
        basic_flash_assists = 0
        for _, death in death_events.iterrows():
            assister_id = str(death.get("assister_steamid")) if pd.notna(death.get("assister_steamid")) else None
            if assister_id and assister_id in player_stats:
                if death.get("assistedflash", False):
                    player_stats[assister_id]["flash_assists"] += 1
                    basic_flash_assists += 1
        
        # Method 2: Enhanced detection - flashbang throws followed by deaths
        enhanced_flash_assists = 0
        flash_assist_window = 5 * 64  # 5 seconds at 64 tick
        
        # Use flashbang detonations (more reliable than throws)
        if "flashbang_detonate" in events_dict and not events_dict["flashbang_detonate"].empty:
            detonate_events = events_dict["flashbang_detonate"]
            
            for _, detonate in detonate_events.iterrows():
                thrower_id = str(detonate.get("user_steamid")) if pd.notna(detonate.get("user_steamid")) else None
                detonate_tick = detonate.get("tick", 0)
                detonate_round = detonate.get("round", 0)
                
                if not thrower_id or thrower_id not in player_teams:
                    continue
                    
                thrower_team = player_teams[thrower_id]
                
                # Look for enemy deaths within 5 seconds of this flashbang detonation
                deaths_after_flash = 0
                for _, death in death_events.iterrows():
                    death_tick = death.get("tick", 0)
                    death_round = death.get("round", 0)
                    victim_id = str(death.get("user_steamid")) if pd.notna(death.get("user_steamid")) else None
                    attacker_id = str(death.get("attacker_steamid")) if pd.notna(death.get("attacker_steamid")) else None
                    
                    # Check if death is within window, same round, and involves enemies
                    if (death_tick > detonate_tick and 
                        death_tick - detonate_tick <= flash_assist_window and
                        victim_id and victim_id in player_teams and
                        attacker_id and attacker_id in player_teams):
                        
                        victim_team = player_teams[victim_id]
                        attacker_team = player_teams[attacker_id]
                        
                        # Award flash assist if:
                        # - Victim was enemy of flash thrower
                        # - Attacker was teammate of flash thrower (or the thrower themselves)
                        # - Not already flagged with assistedflash
                        if (victim_team != thrower_team and 
                            attacker_team == thrower_team and
                            not death.get("assistedflash", False)):
                            
                            player_stats[thrower_id]["flash_assists"] += 1
                            enhanced_flash_assists += 1
                            deaths_after_flash += 1
                            
                            if deaths_after_flash >= 2:  # Limit to 2 assists per flash
                                break
        
        total_flash_assists = basic_flash_assists + enhanced_flash_assists
        print(f"  Flash assists: {basic_flash_assists} basic + {enhanced_flash_assists} enhanced = {total_flash_assists} total")
        
        # Track who died in each round and when (for trade detection)
        round_deaths = defaultdict(list)  # round -> [(steamid, tick, killer_steamid)]
        
        for _, death in death_events.iterrows():
            victim_id = str(death.get("user_steamid")) if pd.notna(death.get("user_steamid")) else None
            attacker_id = str(death.get("attacker_steamid")) if pd.notna(death.get("attacker_steamid")) else None
            round_num = death.get("round", 0)
            tick = death.get("tick", 0)
            
            if victim_id:
                round_deaths[round_num].append((victim_id, tick, attacker_id))
        
        # Calculate proper KAST components
        all_players = set(player_stats.keys())
        actual_rounds_played = rounds_count
        
        # Reset survival and trade tracking
        for steamid in player_stats:
            player_stats[steamid]["rounds_survived"] = set()
            player_stats[steamid]["rounds_traded"] = set()
        
        # Fix missing round 0 winner - assume it exists
        if 0 not in round_winners and len(round_winners) > 0:
            # Guess round 0 winner based on pattern or assume CT
            round_winners[0] = 3  # Default to CT for round 0
        
        for round_num in range(actual_rounds_played):
            # Who died this round
            died_this_round = {death[0] for death in round_deaths[round_num]}
            # Who survived this round
            survivors_this_round = all_players - died_this_round
            
            # Check if round had a winner
            round_winner = round_winners.get(round_num, 0)
        if round_winner in [2, 3]:  # Valid winner
            for steamid in player_stats:
                if steamid in player_teams:
                    player_team = player_teams[steamid]
                    if player_team == round_winner:
                        player_stats[steamid]["rounds_won"] += 1
                elif player_team in [2, 3]:  # Player was on losing team
                    player_stats[steamid]["rounds_lost"] += 1

                print(f"  Rounds won/lost tracking complete")
            
            # Award survival points only if player survived AND their team won
            if round_winner in [2, 3]:  # Valid winner
                for survivor_id in survivors_this_round:
                    if survivor_id in player_stats and survivor_id in player_teams:
                        player_team = player_teams[survivor_id]
                        if player_team == round_winner:  # Player's team won
                            player_stats[survivor_id]["rounds_survived"].add(round_num)
            
            # Calculate traded deaths - teammate killed your killer within 5 seconds
            round_death_list = round_deaths[round_num]
            if len(round_death_list) >= 2:
                deaths_sorted = sorted(round_death_list, key=lambda x: x[1])  # Sort by tick
                trade_window_ticks = 5 * 64  # 5 seconds at 64 ticks per second
                
                for i, (victim_id, victim_tick, killer_id) in enumerate(deaths_sorted):
                    if not killer_id or victim_id not in player_teams:
                        continue
                        
                    victim_team = player_teams[victim_id]
                    
                    # Look for teammate killing the SPECIFIC killer within 5 seconds
                    for j in range(i + 1, len(deaths_sorted)):
                        later_victim_id, later_tick, later_killer_id = deaths_sorted[j]
                        
                        # Check if trade window exceeded
                        if later_tick - victim_tick > trade_window_ticks:
                            break
                            
                        # TRADED DEATH: teammate killed the specific killer
                        if (later_victim_id == killer_id and  # Killed the exact killer
                            later_killer_id in player_teams and 
                            player_teams[later_killer_id] == victim_team):  # By a teammate
                            player_stats[victim_id]["rounds_traded"].add(round_num)
                            break  # Found the trade, stop looking
        
        # Debug KAST components
        print(f"  KAST Debug - Total rounds: {actual_rounds_played}")
        print(f"  Round winners available: {len(round_winners)}")
        print(f"  Sample round winners: {dict(list(round_winners.items())[:5])}")
        
        for i, (steamid, stats) in enumerate(player_stats.items()):
            if stats["name"]:
                kills_rounds = len(stats["rounds_with_kill"])
                assist_rounds = len(stats["rounds_with_assist"])
                survive_rounds = len(stats["rounds_survived"])
                trade_rounds = len(stats["rounds_traded"])
                
                # KAST = unique rounds where player had K, A, S, or T
                kast_rounds = len(stats["rounds_with_kill"] | stats["rounds_with_assist"] | 
                                stats["rounds_survived"] | stats["rounds_traded"])
                
                stats["kast_rounds"] = kast_rounds
                
                if i < 1:  # Debug first player only to reduce output
                    print(f"    Debug KAST for {stats['name']}:")
                    print(f"      K rounds: {kills_rounds}")
                    print(f"      A rounds: {assist_rounds}")  
                    print(f"      S rounds (survive+win): {survive_rounds}")
                    print(f"      T rounds (traded): {trade_rounds}")
                    print(f"      Total KAST rounds: {kast_rounds}/{actual_rounds_played} = {(kast_rounds/actual_rounds_played)*100:.1f}%")
        
        print(f"  KAST calculation complete")
        
        # Calculate clutch situations (1vsX rounds - both WON and LOST)
        print(f"  Calculating clutch rounds (1vsX wins and losses)...")
        
        clutch_rounds_won = defaultdict(int)
        clutch_rounds_lost = defaultdict(int)
        
        for round_num in range(actual_rounds_played):
            round_winner = round_winners.get(round_num, 0)
            if round_winner not in [2, 3]:
                continue
                
            # Get all deaths in this round, sorted by time
            round_death_list = round_deaths[round_num]
            if len(round_death_list) < 2:
                continue
                
            deaths_sorted = sorted(round_death_list, key=lambda x: x[1])  # Sort by tick
            
            # Track who's alive throughout the round for each team
            team_alive = {2: set(), 3: set()}  # T and CT
            
            # Initialize with all players
            for steamid in player_stats:
                if steamid in player_teams:
                    team = player_teams[steamid]
                    if team in [2, 3]:
                        team_alive[team].add(steamid)
            
            # Process deaths in chronological order
            potential_clutcher = None
            clutch_team = None
            
            for victim_id, tick, killer_id in deaths_sorted:
                if victim_id in player_teams:
                    victim_team = player_teams[victim_id]
                    if victim_team in team_alive:
                        team_alive[victim_team].discard(victim_id)
                
                # Check if this creates a 1vsX situation
                alive_counts = {team: len(players) for team, players in team_alive.items()}
                
                # Look for 1vsX situations (one player left vs multiple opponents)
                if alive_counts[2] == 1 and alive_counts[3] > 1:  # 1 T vs X CT
                    potential_clutcher = list(team_alive[2])[0]
                    clutch_team = 2
                elif alive_counts[3] == 1 and alive_counts[2] > 1:  # 1 CT vs X T
                    potential_clutcher = list(team_alive[3])[0]
                    clutch_team = 3
            
            # Award clutch win or loss based on round outcome
            if potential_clutcher and potential_clutcher in player_stats:
                if clutch_team == round_winner:
                    player_stats[potential_clutcher]["clutches_won"] += 1
                    clutch_rounds_won[potential_clutcher] += 1
                else:
                    player_stats[potential_clutcher]["clutches_lost"] += 1
                    clutch_rounds_lost[potential_clutcher] += 1
        
        total_clutches = sum(clutch_rounds_won.values()) + sum(clutch_rounds_lost.values())
        total_wins = sum(clutch_rounds_won.values())
        total_losses = sum(clutch_rounds_lost.values())
        print(f"  Clutch rounds detected: {total_clutches} total ({total_wins} wins, {total_losses} losses)")
    
    # Process player_hurt events for damage - use health difference method
    if "player_hurt" in events_dict and not events_dict["player_hurt"].empty:
        hurt_events = events_dict["player_hurt"]
        print(f"  Processing {len(hurt_events)} hurt events")
        
        # Track player health state throughout the game (for damage taken calculation)
        player_health_tracker = {}
        player_damage_taken = defaultdict(int)  # Track damage taken by each player
        
        # Initialize all players with 100 HP
        for steamid in player_stats.keys():
            player_health_tracker[steamid] = 100
        
        # Sort events by tick to process chronologically
        hurt_events_sorted = hurt_events.sort_values('tick')
        
        # Also need to track deaths to reset health
        death_events_sorted = events_dict["player_death"].sort_values('tick') if "player_death" in events_dict else pd.DataFrame()
        
        # Combine hurt and death events and sort by tick
        all_events = []
        
        for _, hurt in hurt_events_sorted.iterrows():
            all_events.append(('hurt', hurt))
            
        for _, death in death_events_sorted.iterrows():
            all_events.append(('death', death))
        
        # Sort all events by tick
        all_events.sort(key=lambda x: x[1]['tick'])
        
        total_damage_dealt = 0
        total_damage_taken = 0
        damage_events_processed = 0
        
        for event_type, event_data in all_events:
            if event_type == 'hurt':
                attacker_id = str(event_data.get("attacker_steamid")) if pd.notna(event_data.get("attacker_steamid")) else None
                victim_id = str(event_data.get("user_steamid")) if pd.notna(event_data.get("user_steamid")) else None
                
                if victim_id and victim_id in player_health_tracker:
                    health_before = player_health_tracker[victim_id]
                    health_after = max(0, event_data.get("health", 0))
                    
                    # Calculate actual health damage
                    actual_damage = health_before - health_after
                    actual_damage = max(0, min(actual_damage, 100))  # Clamp to 0-100
                    
                    # Update victim's health
                    player_health_tracker[victim_id] = health_after
                    
                    # Track damage taken by victim (from enemies only)
                    if (attacker_id and attacker_id != victim_id and
                        attacker_id in player_teams and victim_id in player_teams and
                        attacker_id in player_stats and actual_damage > 0):
                        
                        attacker_team = player_teams[attacker_id]
                        victim_team = player_teams[victim_id]
                        
                        # Only count damage between different teams
                        if attacker_team != victim_team and attacker_team in [2, 3] and victim_team in [2, 3]:
                            # Damage dealt (for attacker)
                            player_stats[attacker_id]["damage_dealt"] += actual_damage
                            total_damage_dealt += actual_damage
                            
                            # Damage taken (for victim)
                            player_damage_taken[victim_id] += actual_damage
                            total_damage_taken += actual_damage
                            
                            damage_events_processed += 1
            
            elif event_type == 'death':
                # Reset victim's health to 100 when they die
                victim_id = str(event_data.get("user_steamid")) if pd.notna(event_data.get("user_steamid")) else None
                if victim_id and victim_id in player_health_tracker:
                    player_health_tracker[victim_id] = 100
        
        print(f"  Damage calculation: {damage_events_processed} damage events processed")
        print(f"  Total damage dealt: {total_damage_dealt}, Total damage taken: {total_damage_taken}")
        avg_damage_per_event = total_damage_dealt / max(damage_events_processed, 1)
        print(f"  Average damage per hit: {avg_damage_per_event:.1f}")
    
    print(f"  Total rounds used for calculations: {rounds_count}")
    
    # Convert to final stats format matching HLTV
    for steamid, stats in player_stats.items():
        adr = stats["damage_dealt"] / rounds_count if rounds_count > 0 else 0
        damage_taken = player_damage_taken.get(steamid, 0)
        adr_taken = damage_taken / rounds_count if rounds_count > 0 else 0
        adr_diff = adr - adr_taken
        
        # ADR difference swing: +/- 1% swing per 10 ADR difference
        adr_diff_swing = (adr_diff / 10.0) * 1.0
        stats["swing_total"] += adr_diff_swing
        
        # Flash assists bonus: +10% swing per flash assist per round
        flash_assists_per_round = stats["flash_assists"] / rounds_count if rounds_count > 0 else 0
        flash_bonus = flash_assists_per_round * 10.0
        stats["swing_total"] += flash_bonus
        
        # Clutch bonuses/penalties: equal value to opening kills/deaths (+50/-50)
        clutch_won_bonus = stats["clutches_won"] * 50.0
        clutch_lost_penalty = stats["clutches_lost"] * -50.0
        stats["swing_total"] += clutch_won_bonus + clutch_lost_penalty
        
        # Traded death bonus: +25% per traded death (helps team momentum)
        traded_death_count = len(stats["rounds_traded"])
        traded_bonus = traded_death_count * 25.0
        stats["swing_total"] += traded_bonus
        
        # Round-weighted kills and deaths impact
        rounds_won = stats["rounds_won"]
        rounds_lost = stats["rounds_lost"]
        total_rounds_played = rounds_won + rounds_lost
        
        if total_rounds_played > 0:
            # Win rate impact: +/- swing based on how much they won vs lost
            win_rate = rounds_won / total_rounds_played
            win_rate_swing = (win_rate - 0.5) * 30.0  # +/-15% max swing based on win rate
            stats["swing_total"] += win_rate_swing
            
            # Weighted kills: kills on won rounds are more valuable
            kills_on_won_rounds = 0
            kills_on_lost_rounds = 0
            deaths_on_won_rounds = 0  
            deaths_on_lost_rounds = 0
            
            player_team = player_teams.get(steamid, 0)
            if player_team in [2, 3]:
                for round_num, round_winner in round_winners.items():
                    # Count kills/deaths for this player in this round
                    round_kills = 0
                    round_deaths = 0
                    
                    for _, death in death_events.iterrows():
                        if death.get("round", -1) == round_num:
                            if str(death.get("attacker_steamid")) == steamid:
                                round_kills += 1
                            if str(death.get("user_steamid")) == steamid:
                                round_deaths += 1
                    
                    # Categorize based on round outcome
                    if round_winner == player_team:  # Won round
                        kills_on_won_rounds += round_kills
                        deaths_on_won_rounds += round_deaths
                    elif round_winner in [2, 3]:  # Lost round
                        kills_on_lost_rounds += round_kills
                        deaths_on_lost_rounds += round_deaths
            
            # Apply round-weighted bonuses/penalties
            # Kills on won rounds: +30% per kill per round
            won_round_kill_bonus = (kills_on_won_rounds / rounds_count) * 30.0 if rounds_count > 0 else 0
            # Kills on lost rounds: +10% per kill per round (less valuable but still positive)
            lost_round_kill_bonus = (kills_on_lost_rounds / rounds_count) * 10.0 if rounds_count > 0 else 0
            # Deaths on won rounds: -15% per death per round (less severe since team won)
            won_round_death_penalty = (deaths_on_won_rounds / rounds_count) * -15.0 if rounds_count > 0 else 0
            # Deaths on lost rounds: -40% per death per round (more severe since team lost)
            lost_round_death_penalty = (deaths_on_lost_rounds / rounds_count) * -40.0 if rounds_count > 0 else 0
            
            total_round_weighted_impact = (won_round_kill_bonus + lost_round_kill_bonus + 
                                         won_round_death_penalty + lost_round_death_penalty)
            stats["swing_total"] += total_round_weighted_impact
            
            # Debug for first few players
            if list(player_stats.keys()).index(steamid) < 2:
                print(f"    {stats['name']}:")
                print(f"      ADR_diff: {adr_diff:.1f} -> {adr_diff_swing:.1f}% swing")
                print(f"      Win rate: {win_rate:.3f} -> {win_rate_swing:.1f}% swing") 
                print(f"      Round-weighted: {total_round_weighted_impact:.1f}% swing")
                print(f"      Flash: {flash_bonus:.1f}%, Clutch: {(clutch_won_bonus + clutch_lost_penalty):.1f}%, Traded: {traded_bonus:.1f}%")

            print(f"  Enhanced swing modifiers applied")
        if stats["name"]:  # Only include players with names
            # Calculate derived stats
            headshot_pct = (stats["headshots"] / max(stats["kills"], 1)) * 100
            adr = stats["damage_dealt"] / rounds_count  # ADR = Average Damage per Round
            
            # Calculate damage taken per round
            damage_taken = player_damage_taken.get(steamid, 0)
            adr_taken = damage_taken / rounds_count
            
            # Calculate ADR difference (damage efficiency)
            adr_diff = adr - adr_taken
            
            # Calculate equipment efficiency (average equipment value destroyed per round)
            equipment_ef = stats["equipment_value_destroyed"] / rounds_count if rounds_count > 0 else 0
            
            # Proper KAST calculation using tracked rounds
            kast_rounds = stats.get("kast_rounds", 0)
            kast_pct = (kast_rounds / rounds_count) * 100
            
            # Opening kill/death ratio
            opk_opd = f"{stats['opening_kills']}:{stats['opening_deaths']}"
            
            # Calculate Impact Rating components with updated weights including flash assists and equipment efficiency
            kpr = stats["kills"] / rounds_count if rounds_count > 0 else 0
            opkpr = stats["opening_kills"] / rounds_count if rounds_count > 0 else 0
            opdpr = stats["opening_deaths"] / rounds_count if rounds_count > 0 else 0
            fapr = stats["flash_assists"] / rounds_count if rounds_count > 0 else 0  # Flash assists per round
            equipment_ef = stats["equipment_value_destroyed"] / rounds_count if rounds_count > 0 else 0
            
            # Convert equipment efficiency to per-round basis with scaling
            # Scale equipment efficiency: divide by 1000 to get reasonable weight (2000 avg becomes 2.0)
            equipment_per_round_scaled = equipment_ef / 1000 if equipment_ef > 0 else 0
            
            # Clutch rates - separate for wins and losses
            clutch_win_rate = stats["clutches_won"] / rounds_count if rounds_count > 0 else 0
            clutch_loss_rate = stats["clutches_lost"] / rounds_count if rounds_count > 0 else 0
            
            # Calculate base impact rating
            base_impact = (1.8 * kpr) + (0.5 * opkpr) - (0.5 * opdpr) + (1.3 * fapr) + (0.3 * clutch_win_rate) - (0.3 * clutch_loss_rate) + (0.8 * equipment_per_round_scaled)
            
            # Store this for later use in team outcome calculation
            stats["weighted_impact_per_round"] = base_impact
            
            # Calculate swing per round
            swing_per_round = stats["swing_total"] / rounds_count if rounds_count > 0 else 0
            
            # Calculate Economy Efficiency
            equipment_destroyed = stats["equipment_value_destroyed"]
            equipment_lost = stats["equipment_value_lost"]
            total_equipment_interactions = stats["kills_for_equipment"] + stats["deaths_with_equipment"]
            
            # Debug for the first few players
            if list(player_stats.keys()).index(steamid) < 2:
                print(f"    {stats['name']}: flash_bonus={flash_bonus:.1f}%, clutch_bonus={(clutch_won_bonus + clutch_lost_penalty):.1f}%, traded_bonus={traded_bonus:.1f}%, kills_won_bonus={kills_won_bonus:.1f}%, deaths_lost_penalty={deaths_lost_penalty:.1f}%")

            if total_equipment_interactions > 0:
                # Calculate net equipment impact per interaction
                net_equipment_impact = equipment_destroyed - equipment_lost
                avg_impact_per_interaction = net_equipment_impact / total_equipment_interactions
                
                # Convert to weighted percentage (scale: $1000 impact = 10%)
                economy_efficiency = (avg_impact_per_interaction / 1000) * 10
            else:
                economy_efficiency = 0.0
            
            stats_list.append({
                "player_name": stats["name"],
                "opk_d": opk_opd,  # Opening Kills:Deaths
                "mks": stats["multi_kills"],  # Multi-kills (2+ in round)
                "kast": round(kast_pct, 1),  # KAST percentage
                "1vsx": stats["clutches_won"],  # Clutches won
                "swing": round(swing_per_round, 1),  # Swing rating percentage per round
                "economy_eff": round(economy_efficiency, 1),  # Economy efficiency percentage
                "kills": stats["kills"],
                "headshots": stats["headshots"],
                "headshot_percentage": round(headshot_pct, 1),
                "assists": stats["assists"],
                "flash_assists": stats["flash_assists"],
                "deaths": stats["deaths"],
                "traded_deaths": len(stats["rounds_traded"]),
                "adr": round(adr, 1),
                "adr_taken": round(adr_taken, 1),  # Average damage taken per round
                "adr_diff": round(adr_diff, 1),  # ADR efficiency difference
                "equipment_ef": round(equipment_ef, 0),  # Equipment value destroyed per round
                "equipment_lost": round(stats["equipment_value_lost"] / rounds_count, 0),  # Equipment value lost per round
                "kpr": round(kpr, 3),  # Kills per round
                "opkpr": round(opkpr, 3),  # Opening kills per round
                "opdpr": round(opdpr, 3),  # Opening deaths per round
                "fapr": round(fapr, 3),  # Flash assists per round
                "equipment_ef_scaled": round(equipment_per_round_scaled, 3),  # Equipment efficiency per round (scaled)
                "clutch_win_rate": round(clutch_win_rate, 3),  # Clutch wins per round
                "clutch_loss_rate": round(clutch_loss_rate, 3),  # Clutch losses per round
                "rounds_count": rounds_count,
                "rounds_won": stats["rounds_won"],
                "rounds_lost": stats["rounds_lost"],
                "map_name": map_name,
                "demo_file": stats["demo_file"]
            })
    
    print(f"  Generated stats for {len(stats_list)} players")
    
    # Clean up memory
    del demo_data
    gc.collect()
    
    return stats_list

# Main processing with session limits and resume capability

# Capped at 30 to prevent memory overload
def process_all_demos(demo_directory, output_file="player_statistics.csv", max_demos_per_session=30):
    """Process all .dem files in a directory with session limits and resume capability."""
    
    # Check for already processed demos
    processed_demos = get_processed_demos(output_file)
    
    # Find all .dem files
    dem_files = glob.glob(os.path.join(demo_directory, "*.dem"))
    
    if not dem_files:
        print(f"No .dem files found in {demo_directory}")
        return None
    
    # Filter out already processed demos
    unprocessed_files = []
    for dem_file in dem_files:
        demo_filename = os.path.basename(dem_file)
        if demo_filename not in processed_demos:
            unprocessed_files.append(dem_file)
        else:
            print(f"Skipping already processed: {demo_filename}")
    
    if not unprocessed_files:
        print("All demo files have already been processed!")
        return "ALL_PROCESSED"
    
    print(f"Found {len(dem_files)} demo files ({len(processed_demos)} already processed)")
    print(f"Processing {len(unprocessed_files)} unprocessed files")
    
    # Sort files for consistent processing order
    unprocessed_files.sort()
    
    session_demos_processed = 0
    total_players_processed = 0
    failed_files = []
    
    # Process each demo file individually
    for i, dem_file in enumerate(unprocessed_files):
        # Check if we've reached the session limit
        if session_demos_processed >= max_demos_per_session:
            print(f"\n Session limit reached! Processed {session_demos_processed} demos in this session.")
            print(f"Remaining demos to process: {len(unprocessed_files) - i}")
            print(f"All progress has been saved to {output_file}")
            print(f"Restart the script to continue processing...")
            return "SESSION_LIMIT_REACHED"
        
        demo_filename = os.path.basename(dem_file)
        print(f"\n[{i+1}/{len(unprocessed_files)}] [Session: {session_demos_processed+1}/{max_demos_per_session}] Processing: {demo_filename}")
        
        try:
            # Parse single demo file
            parsed_demo = parse_demo_file(dem_file)
            
            if parsed_demo:
                # Calculate statistics for this demo
                player_stats = calculate_player_stats_single_demo(parsed_demo)
                
                # Save immediately after processing this demo
                if player_stats:
                    save_progress(player_stats, output_file, append_mode=True)
                    total_players_processed += len(player_stats)
                    print(f"   Processed {len(player_stats)} players from {demo_filename}")
                else:
                    print(f"   No player stats generated for {demo_filename}")
                    failed_files.append(demo_filename)
                
                session_demos_processed += 1
            else:
                print(f"  Failed to parse {demo_filename}")
                failed_files.append(demo_filename)
                session_demos_processed += 1  # Count failed attempts too
                
        except Exception as e:
            print(f"  ERROR processing {demo_filename}: {e}")
            import traceback
            traceback.print_exc()
            failed_files.append(demo_filename)
            session_demos_processed += 1  # Count failed attempts too
            continue
            
        # Force garbage collection every few files
        if (i + 1) % 3 == 0:
            gc.collect()
            print(f"  Memory cleanup - processed {i+1}/{len(unprocessed_files)} files")
    
    print(f"\n Session processing complete!")
    print(f"Successfully processed {session_demos_processed - len(failed_files)} demo files.")
    print(f"Total players processed in this session: {total_players_processed}")
    
    if failed_files:
        print(f"Failed to parse {len(failed_files)} files: {failed_files[:5]}{'...' if len(failed_files) > 5 else ''}")
    
    return "SESSION_COMPLETE"

def main():
    """Main function to process demos and save statistics with session limits."""
    print("CS2 Demo Parser - Player Statistics Generator with Enhanced Swing Rating")
    
    parser = argparse.ArgumentParser(description="Parse CS2 demos and output player statistics CSV with enhanced swing rating and economy efficiency.")
    parser.add_argument("--demo_dir", default=r"YOUR_DEMO_FILEPATH", help="Directory containing .dem files") # Set demo filepath relevant to user
    parser.add_argument("--output", default="player_statistics.csv", help="Output CSV path")
    parser.add_argument("--max_per_session", type=int, default=30, help="Maximum number of demos to process per session (default: 30)")
    args = parser.parse_args()
    
    demo_directory = args.demo_dir
    output_file = args.output
    max_demos_per_session = args.max_per_session
    
    print("=" * 60)
    print(f"Demo Directory: {demo_directory}")
    print(f"Output File: {output_file}")
    print(f"Max demos per session: {max_demos_per_session}")
    print("=" * 60)
    
    # Check if demo directory exists
    if not os.path.exists(demo_directory):
        print(f"ERROR: Demo directory '{demo_directory}' not found!")
        print("Please create the directory and place your .dem files there.")
        return
    
    try:
        # Process demos with session limits
        result = process_all_demos(demo_directory, output_file, max_demos_per_session)
        
        if result == "SESSION_LIMIT_REACHED":
            print(f"\n" + "=" * 60)
            print(f"SESSION SUMMARY")
            print(f"=" * 60)
            print(f"Successfully processed {max_demos_per_session} demos in this session")
            print(f"All data saved to: {output_file}")
            print(f"To continue processing, run the same command again:")
            print(f"   python {os.path.basename(__file__)} --demo_dir \"{demo_directory}\" --output \"{output_file}\" --max_per_session {max_demos_per_session}")
            print(f"=" * 60)
            
        elif result == "SESSION_COMPLETE":
            print(f"Session processing complete! Check {output_file}")
            
        elif result == "ALL_PROCESSED":
            print(f"All demos have already been processed!")
        
        # Show final stats if CSV exists
        if os.path.exists(output_file):
            try:
                df = pd.read_csv(output_file)
                print(f"Current Statistics:")
                print(f"   Total player records: {len(df)}")
                print(f"   Unique demo files processed: {len(df['demo_file'].unique())}")
                print(f"   Unique players: {len(df['player_name'].unique())}")
                print(f"   Maps covered: {len(df['map_name'].unique())}")
                
                # Show swing statistics with detailed metrics
                if 'swing' in df.columns:
                    swing_data = df['swing']
                    avg_swing = swing_data.mean()
                    max_swing = swing_data.max()
                    min_swing = swing_data.min()
                    median_swing = swing_data.median()
                    std_swing = swing_data.std()
                    
                    # Count positive/negative swings
                    positive_swings = len(swing_data[swing_data > 0])
                    negative_swings = len(swing_data[swing_data < 0])
                    neutral_swings = len(swing_data[swing_data == 0])
                    
                    # Percentiles
                    p25_swing = swing_data.quantile(0.25)
                    p75_swing = swing_data.quantile(0.75)
                    p90_swing = swing_data.quantile(0.90)
                    p10_swing = swing_data.quantile(0.10)
                    
                    # Find extreme players
                    max_swing_player = df.loc[swing_data.idxmax(), 'player_name'] if not swing_data.empty else "N/A"
                    min_swing_player = df.loc[swing_data.idxmin(), 'player_name'] if not swing_data.empty else "N/A"
                    
                    print(f"   SWING RATING ANALYSIS:")
                    print(f"   ├─ Range: {min_swing:.1f}% to {max_swing:.1f}% (spread: {max_swing-min_swing:.1f}%)")
                    print(f"   ├─ Central: avg={avg_swing:.1f}%, median={median_swing:.1f}%, std_dev={std_swing:.1f}%")
                    print(f"   ├─ Percentiles: 10th={p10_swing:.1f}%, 25th={p25_swing:.1f}%, 75th={p75_swing:.1f}%, 90th={p90_swing:.1f}%")
                    print(f"   ├─ Distribution: {positive_swings} positive, {negative_swings} negative, {neutral_swings} neutral")
                    print(f"   ├─ Best performer: {max_swing_player} ({max_swing:.1f}%)")
                    print(f"   └─ Worst performer: {min_swing_player} ({min_swing:.1f}%)")
                
                # Show economy efficiency statistics with detailed metrics
                if 'economy_eff' in df.columns:
                    econ_data = df['economy_eff']
                    avg_economy = econ_data.mean()
                    max_economy = econ_data.max()
                    min_economy = econ_data.min()
                    median_economy = econ_data.median()
                    std_economy = econ_data.std()
                    
                    # Count efficient/inefficient players
                    efficient_players = len(econ_data[econ_data > 0])
                    inefficient_players = len(econ_data[econ_data < 0])
                    neutral_economy = len(econ_data[econ_data == 0])
                    
                    # Find extreme players
                    max_econ_player = df.loc[econ_data.idxmax(), 'player_name'] if not econ_data.empty else "N/A"
                    min_econ_player = df.loc[econ_data.idxmin(), 'player_name'] if not econ_data.empty else "N/A"
                    
                    print(f"   ECONOMY EFFICIENCY ANALYSIS:")
                    print(f"   ├─ Range: {min_economy:.1f}% to {max_economy:.1f}% (spread: {max_economy-min_economy:.1f}%)")
                    print(f"   ├─ Central: avg={avg_economy:.1f}%, median={median_economy:.1f}%, std_dev={std_economy:.1f}%")
                    print(f"   ├─ Distribution: {efficient_players} efficient, {inefficient_players} inefficient, {neutral_economy} neutral")
                    print(f"   ├─ Most efficient: {max_econ_player} ({max_economy:.1f}%)")
                    print(f"   └─ Least efficient: {min_econ_player} ({min_economy:.1f}%)")
                
                # Show correlation between swing and other key metrics
                if 'swing' in df.columns and 'kills' in df.columns and 'deaths' in df.columns:
                    print(f"   SWING CORRELATIONS:")
                    
                    # Calculate K/D ratio
                    df['kd_ratio'] = df['kills'] / df['deaths'].replace(0, 1)  # Avoid division by zero
                    
                    # Correlations
                    swing_kd_corr = df['swing'].corr(df['kd_ratio'])
                    swing_kast_corr = df['swing'].corr(df['kast']) if 'kast' in df.columns else 0
                    swing_adr_corr = df['swing'].corr(df['adr']) if 'adr' in df.columns else 0
                    swing_1vsx_corr = df['swing'].corr(df['1vsx']) if '1vsx' in df.columns else 0
                    
                    print(f"   ├─ K/D Ratio: r={swing_kd_corr:.3f}")
                    print(f"   ├─ KAST: r={swing_kast_corr:.3f}")
                    print(f"   ├─ ADR: r={swing_adr_corr:.3f}")
                    print(f"   └─ Clutches: r={swing_1vsx_corr:.3f}")
                    
                # Show top and bottom swing performers with context
                if len(df) > 5:
                    print(f"   TOP/BOTTOM SWING PERFORMERS:")
                    
                    # Top 3 swing performers
                    top_players = df.nlargest(3, 'swing')[['player_name', 'swing', 'kills', 'deaths', 'kast']].round(1)
                    print(f"   Top performers:")
                    for _, player in top_players.iterrows():
                        kd = player['kills'] / max(player['deaths'], 1)
                        print(f"   ├─ {player['player_name']}: {player['swing']:.1f}% swing, {kd:.2f} K/D, {player['kast']:.1f}% KAST")
                    
                    # Bottom 3 swing performers  
                    bottom_players = df.nsmallest(3, 'swing')[['player_name', 'swing', 'kills', 'deaths', 'kast']].round(1)
                    print(f"   Bottom performers:")
                    for _, player in bottom_players.iterrows():
                        kd = player['kills'] / max(player['deaths'], 1)
                        print(f"   ├─ {player['player_name']}: {player['swing']:.1f}% swing, {kd:.2f} K/D, {player['kast']:.1f}% KAST")
            except Exception as e:
                print(f"Could not read final stats: {e}")
                
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()