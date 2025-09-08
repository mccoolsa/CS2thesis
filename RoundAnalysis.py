import os
import pandas as pd
from demoparser2 import DemoParser
import numpy as np
import glob
import re
import argparse
from datetime import datetime
import gc

# ---------------------------
# Small utilities
# ---------------------------

PART_PATTERNS = [
    r'(?:^|[-_])p(?:art)?\s*(\d+)\b',   # -p2, _p2, part2, -part2
    r'\bpart\s*(\d+)\b',                # "part 2"
]

def extract_base_and_part(filename_no_ext: str):
    """
    From a filename like 'matchname-mapname-p2' -> ('matchname-mapname', 2).
    If no part number, returns (filename_no_ext, None).
    """
    for pat in PART_PATTERNS:
        m = re.search(pat, filename_no_ext, flags=re.IGNORECASE)
        if m:
            try:
                part = int(m.group(1))
            except Exception:
                part = None
            base = filename_no_ext[:m.start()].rstrip('-_ ').strip()
            if not base:
                base = filename_no_ext
            return base, part
    return filename_no_ext, None

def get_map_name_quick(demo_path: str):
    """Read only the header to get the map name. Returns 'unknown' on error."""
    try:
        parser = DemoParser(demo_path)
        header = parser.parse_header()
        map_name = header.get('map_name', 'unknown') or 'unknown'
        del parser  # Help with memory cleanup
        return map_name
    except Exception:
        return 'unknown'

def read_file_list_with_dates(file_list_path: str):
    """Read file_list.txt and extract filenames with their modification dates."""
    file_dates = {}
    
    if not os.path.exists(file_list_path):
        return file_dates
    
    try:
        with open(file_list_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        for line in lines:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            parts = line.split()
            if len(parts) >= 2:
                filename = parts[0].strip()
                date_str = ' '.join(parts[1:]).strip()
                file_dates[filename] = date_str
            elif len(parts) == 1:
                filename = parts[0].strip()
                if filename.endswith('.dem'):
                    file_dates[filename] = "Unknown"
        
    except Exception as e:
        print(f"ERROR reading {file_list_path}: {e}")
        
    return file_dates

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

def save_progress(round_analysis_list, output_file, append_mode=False):
    """Save current progress to CSV."""
    if not round_analysis_list:
        return
        
    df = pd.DataFrame(round_analysis_list)
    
    if append_mode and os.path.exists(output_file):
        df.to_csv(output_file, mode='a', header=False, index=False)
        print(f"✅ Appended {len(df)} rounds to {output_file}")
    else:
        df.to_csv(output_file, index=False)
        print(f"✅ Saved {len(df)} rounds to {output_file}")

def get_file_modification_date(filepath: str):
    """Get the actual file modification date as fallback."""
    try:
        mtime = os.path.getmtime(filepath)
        return datetime.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M:%S')
    except:
        return "Unknown"

def parse_date_for_sorting(date_str):
    """Convert date string to datetime for sorting."""
    if date_str == "Unknown":
        return datetime.min
    
    try:
        for fmt in ["%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d"]:
            try:
                return datetime.strptime(date_str, fmt)
            except:
                continue
        return datetime.min
    except:
        return datetime.min

# ---------------------------
# Parsing helpers
# ---------------------------

def parse_demo_file(demo_path):
    """Parse a single demo file and extract round and equipment data."""
    try:
        parser = DemoParser(demo_path)

        events = parser.parse_events([
            "round_end",
            "round_start",
            "bomb_planted",
            "bomb_defused",
            "bomb_exploded",
            "player_death"
        ])

        ticks = parser.parse_ticks([
            "health", "armor", "active_weapon_name", "team_name",
            "name", "steamid", "tick", "inventory", "total_rounds_played"
        ])

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
        print(f"ERROR parsing {os.path.basename(demo_path)}: {e}")
        return None

# ---------------------------
# Feature logic
# ---------------------------

def calculate_player_equipment(tick):
    """Calculate total equipment value for a player based on their tick data."""
    weapon = tick.get('active_weapon_name', '')
    armor = tick.get('armor', 0)
    inventory = tick.get('inventory', '')
    team = str(tick.get('team_name', '')).upper()

    weapon_str = str(weapon).lower()
    primary_weapon = weapon
    found_weapon_value = 0

    clean_weapon = weapon_str.replace('weapon_', '').replace('_', '')
    weapon_prices = {
        'ak47': 2700, 'awp': 4750, 'm4a4': 2900, 'm4a1silencer': 2900, 'm4a1': 2900, 'm4a1s': 2900,
        'famas': 2250, 'galil': 1800, 'galilar': 1800, 'aug': 3300, 'sg553': 3000, 'sg556': 3000,
        'scar20': 5000, 'g3sg1': 5000, 'ssg08': 1700, 'scout': 1700,
        'mp9': 1250, 'mac10': 1050, 'mp7': 1500, 'ump45': 1200, 'p90': 2350,
        'bizon': 1400, 'ppbizon': 1400, 'mp5sd': 1500, 'mp5': 1500,
        'glock': 0, 'glock18': 0, 'usp': 0, 'uspsilencer': 0, 'hkp2000': 0,
        'p250': 300, 'fiveseven': 500, 'tec9': 500, 'deagle': 700, 'deserteagle': 700,
        'dualberettas': 300, 'cz75a': 500, 'cz75auto': 500, 'revolver': 600, 'r8revolver': 600,
        'nova': 1050, 'xm1014': 2000, 'sawedoff': 1100, 'mag7': 1300,
        'm249': 5200, 'negev': 1700,
        'knife': 0, 'knifegg': 0
    }

    if clean_weapon in weapon_prices:
        found_weapon_value = weapon_prices[clean_weapon]
        primary_weapon = clean_weapon

    if found_weapon_value <= 0 and inventory:
        inventory_str = str(inventory).lower().replace('weapon_', '').replace('_', '')
        for weapon_name, price in weapon_prices.items():
            if price > 1000 and weapon_name in inventory_str:
                found_weapon_value = price
                primary_weapon = weapon_name
                break
        if found_weapon_value <= 0:
            for weapon_name, price in weapon_prices.items():
                if 200 < price <= 1000 and weapon_name in inventory_str:
                    found_weapon_value = price
                    primary_weapon = weapon_name
                    break

    if found_weapon_value <= 0:
        primary_weapon = 'glock' if 'TERRORIST' in team else 'hkp2000'
        found_weapon_value = 0

    armor = tick.get('armor', 0)
    armor_cost = 0
    if armor >= 100:
        armor_cost = 1000
    elif armor >= 50:
        armor_cost = 1000
    elif armor >= 1:
        armor_cost = 650

    utility_value = 0
    if inventory:
        inventory_str = str(inventory).lower()
        utility_value += inventory_str.count('he') * 300
        utility_value += inventory_str.count('flash') * 200
        utility_value += inventory_str.count('smoke') * 300
        utility_value += inventory_str.count('inc') * 500
        utility_value += inventory_str.count('molotov') * 400
        utility_value += inventory_str.count('decoy') * 50
        if ('defus' in inventory_str or 'kit' in inventory_str) and ('CT' in team or 'COUNTER' in team):
            utility_value += 400
        if utility_value < 200 and found_weapon_value > 2000:
            utility_value = 500

    total_equipment = found_weapon_value + armor_cost + utility_value
    return total_equipment, primary_weapon, utility_value

def detect_team_from_round(ticks, round_start_tick, round_end_tick, target_side):
    """Detect actual team name from players on a specific side during a round."""
    team_rosters = {
        'Team Vitality': ['apex', 'zywoo', 'flamez', 'mezii', 'ropz', 'apEX'],
        'MOUZ': ['torzsi', 'xertion', 'jimpphat', 'brollan', 'spinx', 'xertioN', 'Jimpphat', 'Brollan', 'Spinx'],
        'Team Spirit': ['chopper', 'zont1x', 'donk', 'sh1ro', 'zweih', 'zweiH'],
        'The MongolZ': ['blitz', 'techno4k', '910', 'mzinho', 'senzu', 'bLitz', 'Techno4K', 'Senzu'],
        'Aurora Gaming': ['xantares', 'maj3r', 'wicadia', 'woxic', 'jottaaa', 'XANTARES', 'MAJ3R', 'jottAAA'],
        'Virtus.pro': ['fl1t', 'fame', 'electronic', 'icy', 'perfecto', 'FL1T', 'electroNic', 'ICY', 'Perfecto'],
        'Astralis': ['dev1ce', 'staehr', 'stavn', 'jabbi', 'hooxi', 'Staehr', 'HooXi'],
        'Team Liquid': ['naf', 'twistzz', 'ultimate', 'nertz', 'siuhy', 'NAF', 'Twistzz', 'NertZ', 'siuhyC']
    }

    buffer_time = 5000
    round_ticks = ticks[(ticks['tick'] >= round_start_tick - buffer_time) & (ticks['tick'] <= round_end_tick + buffer_time)]
    if round_ticks.empty:
        return f"{target_side}_Team"

    target_players = round_ticks[round_ticks['team_name'].str.contains('terrorist' if target_side == 'T' else 'counter|ct', case=False, na=False)]
    if target_players.empty:
        return f"{target_side}_Team"

    player_names = target_players['name'].unique()
    team_matches = {}
    for team_name, roster in team_rosters.items():
        matches = 0
        for player_name in player_names:
            if pd.isna(player_name):
                continue
            clean_player = str(player_name).lower().strip()
            for roster_player in roster:
                if (roster_player.lower() in clean_player or clean_player in roster_player.lower() or clean_player == roster_player.lower()):
                    matches += 1
                    break
        if matches > 0:
            team_matches[team_name] = matches

    if team_matches:
        best_team = max(team_matches.items(), key=lambda x: x[1])
        if best_team[1] >= 1:
            return best_team[0]

    if not target_players.empty:
        sample_team_name = target_players['team_name'].iloc[0]
        clean_team_name = str(sample_team_name).replace('Terrorists', '').replace('Counter-Terrorists', '').strip()
        if clean_team_name and len(clean_team_name) > 1 and clean_team_name.lower() not in ['', 'team']:
            return clean_team_name

    return f"{target_side}_Team"

def analyze_first_kill_and_advantage(death_events, round_start_tick, round_end_tick, ticks, round_num, t_team_name, ct_team_name):
    """Analyze first kill and two-person advantage for a round."""
    first_kill_team = None
    two_person_advantage_team = None

    if death_events.empty:
        return first_kill_team, two_person_advantage_team

    buffer_ticks = 128 * 5
    round_deaths = death_events[(death_events['tick'] >= round_start_tick - buffer_ticks) & (death_events['tick'] <= round_end_tick + buffer_ticks)].copy()
    if round_deaths.empty:
        return first_kill_team, two_person_advantage_team

    round_deaths = round_deaths.sort_values('tick')
    mapping_ticks = ticks[(ticks['tick'] >= round_start_tick - 15000) & (ticks['tick'] <= round_end_tick + 15000)]
    steamid_to_team = {}
    if not mapping_ticks.empty:
        for steamid in mapping_ticks['steamid'].unique():
            if pd.isna(steamid):
                continue
            player_ticks = mapping_ticks[mapping_ticks['steamid'] == steamid]
            if not player_ticks.empty:
                team_name = player_ticks['team_name'].mode()
                if len(team_name) > 0:
                    steamid_to_team[str(steamid)] = str(team_name.iloc[0]).upper()

    if not round_deaths.empty:
        first_death = round_deaths.iloc[0]
        victim_steamid = str(first_death.get('user_steamid'))
        victim_team = steamid_to_team.get(victim_steamid)
        if victim_team and 'TERRORIST' in victim_team:
            first_kill_team = ct_team_name
        elif victim_team and ('CT' in victim_team or 'COUNTER' in victim_team):
            first_kill_team = t_team_name

    t_alive = 5
    ct_alive = 5
    for _, death in round_deaths.iterrows():
        victim_steamid = str(death.get('user_steamid'))
        victim_team = steamid_to_team.get(victim_steamid)
        if victim_team and 'TERRORIST' in victim_team:
            t_alive -= 1
        elif victim_team and ('CT' in victim_team or 'COUNTER' in victim_team):
            ct_alive -= 1
        if two_person_advantage_team is None:
            advantage = abs(t_alive - ct_alive)
            if advantage >= 2:
                two_person_advantage_team = t_team_name if t_alive > ct_alive else ct_team_name
                break

    return first_kill_team, two_person_advantage_team

# ---------------------------
# Round analysis driver
# ---------------------------

def analyze_single_demo(demo_data, output_file):
    """Analyze rounds for a single demo file and save immediately."""
    round_analysis_list = []
    
    if demo_data is None:
        return []
        
    print(f"Analyzing {os.path.basename(demo_data['demo_path'])}...", end="")
    
    events = demo_data["events"]
    ticks = demo_data["ticks"]
    map_name = demo_data["map_name"]
    demo_path = demo_data["demo_path"]
    file_date = demo_data.get("file_date", "Unknown")
    round_offset = demo_data.get("round_offset", 0)

    events_dict = {}
    if isinstance(events, list):
        for event_tuple in events:
            if len(event_tuple) >= 2:
                event_name = event_tuple[0]
                event_df = event_tuple[1]
                events_dict[event_name] = event_df

    if "round_end" not in events_dict or events_dict["round_end"].empty:
        print(" ❌ No rounds found")
        return []

    round_end_events = events_dict["round_end"]
    round_start_events = events_dict.get("round_start", pd.DataFrame())
    death_events = events_dict.get("player_death", pd.DataFrame())

    valid_rounds = []
    for idx, round_event in round_end_events.iterrows():
        winner = round_event.get('winner', '')
        if winner in ['T', 'CT']:
            valid_rounds.append(round_event)

    rounds_processed = 0
    for round_idx, round_event in enumerate(valid_rounds):
        round_num = round_idx + 1 + round_offset
        winner = round_event.get('winner', '')
        round_end_tick = round_event.get('tick', 0)

        is_pistol_round = (round_num == 1 or round_num == 13)

        round_start_tick = round_end_tick - 115 * 128
        if not round_start_events.empty:
            potential_starts = round_start_events[(round_start_events['tick'] < round_end_tick) & (round_start_events['tick'] > round_end_tick - 120 * 128)]
            if not potential_starts.empty:
                closest_start = potential_starts.iloc[-1]
                round_start_tick = closest_start['tick']

        team_name_win = detect_team_from_round(ticks, round_start_tick, round_end_tick, winner)
        t_team_name = detect_team_from_round(ticks, round_start_tick, round_end_tick, 'T')
        ct_team_name = detect_team_from_round(ticks, round_start_tick, round_end_tick, 'CT')

        first_kill_team, two_person_advantage_team = analyze_first_kill_and_advantage(
            death_events, round_start_tick, round_end_tick, ticks, round_num, t_team_name, ct_team_name
        )

        buy_time_start = round_start_tick + int(0.5 * 128)
        buy_time_end = round_start_tick + int(15 * 128)

        win_method = 'elimination'
        bomb_planted = False
        bomb_defused = False
        bomb_exploded = False

        for event_name in ['bomb_planted', 'bomb_defused', 'bomb_exploded']:
            if event_name in events_dict and not events_dict[event_name].empty:
                bomb_events = events_dict[event_name]
                round_bomb_events = bomb_events[(bomb_events['tick'] >= round_start_tick) & (bomb_events['tick'] <= round_end_tick)]
                if not round_bomb_events.empty:
                    if event_name == 'bomb_planted':
                        bomb_planted = True
                    elif event_name == 'bomb_defused':
                        bomb_defused = True
                    elif event_name == 'bomb_exploded':
                        bomb_exploded = True

        if bomb_exploded and winner == 'T':
            win_method = 'bomb_explosion'
        elif bomb_defused and winner == 'CT':
            win_method = 'bomb_defuse'
        else:
            round_check_ticks = ticks[(ticks['tick'] >= round_end_tick - 1000) & (ticks['tick'] <= round_end_tick)]
            alive_terrorists = 0
            alive_cts = 0
            total_t_players = 0
            total_ct_players = 0

            if not round_check_ticks.empty:
                latest_player_ticks = round_check_ticks.groupby('steamid').last()
                for steamid, player_tick in latest_player_ticks.iterrows():
                    health = player_tick.get('health', 0)
                    team = str(player_tick.get('team_name', '')).upper()
                    if 'TERRORIST' in team:
                        total_t_players += 1
                        if health > 0:
                            alive_terrorists += 1
                    elif 'CT' in team or 'COUNTER' in team:
                        total_ct_players += 1
                        if health > 0:
                            alive_cts += 1

            if winner == 'CT':
                if bomb_planted and not bomb_defused and not bomb_exploded:
                    win_method = 'time_expired'
                elif alive_terrorists == 0 and total_t_players > 0:
                    win_method = 'elimination'
                elif alive_terrorists > 0 and not bomb_planted:
                    win_method = 'time_expired'
                else:
                    win_method = 'elimination'
            elif winner == 'T':
                if alive_cts == 0 and total_ct_players > 0:
                    win_method = 'elimination'
                elif alive_cts > 0:
                    win_method = 'time_expired'
                else:
                    win_method = 'elimination'
            else:
                win_method = 'elimination'

        round_end_ticks = ticks[(ticks['tick'] >= round_end_tick - 200) & (ticks['tick'] <= round_end_tick + 200)]
        t_alive = 0
        ct_alive = 0
        if not round_end_ticks.empty:
            latest_player_ticks = round_end_ticks.groupby('steamid').last()
            for steamid, player_tick in latest_player_ticks.iterrows():
                health = player_tick.get('health', 0)
                team = str(player_tick.get('team_name', '')).upper()
                if health > 0:
                    if 'TERRORIST' in team:
                        t_alive += 1
                    elif 'CT' in team or 'COUNTER' in team:
                        ct_alive += 1

        if win_method == 'elimination':
            if winner == 'T':
                ct_alive = 0
                t_alive = max(1, t_alive)
            elif winner == 'CT':
                t_alive = 0
                ct_alive = max(1, ct_alive)

        t_equipment = 0
        ct_equipment = 0
        equipment_ticks = ticks[(ticks['tick'] >= buy_time_start) & (ticks['tick'] <= buy_time_end)]

        if not equipment_ticks.empty:
            t_players = {}
            ct_players = {}
            for steamid in equipment_ticks['steamid'].unique():
                if pd.isna(steamid):
                    continue
                player_ticks = equipment_ticks[equipment_ticks['steamid'] == steamid]
                best_equipment = 0
                team = ''
                for _, tick in player_ticks.iterrows():
                    team = str(tick.get('team_name', '')).upper()
                    total_player_equipment, primary_weapon, utility_value = calculate_player_equipment(tick)
                    if total_player_equipment > best_equipment:
                        best_equipment = total_player_equipment
                if 'TERRORIST' in team:
                    t_players[steamid] = best_equipment
                elif 'CT' in team or 'COUNTER' in team:
                    ct_players[steamid] = best_equipment

            t_equipment = sum(t_players.values())
            ct_equipment = sum(ct_players.values())

            if len(t_players) < 5:
                missing_t = 5 - len(t_players)
                if is_pistol_round:
                    default_equipment = 800
                elif round_num <= 3:
                    default_equipment = 1500
                elif round_num >= 25:
                    default_equipment = 5000
                else:
                    default_equipment = 3500
                t_equipment += missing_t * default_equipment

            if len(ct_players) < 5:
                missing_ct = 5 - len(ct_players)
                if is_pistol_round:
                    default_equipment = 800
                elif round_num <= 3:
                    default_equipment = 1500
                elif round_num >= 25:
                    default_equipment = 5200
                else:
                    default_equipment = 3500
                ct_equipment += missing_ct * default_equipment

        if t_equipment == 0 or ct_equipment == 0:
            if round_num == 1 or round_num == 13:
                t_equipment = 4000 if t_equipment == 0 else t_equipment
                ct_equipment = 4000 if ct_equipment == 0 else ct_equipment
            elif round_num >= 25:
                t_equipment = 25000 if t_equipment == 0 else t_equipment
                ct_equipment = 26000 if ct_equipment == 0 else ct_equipment
            else:
                t_equipment = 15000 if t_equipment == 0 else t_equipment
                ct_equipment = 16000 if ct_equipment == 0 else ct_equipment

        if round_num == 1 or round_num == 13:
            t_equipment = min(t_equipment, 4000)
            ct_equipment = min(ct_equipment, 4000)
        elif round_num >= 25:
            if t_equipment < 20000 or ct_equipment < 21000:
                t_equipment = max(t_equipment, 20000)
                ct_equipment = max(ct_equipment, 21000)
        else:
            t_equipment = max(t_equipment, 500)
            ct_equipment = max(ct_equipment, 500)

        def categorize_round(round_num, equipment_value, is_pistol):
            if is_pistol:
                return "pistol"
            elif round_num >= 25:
                if equipment_value < 8000:
                    return "eco"
                elif equipment_value < 12000:
                    return "force"
                elif equipment_value < 18000:
                    return "low_gun"
                else:
                    return "full_gun"
            else:
                if equipment_value < 4000:
                    return "eco"
                elif equipment_value < 9000:
                    return "force"
                elif equipment_value < 15000:
                    return "low_gun"
                else:
                    return "full_gun"

        t_round_type = categorize_round(round_num, t_equipment, is_pistol_round)
        ct_round_type = categorize_round(round_num, ct_equipment, is_pistol_round)

        round_analysis_list.append({
            'demo_file': os.path.basename(demo_path),
            'map_name': map_name,
            'file_modification_date': file_date,
            'round_number': round_num,
            'round_winner': winner,
            'team_name_win': team_name_win,
            'win_method': win_method,
            't_equipment_value': t_equipment,
            'ct_equipment_value': ct_equipment,
            't_round_type': t_round_type,
            'ct_round_type': ct_round_type,
            't_alive_end': t_alive,
            'ct_alive_end': ct_alive,
            'bomb_planted': bomb_planted,
            'bomb_defused': bomb_defused,
            'bomb_exploded': bomb_exploded,
            'first_kill_team': first_kill_team,
            'two_person_advantage_team': two_person_advantage_team
        })
        
        rounds_processed += 1

    print(f" ✅ {rounds_processed} rounds completed")
    
    # Save immediately after processing this demo
    if round_analysis_list:
        save_progress(round_analysis_list, output_file, append_mode=True)
    
    # Clear memory
    del demo_data
    gc.collect()
    
    return round_analysis_list

# ---------------------------
# Main processing logic
# ---------------------------

def process_all_demos(demo_directory, output_file="round_analysis.csv", file_list_path="file_list.txt"):
    """Process all demo files sorted by date (oldest first) with resume capability."""
    
    # Check for already processed demos
    processed_demos = get_processed_demos(output_file)
    
    # Read file list and get file info
    file_dates = read_file_list_with_dates(file_list_path)
    dem_files = glob.glob(os.path.join(demo_directory, "*.dem"))
    if not dem_files:
        print(f"No .dem files found in {demo_directory}")
        return None

    file_infos = []
    
    # Add files from file_list.txt
    for filename in file_dates.keys():
        path = os.path.join(demo_directory, filename)
        if os.path.exists(path) and path in dem_files:
            base_no_ext = filename[:-4] if filename.lower().endswith('.dem') else filename
            base, part = extract_base_and_part(base_no_ext)
            map_name = get_map_name_quick(path)
            info = {
                'path': path,
                'filename': filename,
                'base': base,
                'part_number': part,
                'map_name': map_name,
                'file_date': file_dates[filename]
            }
            file_infos.append(info)

    # Add any remaining files
    processed_paths = {info['path'] for info in file_infos}
    for path in dem_files:
        if path not in processed_paths:
            filename = os.path.basename(path)
            base_no_ext = filename[:-4] if filename.lower().endswith('.dem') else filename
            base, part = extract_base_and_part(base_no_ext)
            map_name = get_map_name_quick(path)
            info = {
                'path': path,
                'filename': filename,
                'base': base,
                'part_number': part,
                'map_name': map_name,
                'file_date': get_file_modification_date(path)
            }
            file_infos.append(info)

    # Filter out already processed demos
    unprocessed_files = []
    for info in file_infos:
        if info['filename'] not in processed_demos:
            unprocessed_files.append(info)
        else:
            print(f"Skipping already processed: {info['filename']}")

    # Sort all files by date (oldest first)
    unprocessed_files.sort(key=lambda x: parse_date_for_sorting(x['file_date']))

    if not unprocessed_files:
        print("All demo files have already been processed!")
        return None

    print(f"Processing {len(unprocessed_files)} unprocessed files in chronological order...")
    print(f"({len(processed_demos)} files were already processed)")

    round_counters = {}
    failed_files = []
    total_rounds_analyzed = 0
    
    # Initialize round counters based on existing data
    if processed_demos:
        try:
            existing_df = pd.read_csv(output_file)
            for _, row in existing_df.iterrows():
                demo_file = row['demo_file']
                map_name = row['map_name']
                round_num = row['round_number']
                
                # Extract base name from demo file
                base_no_ext = demo_file[:-4] if demo_file.lower().endswith('.dem') else demo_file
                base, _ = extract_base_and_part(base_no_ext)
                match_key = f"{base}_{map_name}"
                
                if match_key not in round_counters:
                    round_counters[match_key] = 0
                round_counters[match_key] = max(round_counters[match_key], round_num)
                
            print(f"Restored round counters for {len(round_counters)} match contexts")
        except Exception as e:
            print(f"Warning: Could not restore round counters: {e}")

    # Process each file in chronological order
    for i, info in enumerate(unprocessed_files):
        match_key = f"{info['base']}_{info['map_name']}"
        
        if match_key not in round_counters:
            round_counters[match_key] = 0
        
        current_round_offset = round_counters[match_key]
        
        print(f"[{i+1}/{len(unprocessed_files)}] Processing: {info['filename']}")
        
        try:
            result = parse_demo_file(info['path'])
            if result:
                result['match_name'] = f"{info['base']} [{info['map_name']}]"
                result['part_number'] = info['part_number'] if info['part_number'] is not None else 0
                result['file_date'] = info['file_date']
                result['round_offset'] = current_round_offset
                
                # Process this single demo and save immediately
                round_analysis = analyze_single_demo(result, output_file)
                
                # Count rounds in this file and update counter
                rounds_in_this_file = len(round_analysis)
                if rounds_in_this_file > 0:
                    round_counters[match_key] += rounds_in_this_file
                    total_rounds_analyzed += rounds_in_this_file
                    
            else:
                failed_files.append(info['filename'])
                
        except Exception as e:
            print(f"ERROR processing {info['filename']}: {e}")
            failed_files.append(info['filename'])
            continue
            
        # Force garbage collection every few files
        if (i + 1) % 5 == 0:
            gc.collect()
            print(f"Memory cleanup - processed {i+1}/{len(unprocessed_files)} files")

    print(f"\n🎯 Processing complete!")
    print(f"Successfully processed {len(unprocessed_files) - len(failed_files)} demo files.")
    print(f"Total rounds analyzed in this session: {total_rounds_analyzed}")
    
    if failed_files:
        print(f"Failed to parse {len(failed_files)} files: {failed_files}")

    return "Processing complete - data saved to CSV"

def main():
    """Main function with CLI args."""
    print("CS2 Demo Parser - Resumable Round Analysis")

    parser = argparse.ArgumentParser(description="Parse CS2 demos and output round analysis CSV with resume capability.")
    parser.add_argument("--demo_dir", default=r"C:\\Users\\conor\\DemoLaiho\\demoparser\\demofiles", help="Directory containing .dem files")
    parser.add_argument("--output", default="round_analysis.csv", help="Output CSV path")
    parser.add_argument("--file_list", default="file_list.txt", help="File containing list of demo files with dates")
    args = parser.parse_args()

    demo_directory = args.demo_dir
    output_file = args.output
    file_list_path = args.file_list

    if not os.path.exists(demo_directory):
        print(f"ERROR: Demo directory '{demo_directory}' not found!")
        return

    try:
        result = process_all_demos(demo_directory, output_file, file_list_path)
        if result:
            print(f"✅ Analysis complete! Check {output_file}")
            
            # Show final stats
            if os.path.exists(output_file):
                df = pd.read_csv(output_file)
                print(f"Total records in CSV: {len(df)}")
                print(f"Unique demo files processed: {len(df['demo_file'].unique())}")
        else:
            print("No new data to process.")
            
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()