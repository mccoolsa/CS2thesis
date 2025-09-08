import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
import warnings
warnings.filterwarnings('ignore')

# Set up plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def parse_opening_kills_deaths(opk_d_str):
    """Parse opening kills:deaths string like '2:0', '[2:0]', '[5:1]' into separate values."""
    try:
        if pd.isna(opk_d_str) or opk_d_str == '':
            return 0, 0
        
        # Convert to string and remove brackets if present
        clean_str = str(opk_d_str).strip('[]')
        parts = clean_str.split(':')
        
        if len(parts) == 2:
            return int(parts[0]), int(parts[1])
        return 0, 0
    except Exception as e:
        print(f"Error parsing opk_d: {opk_d_str} - {e}")
        return 0, 0

def create_team_rosters():
    """Define the correct team rosters based on co-occurrence analysis."""
    team_rosters = {
        'Vitality': ['ZywOo', 'apEX', 'flameZ', 'ropz', 'mezii'],
        'Spirit': ['donk', 'zont1x', 'chopper', 'sh1ro', 'magixx'],
        'Astralis': ['stavn', 'HooXi', 'Staehr', 'jabbi', 'dev1ce'],
        'mouz': ['torzsi', 'Spinx', 'xertioN', 'Brollan', 'Jimpphat'],
        'Liquid': ['Twistzz', 'ultimate', 'siuhy', 'NertZ', 'NAF'],
        'Aurora': ['deko', 'clax', 'Woro2k', 'gr1ks', 'KENSI'],  # Need to identify from data
        'The Mongolz': ['910', 'bLitz', 'Techno4K', 'mzinho', 'senzu'],  # Need to identify
        'Virtus Pro': ['fame', 'FL1T', 'XANTARES', 'Jame', 'n0rb3r7']  # Need to identify
    }
    
    # Create reverse mapping: player -> team
    player_team_map = {}
    for team, players in team_rosters.items():
        for player in players:
            player_team_map[player] = team
    
    return team_rosters, player_team_map

def identify_remaining_rosters(df):
    """Identify remaining team rosters using co-occurrence analysis."""
    
    # Filter for matches involving target teams
    target_teams = ['aurora', 'vitality', 'spirit', 'mongolz', 'liquid', 'virtus', 'astralis', 'mouz']
    
    relevant_matches = df[df['demo_file'].str.lower().str.contains('|'.join(target_teams))].copy()
    
    # Group by demo file and analyze player co-occurrence
    from collections import defaultdict
    
    player_pairs = defaultdict(int)
    demo_groups = relevant_matches.groupby('demo_file')
    
    for demo, group in demo_groups:
        players = group['player_name'].unique()
        # Count co-occurrences
        for i, p1 in enumerate(players):
            for p2 in players[i+1:]:
                pair = tuple(sorted([p1, p2]))
                player_pairs[pair] += 1
    
    # Find players who frequently appear together (likely teammates)
    def find_team_clusters(min_cooccurrence=8):
        strong_pairs = {pair: count for pair, count in player_pairs.items() 
                       if count >= min_cooccurrence}
        
        # Build adjacency list
        adjacency = defaultdict(set)
        for (p1, p2), count in strong_pairs.items():
            adjacency[p1].add(p2)
            adjacency[p2].add(p1)
        
        return adjacency, strong_pairs
    
    adjacency, strong_pairs = find_team_clusters()
    
    # Manual identification based on known patterns from the data
    updated_rosters = {
        'Vitality': ['ZywOo', 'apEX', 'flameZ', 'ropz', 'mezii'],
        'Spirit': ['donk', 'zont1x', 'chopper', 'sh1ro', 'magixx'],
        'Astralis': ['stavn', 'HooXi', 'Staehr', 'jabbi', 'dev1ce'],
        'mouz': ['torzsi', 'Spinx', 'xertioN', 'Brollan', 'Jimpphat'],
        'Liquid': ['Twistzz', 'ultimate', 'siuhy', 'NertZ', 'NAF'],
        'Aurora': ['deko', 'clax', 'Woro2k', 'gr1ks', 'KENSI'],
        'The Mongolz': ['910', 'bLitz', 'Techno4K', 'Mzinho', 'Senzu'],
        'Virtus Pro': ['fame', 'FL1T', 'XANTARES', 'Jame', 'n0rb3r7']
    }
    
    # Try to identify Aurora, Mongolz, and Virtus Pro players from the data
    # Look for players who appear in demos with these team names
    aurora_demos = relevant_matches[relevant_matches['demo_file'].str.contains('aurora', case=False)]['player_name'].value_counts()
    mongolz_demos = relevant_matches[relevant_matches['demo_file'].str.contains('mongolz', case=False)]['player_name'].value_counts()
    virtus_demos = relevant_matches[relevant_matches['demo_file'].str.contains('virtus', case=False)]['player_name'].value_counts()
    
    # Get most frequent players from each team (excluding already identified players)
    known_players = set()
    for players in updated_rosters.values():
        known_players.update(players)
    
    # Update rosters with actual data
    if len(aurora_demos) > 0:
        aurora_candidates = [p for p in aurora_demos.head(10).index if p not in known_players]
        if len(aurora_candidates) >= 5:
            updated_rosters['Aurora'] = aurora_candidates[:5]
    
    if len(mongolz_demos) > 0:
        mongolz_candidates = [p for p in mongolz_demos.head(10).index if p not in known_players]
        if len(mongolz_candidates) >= 5:
            updated_rosters['The Mongolz'] = mongolz_candidates[:5]
    
    if len(virtus_demos) > 0:
        virtus_candidates = [p for p in virtus_demos.head(10).index if p not in known_players]
        if len(virtus_candidates) >= 5:
            updated_rosters['Virtus Pro'] = virtus_candidates[:5]
    
    return updated_rosters

def load_and_filter_data(file_path):
    """Load CSV data and filter for target teams with correct rosters."""
    print("🎮 COUNTER-STRIKE ANALYSIS - CORRECTED TEAM ROSTERS")
    print("=" * 60)
    
    # Load data
    df = pd.read_csv(file_path)
    
    # Identify team rosters from the actual data
    print("🔍 Analyzing player co-occurrence to identify correct team rosters...")
    team_rosters = identify_remaining_rosters(df)
    
    # Create player-team mapping
    player_team_map = {}
    for team, players in team_rosters.items():
        for player in players:
            player_team_map[player] = team
    
    # Filter data to only include players from target teams (excluding xfloud)
    target_players = [p for p in player_team_map.keys() if p != 'xfloud']
    filtered_df = df[(df['player_name'].isin(target_players)) & (df['player_name'] != 'xfloud')].copy()
    
    # Add team column
    filtered_df['team'] = filtered_df['player_name'].map(player_team_map)
    
    print(f"📊 FILTERED DATA OVERVIEW:")
    print(f"   • Total records: {len(filtered_df):,} (filtered from {len(df):,})")
    print(f"   • Players from target teams: {len(target_players)}")
    print(f"   • Target teams: {len(team_rosters)}")
    print(f"   • Maps analyzed: {filtered_df['map_name'].nunique()}")
    print()
    
    # Show corrected team rosters
    print("👥 CORRECTED TEAM ROSTERS:")
    print("-" * 50)
    for team, players in team_rosters.items():
        print(f"{team:12s}: {', '.join(players)}")
    print()
    
    # Show team statistics
    team_counts = filtered_df['team'].value_counts()
    print("📈 TEAM MATCH STATISTICS:")
    for team, count in team_counts.items():
        unique_players = filtered_df[filtered_df['team'] == team]['player_name'].nunique()
        print(f"   • {team}: {unique_players} players, {count} match records")
    print()
    
    return filtered_df, player_team_map, team_rosters

def parse_opening_kills_deaths(opk_d_str):
    """Parse opening kills:deaths string like '2:0' into separate values."""
    try:
        if pd.isna(opk_d_str) or opk_d_str == '':
            return 0, 0
        parts = str(opk_d_str).split(':')
        if len(parts) == 2:
            return int(parts[0]), int(parts[1])
        return 0, 0
    except:
        return 0, 0

def calculate_player_stats(df):
    """Calculate comprehensive player statistics including new metrics."""
    
    # Exclude xfloud from analysis
    df_filtered = df[df['player_name'] != 'xfloud'].copy()
    
    # Parse opening kills and deaths
    df_filtered['opening_kills'], df_filtered['opening_deaths'] = zip(*df_filtered['opk_d'].apply(parse_opening_kills_deaths))
    
    # Group by player and calculate aggregate stats
    player_stats = df_filtered.groupby(['player_name', 'team']).agg({
        'kills': 'sum',
        'deaths': 'sum',
        'assists': 'sum',
        'flash_assists': 'sum',
        'headshots': 'sum',
        'headshot_percentage': 'mean',
        'swing': 'mean',
        'adr': 'mean',
        'adr_diff': 'mean',
        'kast': 'mean',
        'rounds_count': 'sum',
        'opening_kills': 'sum',
        'opening_deaths': 'sum',
        'map_name': 'count'  # This gives us match count
    }).round(2)
    
    # Flatten column names and reset index
    player_stats.columns = ['kills', 'deaths', 'assists', 'flash_assists', 'headshots', 'headshot_percentage', 
                           'swing', 'adr', 'adr_diff', 'kast', 'rounds_count', 'opening_kills', 
                           'opening_deaths', 'matches_played']
    player_stats.reset_index(inplace=True)
    player_stats.set_index('player_name', inplace=True)
    
    # Calculate derived metrics
    player_stats['kd_ratio'] = (player_stats['kills'] / player_stats['deaths']).round(3)
    player_stats['kills_per_round'] = (player_stats['kills'] / player_stats['rounds_count']).round(3)
    player_stats['flash_assists_per_round'] = (player_stats['flash_assists'] / player_stats['rounds_count']).round(3)
    
    # Handle division by zero for opening deaths
    player_stats['opening_kd_ratio'] = (player_stats['opening_kills'] / (player_stats['opening_deaths'].replace(0, 0.1))).round(3)
    player_stats['opening_kills_per_round'] = (player_stats['opening_kills'] / player_stats['rounds_count']).round(3)
    
    # Sort by K/D ratio
    player_stats = player_stats.sort_values('kd_ratio', ascending=False)
    
    # Debug: Print available columns
    print(f"🔧 DEBUG: Available columns in player_stats: {list(player_stats.columns)}")
    
    return player_stats

def display_top_performers(player_stats):
    """Display top performers in various categories including new metrics."""
    
    print("🔫 TOP 15 PLAYERS BY K/D RATIO (Correct Team Assignments):")
    print("-" * 85)
    top_kd = player_stats.head(15)[['team', 'kills', 'deaths', 'kd_ratio', 'matches_played']]
    for i, (player, stats) in enumerate(top_kd.iterrows(), 1):
        print(f"{i:2d}. {player:15s} ({stats['team']:12s}) | K/D: {stats['kd_ratio']:5.3f} | "
              f"({stats['kills']:3.0f}K/{stats['deaths']:3.0f}D) | {stats['matches_played']:2.0f} matches")
    print()
    
    print("⚡ TOP 15 PLAYERS BY SWING RATING:")
    print("-" * 85)
    top_swing = player_stats.sort_values('swing', ascending=False).head(15)
    for i, (player, stats) in enumerate(top_swing[['team', 'swing', 'matches_played']].iterrows(), 1):
        print(f"{i:2d}. {player:15s} ({stats['team']:12s}) | Swing: {stats['swing']:6.2f} | "
              f"{stats['matches_played']:2.0f} matches")
    print()
    
    print("💡 TOP 15 PLAYERS BY FLASH ASSISTS PER ROUND:")
    print("-" * 95)
    top_flash = player_stats.sort_values('flash_assists_per_round', ascending=False).head(15)
    for i, (player, stats) in enumerate(top_flash[['team', 'flash_assists', 'flash_assists_per_round', 'matches_played']].iterrows(), 1):
        print(f"{i:2d}. {player:15s} ({stats['team']:12s}) | Flash/Round: {stats['flash_assists_per_round']:5.3f} | "
              f"Total: {stats['flash_assists']:3.0f} | {stats['matches_played']:2.0f} matches")
    print()
    
    print("📊 TOP 15 PLAYERS BY ADR DIFFERENCE:")
    print("-" * 85)
    top_adr_diff = player_stats.sort_values('adr_diff', ascending=False).head(15)
    for i, (player, stats) in enumerate(top_adr_diff[['team', 'adr', 'adr_diff', 'matches_played']].iterrows(), 1):
        print(f"{i:2d}. {player:15s} ({stats['team']:12s}) | ADR Diff: {stats['adr_diff']:6.1f} | "
              f"ADR: {stats['adr']:5.1f} | {stats['matches_played']:2.0f} matches")
    print()
    
    print("🎯 TOP 15 PLAYERS BY OPENING KILLS PER ROUND:")
    print("-" * 95)
    top_opening = player_stats.sort_values('opening_kills_per_round', ascending=False).head(15)
    for i, (player, stats) in enumerate(top_opening.iterrows(), 1):
        ok_per_round = stats['opening_kills_per_round'] if 'opening_kills_per_round' in stats else 0
        ok_kd = stats['opening_kd_ratio'] if 'opening_kd_ratio' in stats else 0
        print(f"{i:2d}. {player:15s} ({stats['team']:12s}) | OK/Round: {ok_per_round:5.3f} | "
              f"OK/OD: {stats['opening_kills']:2.0f}/{stats['opening_deaths']:2.0f} | "
              f"OK K/D: {ok_kd:5.2f} | {stats['matches_played']:2.0f} matches")
    print()
    
    # Elite performers (adjusted thresholds for target teams)
    elite = player_stats[
        (player_stats['matches_played'] >= 8) & 
        (player_stats['kd_ratio'] >= 1.0) & 
        (player_stats['swing'] >= 3)
    ].copy()
    
    # Create composite score
    elite['composite_score'] = elite['kd_ratio'] + (elite['swing'] / 10)
    elite = elite.sort_values('composite_score', ascending=False)
    
    print("🏆 ELITE PERFORMERS FROM TARGET TEAMS:")
    print("-" * 110)
    for i, (player, stats) in enumerate(elite.head(15).iterrows(), 1):
        print(f"{i:2d}. {player:15s} ({stats['team']:12s}) | K/D: {stats['kd_ratio']:5.3f} | "
              f"Swing: {stats['swing']:6.2f} | ADR: {stats['adr']:5.1f} | "
              f"Flash/R: {stats['flash_assists_per_round']:5.3f} | HS%: {stats['headshot_percentage']:4.1f}%")
    print()

def analyze_team_performance(df, player_stats):
    """Analyze performance by team including new metrics."""
    
    team_stats = player_stats.groupby('team').agg({
        'kills': 'sum',
        'deaths': 'sum',
        'kd_ratio': 'mean',
        'swing': 'mean',
        'adr': 'mean',
        'adr_diff': 'mean',
        'flash_assists': 'sum',
        'flash_assists_per_round': 'mean',
        'opening_kills': 'sum',
        'opening_deaths': 'sum',
        'opening_kd_ratio': 'mean',
        'matches_played': 'sum',
        'headshot_percentage': 'mean'
    }).round(2)
    
    team_stats['avg_kd'] = (team_stats['kills'] / team_stats['deaths']).round(3)
    team_stats['players'] = player_stats.groupby('team').size()
    
    # Sort by average K/D
    team_stats = team_stats.sort_values('avg_kd', ascending=False)
    
    print("🏆 TEAM PERFORMANCE RANKING (Correct Rosters):")
    print("-" * 110)
    for i, (team, stats) in enumerate(team_stats.iterrows(), 1):
        print(f"{i}. {team:12s} | Players: {stats['players']:2.0f} | "
              f"Avg K/D: {stats['avg_kd']:5.3f} | Avg Swing: {stats['swing']:6.2f} | "
              f"Avg ADR Diff: {stats['adr_diff']:6.1f} | Flash/R: {stats['flash_assists_per_round']:5.3f} | "
              f"Matches: {stats['matches_played']:3.0f}")
    print()
    
    print("💡 TEAM FLASH ASSIST RANKINGS:")
    print("-" * 80)
    flash_rankings = team_stats.sort_values('flash_assists_per_round', ascending=False)
    for i, (team, stats) in enumerate(flash_rankings.iterrows(), 1):
        print(f"{i}. {team:12s} | Flash Assists/Round: {stats['flash_assists_per_round']:5.3f} | "
              f"Total Flash Assists: {stats['flash_assists']:4.0f}")
    print()
    
    print("🎯 TEAM OPENING KILLS PERFORMANCE:")
    print("-" * 80)
    opening_rankings = team_stats.sort_values('opening_kd_ratio', ascending=False)
    for i, (team, stats) in enumerate(opening_rankings.iterrows(), 1):
        print(f"{i}. {team:12s} | Opening K/D: {stats['opening_kd_ratio']:5.2f} | "
              f"Total OK: {stats['opening_kills']:3.0f} | Total OD: {stats['opening_deaths']:3.0f}")
    print()
    
    return team_stats

def create_comprehensive_visualization(df, player_stats, team_stats):
    """Create comprehensive visualizations with correct team assignments and new metrics."""
    
    # Set up the figure with subplots
    fig = plt.figure(figsize=(24, 20))
    
    # Create team color mapping
    teams = player_stats['team'].unique()
    team_colors = dict(zip(teams, sns.color_palette("Set1", len(teams))))
    
    # 1. Top 12 K/D Ratios by Team (Top Left)
    ax1 = plt.subplot(3, 3, 1)
    top_kd_12 = player_stats.head(12)
    
    colors = [team_colors[team] for team in top_kd_12['team']]
    bars1 = ax1.barh(range(len(top_kd_12)), top_kd_12['kd_ratio'], color=colors)
    ax1.set_yticks(range(len(top_kd_12)))
    ax1.set_yticklabels([f"{name}\n({team})" for name, team in 
                        zip(top_kd_12.index, top_kd_12['team'])], fontsize=7)
    ax1.set_xlabel('K/D Ratio', fontsize=10, fontweight='bold')
    ax1.set_title('🔫 Top 12 K/D Ratio', fontsize=11, fontweight='bold', pad=15)
    ax1.grid(axis='x', alpha=0.3)
    
    # Add value labels
    for i, bar in enumerate(bars1):
        width = bar.get_width()
        ax1.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
                f'{width:.3f}', ha='left', va='center', fontsize=7, fontweight='bold')
    
    # 2. Flash Assists per Round (Top Middle)
    ax2 = plt.subplot(3, 3, 2)
    top_flash_12 = player_stats.sort_values('flash_assists_per_round', ascending=False).head(12)
    flash_colors = [team_colors[team] for team in top_flash_12['team']]
    
    bars2 = ax2.bar(range(len(top_flash_12)), top_flash_12['flash_assists_per_round'], color=flash_colors)
    ax2.set_xticks(range(len(top_flash_12)))
    ax2.set_xticklabels([f"{name}\n({team})" for name, team in 
                        zip(top_flash_12.index, top_flash_12['team'])], 
                       rotation=45, ha='right', fontsize=6)
    ax2.set_ylabel('Flash Assists/Round', fontsize=10, fontweight='bold')
    ax2.set_title('💡 Top 12 Flash Assists/Round', fontsize=11, fontweight='bold', pad=15)
    ax2.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for i, bar in enumerate(bars2):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2, height + 0.005,
                f'{height:.3f}', ha='center', va='bottom', fontsize=7, fontweight='bold')
    
    # 3. ADR Difference (Top Right)
    ax3 = plt.subplot(3, 3, 3)
    top_adr_diff_12 = player_stats.sort_values('adr_diff', ascending=False).head(12)
    adr_colors = [team_colors[team] for team in top_adr_diff_12['team']]
    
    bars3 = ax3.barh(range(len(top_adr_diff_12)), top_adr_diff_12['adr_diff'], color=adr_colors)
    ax3.set_yticks(range(len(top_adr_diff_12)))
    ax3.set_yticklabels([f"{name}\n({team})" for name, team in 
                        zip(top_adr_diff_12.index, top_adr_diff_12['team'])], fontsize=7)
    ax3.set_xlabel('ADR Difference', fontsize=10, fontweight='bold')
    ax3.set_title('📊 Top 12 ADR Difference', fontsize=11, fontweight='bold', pad=15)
    ax3.grid(axis='x', alpha=0.3)
    
    # Add value labels
    for i, bar in enumerate(bars3):
        width = bar.get_width()
        ax3.text(width + 0.5, bar.get_y() + bar.get_height()/2, 
                f'{width:.1f}', ha='left', va='center', fontsize=7, fontweight='bold')
    
    # 4. Opening Kills per Round (Middle Left)
    ax4 = plt.subplot(3, 3, 4)
    if 'opening_kills_per_round' in player_stats.columns:
        top_opening_12 = player_stats.sort_values('opening_kills_per_round', ascending=False).head(12)
        opening_colors = [team_colors[team] for team in top_opening_12['team']]
        
        bars4 = ax4.bar(range(len(top_opening_12)), top_opening_12['opening_kills_per_round'], color=opening_colors)
        ax4.set_xticks(range(len(top_opening_12)))
        ax4.set_xticklabels([f"{name}\n({team})" for name, team in 
                            zip(top_opening_12.index, top_opening_12['team'])], 
                           rotation=45, ha='right', fontsize=6)
        ax4.set_ylabel('Opening Kills/Round', fontsize=10, fontweight='bold')
        ax4.set_title('🎯 Top 12 Opening Kills/Round', fontsize=11, fontweight='bold', pad=15)
        ax4.grid(axis='y', alpha=0.3)
        
        # Add value labels
        for i, bar in enumerate(bars4):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2, height + 0.002,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=7, fontweight='bold')
    else:
        ax4.text(0.5, 0.5, 'Opening Kills Data\nNot Available', ha='center', va='center', 
                transform=ax4.transAxes, fontsize=12)
    
    # 5. Team Performance Comparison (Middle Center)
    ax5 = plt.subplot(3, 3, 5)
    team_kd_data = team_stats.sort_values('avg_kd', ascending=True)
    bars5 = ax5.barh(range(len(team_kd_data)), team_kd_data['avg_kd'],
                     color=[team_colors[team] for team in team_kd_data.index])
    ax5.set_yticks(range(len(team_kd_data)))
    ax5.set_yticklabels(team_kd_data.index, fontsize=9)
    ax5.set_xlabel('Team Avg K/D', fontsize=10, fontweight='bold')
    ax5.set_title('🏆 Team K/D Rankings', fontsize=11, fontweight='bold', pad=15)
    ax5.grid(axis='x', alpha=0.3)
    
    # Add value labels
    for i, bar in enumerate(bars5):
        width = bar.get_width()
        ax5.text(width + 0.005, bar.get_y() + bar.get_height()/2, 
                f'{width:.3f}', ha='left', va='center', fontsize=8, fontweight='bold')
    
    # 6. Team Flash Assists (Middle Right)
    ax6 = plt.subplot(3, 3, 6)
    team_flash_data = team_stats.sort_values('flash_assists_per_round', ascending=True)
    bars6 = ax6.barh(range(len(team_flash_data)), team_flash_data['flash_assists_per_round'],
                     color=[team_colors[team] for team in team_flash_data.index])
    ax6.set_yticks(range(len(team_flash_data)))
    ax6.set_yticklabels(team_flash_data.index, fontsize=9)
    ax6.set_xlabel('Flash Assists/Round', fontsize=10, fontweight='bold')
    ax6.set_title('💡 Team Flash Assists', fontsize=11, fontweight='bold', pad=15)
    ax6.grid(axis='x', alpha=0.3)
    
    # Add value labels
    for i, bar in enumerate(bars6):
        width = bar.get_width()
        ax6.text(width + 0.005, bar.get_y() + bar.get_height()/2, 
                f'{width:.3f}', ha='left', va='center', fontsize=8, fontweight='bold')
    
    # 7. Multi-metric Performance Scatter (Bottom Span)
    ax7 = plt.subplot2grid((3, 3), (2, 0), colspan=3)
    
    # Filter for players with sufficient matches
    active_players = player_stats[player_stats['matches_played'] >= 5]
    
    scatter_colors = [team_colors[team] for team in active_players['team']]
    scatter = ax7.scatter(active_players['kd_ratio'], active_players['swing'],
                         s=active_players['flash_assists_per_round']*2000,  # Size by flash assists
                         c=scatter_colors, alpha=0.7, edgecolors='black', linewidth=0.5)
    
    # Annotate players
    for player, stats in active_players.iterrows():
        ax7.annotate(f"{player}", 
                    (stats['kd_ratio'], stats['swing']),
                    xytext=(3, 3), textcoords='offset points', fontsize=7,
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
    
    ax7.set_xlabel('K/D Ratio', fontsize=12, fontweight='bold')
    ax7.set_ylabel('Swing Rating', fontsize=12, fontweight='bold')
    ax7.set_title('🎯 Multi-Metric Performance\n(Bubble Size = Flash Assists/Round, Color = Team)', 
                 fontsize=13, fontweight='bold', pad=20)
    ax7.grid(True, alpha=0.3)
    
    # Add legend for teams
    legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                                 markerfacecolor=color, markersize=10, label=team)
                      for team, color in team_colors.items()]
    ax7.legend(handles=legend_elements, loc='upper left', fontsize=9, 
              bbox_to_anchor=(0, 1), ncol=3)
    
    plt.tight_layout(pad=3.0)
    plt.show()

def main():
    """Main analysis function for corrected team rosters."""
    
    # Load and filter data with correct team assignments
    df, player_team_map, team_rosters = load_and_filter_data('player_statistics_cleaned.csv')
    
    # Calculate player statistics
    player_stats = calculate_player_stats(df)
    
    # Display top performers
    display_top_performers(player_stats)
    
    # Analyze team performance
    team_stats = analyze_team_performance(df, player_stats)
    
    # Create visualizations
    print("📈 CREATING COMPREHENSIVE VISUALIZATIONS WITH CORRECT TEAMS...")
    create_comprehensive_visualization(df, player_stats, team_stats)
    
    # Final insights
    print("\n" + "="*90)
    print("🎯 KEY INSIGHTS (CORRECTED TEAM ROSTERS):")
    print("="*90)
    
    # Top performers from target teams
    top_kd_player = player_stats.iloc[0]
    top_swing_player = player_stats.sort_values('swing', ascending=False).iloc[0]
    top_flash_player = player_stats.sort_values('flash_assists_per_round', ascending=False).iloc[0]
    top_adr_diff_player = player_stats.sort_values('adr_diff', ascending=False).iloc[0]
    top_opening_player = player_stats.sort_values('opening_kills_per_round', ascending=False).iloc[0]
    most_active = player_stats.sort_values('matches_played', ascending=False).iloc[0]
    
    print(f"• Best K/D: {top_kd_player.name} ({top_kd_player['team']}) - {top_kd_player['kd_ratio']:.3f}")
    print(f"• Best Swing: {top_swing_player.name} ({top_swing_player['team']}) - {top_swing_player['swing']:.2f}")
    print(f"• Best Flash Support: {top_flash_player.name} ({top_flash_player['team']}) - {top_flash_player['flash_assists_per_round']:.3f}/round")
    print(f"• Best ADR Difference: {top_adr_diff_player.name} ({top_adr_diff_player['team']}) - +{top_adr_diff_player['adr_diff']:.1f}")
    print(f"• Best Opening Kills: {top_opening_player.name} ({top_opening_player['team']}) - {top_opening_player['opening_kills_per_round']:.3f}/round")
    print(f"• Most Active: {most_active.name} ({most_active['team']}) - {most_active['matches_played']:.0f} matches")
    
    # Team insights
    best_team = team_stats.iloc[0]
    best_flash_team = team_stats.sort_values('flash_assists_per_round', ascending=False).iloc[0]
    best_opening_team = team_stats.sort_values('opening_kd_ratio', ascending=False).iloc[0]
    
    print(f"• Best Team (K/D): {best_team.name} - {best_team['avg_kd']:.3f}")
    print(f"• Best Flash Team: {best_flash_team.name} - {team_stats.loc[best_flash_team.name, 'flash_assists_per_round']:.3f}/round")
    print(f"• Best Opening Team: {best_opening_team.name} - {team_stats.loc[best_opening_team.name, 'opening_kd_ratio']:.2f} OK/OD ratio")
    
    print(f"• Total players analyzed: {len(player_stats)}")
    print(f"• Total match records: {len(df):,}")
    print(f"• Teams: {', '.join(sorted(team_rosters.keys()))}")
    
    print("\n🏆 Analysis complete")
    for team, players in team_rosters.items():
        print(f"   {team}: {', '.join(players)}")

if __name__ == "__main__":
    main()