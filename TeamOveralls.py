import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load the data
df = pd.read_csv('round_analysis.csv')
print(f"Original dataset shape: {df.shape}")

# Remove de_anubis map
df_filtered = df[df['map_name'] != 'de_anubis'].copy()
print(f"After removing de_anubis: {df_filtered.shape}")
print(f"Remaining maps: {df_filtered['map_name'].unique()}")

# Define colors - T side brick red, CT side dark blue
T_COLOR = '#B22222'  # Brick red
CT_COLOR = '#1E3A8A'  # Dark blue

# Calculate win rates by map and side
win_rates = df_filtered.groupby(['map_name', 'round_winner']).size().unstack(fill_value=0)
win_rates_pct = win_rates.div(win_rates.sum(axis=1), axis=0) * 100

print("\nWin rates (%) by map:")
print(win_rates_pct.round(2))

# ====================================================================================
# PLOT 1: Map Balance Analysis
# ====================================================================================

fig1, axes1 = plt.subplots(2, 2, figsize=(15, 12))
fig1.suptitle('Counter-Strike Round Analysis: T vs CT Win Rates by Map', fontsize=16, fontweight='bold')

# 1. Stacked bar chart
ax1 = axes1[0, 0]
win_rates_pct.plot(kind='bar', stacked=True, ax=ax1, color=[CT_COLOR, T_COLOR], width=0.8)
ax1.set_title('Win Rate Distribution by Map', fontweight='bold')
ax1.set_xlabel('Map')
ax1.set_ylabel('Win Rate (%)')
ax1.legend(['CT', 'T'], title='Side')
ax1.tick_params(axis='x', rotation=45)

# 2. Side-by-side bar chart
ax2 = axes1[0, 1]
win_rates_pct.plot(kind='bar', ax=ax2, color=[CT_COLOR, T_COLOR], width=0.8)
ax2.set_title('T vs CT Win Rates by Map', fontweight='bold')
ax2.set_xlabel('Map')
ax2.set_ylabel('Win Rate (%)')
ax2.legend(['CT', 'T'], title='Side')
ax2.tick_params(axis='x', rotation=45)
ax2.axhline(y=50, color='black', linestyle='--', alpha=0.7)

# 3. Heatmap
ax3 = axes1[1, 0]
sns.heatmap(win_rates_pct.T, annot=True, fmt='.1f', cmap='RdYlBu_r', 
            ax=ax3, cbar_kws={'label': 'Win Rate (%)'})
ax3.set_title('Win Rate Heatmap', fontweight='bold')
ax3.set_xlabel('Map')
ax3.set_ylabel('Side')

# 4. CT Advantage/Disadvantage
ax4 = axes1[1, 1]
ct_advantage = win_rates_pct['CT'] - win_rates_pct['T']
colors = [T_COLOR if x < 0 else CT_COLOR for x in ct_advantage]
bars = ax4.bar(range(len(ct_advantage)), ct_advantage, color=colors, alpha=0.7)
ax4.set_title('CT Advantage/Disadvantage by Map', fontweight='bold')
ax4.set_xlabel('Map')
ax4.set_ylabel('CT Win Rate - T Win Rate (%)')
ax4.axhline(y=0, color='black', linestyle='-', alpha=0.8)
ax4.set_xticks(range(len(ct_advantage)))
ax4.set_xticklabels(ct_advantage.index, rotation=45)

# Add value labels
for i, (bar, value) in enumerate(zip(bars, ct_advantage)):
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2., height + (1 if height > 0 else -3),
             f'{value:.1f}%', ha='center', va='bottom' if height > 0 else 'top', fontweight='bold')

plt.tight_layout()
plt.show()

# ====================================================================================
# PLOT 2: Team Pistol Analysis
# ====================================================================================

# Filter for pistol rounds and target teams
pistol_rounds = df_filtered[
    (df_filtered['round_number'] == 1) & 
    (~df_filtered['team_name_win'].isin(['CT', 'TERRORIST']))
].copy()

target_teams = {
    'Team Vitality': 'Vitality',
    'Aurora Gaming': 'Aurora',
    'MOUZ': 'MOUZ', 
    'Team Spirit': 'Spirit',
    'Virtus.pro': 'Virtus.pro',
    'Astralis': 'Astralis',
    'Team Liquid': 'Liquid',
    'The MongolZ': 'MongolZ'
}

team_pistol_data = pistol_rounds[pistol_rounds['team_name_win'].isin(target_teams.keys())].copy()

print(f"\n{'='*60}")
print("PISTOL ROUND ANALYSIS")
print(f"{'='*60}")
print(f"Total pistol rounds for target teams: {len(team_pistol_data)}")

if len(team_pistol_data) > 0:
    fig2, axes2 = plt.subplots(2, 2, figsize=(16, 12))
    fig2.suptitle('Team-Specific Pistol Round (Round 1) Win Rates by Map', fontsize=16, fontweight='bold')

    # Calculate pistol win rates by team and map
    pistol_analysis = []
    for team_full, team_short in target_teams.items():
        team_data = team_pistol_data[team_pistol_data['team_name_win'] == team_full]
        if len(team_data) > 0:
            for map_name in team_data['map_name'].unique():
                map_data = team_data[team_data['map_name'] == map_name]
                total_pistols = len(map_data)
                ct_wins = len(map_data[map_data['round_winner'] == 'CT'])
                t_wins = len(map_data[map_data['round_winner'] == 'T'])
                
                pistol_analysis.append({
                    'team': team_short,
                    'team_full': team_full,
                    'map': map_name,
                    'total_pistols': total_pistols,
                    'ct_wins': ct_wins,
                    't_wins': t_wins,
                    'ct_rate': (ct_wins / total_pistols * 100) if total_pistols > 0 else 0,
                    't_rate': (t_wins / total_pistols * 100) if total_pistols > 0 else 0
                })
    
    pistol_df = pd.DataFrame(pistol_analysis)
    
    if len(pistol_df) > 0:
        # Team overall stats
        team_overall = pistol_df.groupby('team').agg({
            'total_pistols': 'sum',
            'ct_wins': 'sum',
            't_wins': 'sum'
        }).reset_index()
        
        team_overall['ct_rate'] = team_overall['ct_wins'] / team_overall['total_pistols'] * 100
        team_overall['t_rate'] = team_overall['t_wins'] / team_overall['total_pistols'] * 100

        # 1. Overall pistol win rate by team
        ax1 = axes2[0, 0]
        x = np.arange(len(team_overall))
        width = 0.35
        
        bars1 = ax1.bar(x - width/2, team_overall['ct_rate'], width, label='CT Side', color=CT_COLOR, alpha=0.8)
        bars2 = ax1.bar(x + width/2, team_overall['t_rate'], width, label='T Side', color=T_COLOR, alpha=0.8)
        
        ax1.set_title('Overall Pistol Round Win Rate by Side', fontweight='bold')
        ax1.set_xlabel('Team')
        ax1.set_ylabel('Win Rate (%)')
        ax1.set_xticks(x)
        ax1.set_xticklabels(team_overall['team'], rotation=45)
        ax1.legend()
        ax1.grid(axis='y', alpha=0.3)
        
        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                             f'{height:.1f}%', ha='center', va='bottom', fontsize=8)

        # 2. Total pistol rounds played per team with win rates
        ax2 = axes2[0, 1]
        all_pistol_rounds = df_filtered[df_filtered['round_number'] == 1].copy()
        
        team_totals = []
        for team_full, team_short in target_teams.items():
            wins = len(team_pistol_data[team_pistol_data['team_name_win'] == team_full])
            
            team_demos = set()
            for _, row in all_pistol_rounds.iterrows():
                demo_file = str(row['demo_file']).lower()
                team_variations = [
                    team_full.lower().replace(' ', '-'),
                    team_full.lower().replace(' ', ''),
                    team_short.lower()
                ]
                
                for variation in team_variations:
                    if variation in demo_file:
                        team_demos.add(row['demo_file'])
                        break
            
            total_played = len(team_demos)
            win_rate = (wins / total_played * 100) if total_played > 0 else 0
            
            if total_played > 0:
                team_totals.append({
                    'team': team_short,
                    'wins': wins,
                    'total_played': total_played,
                    'win_rate': win_rate
                })
        
        if team_totals:
            team_totals_df = pd.DataFrame(team_totals).sort_values('total_played', ascending=True)
            
            bars = ax2.barh(team_totals_df['team'], team_totals_df['total_played'], color='#2563EB', alpha=0.7)
            ax2.set_title('Total Pistol Rounds Played by Team (Win Rate)', fontweight='bold')
            ax2.set_xlabel('Number of Pistol Rounds Played')
            ax2.set_ylabel('Team')
            
            for i, (bar, row) in enumerate(zip(bars, team_totals_df.itertuples())):
                width = bar.get_width()
                label = f'{int(width)} ({row.win_rate:.1f}%)'
                ax2.text(width + 0.2, bar.get_y() + bar.get_height()/2.,
                         label, ha='left', va='center', fontweight='bold', fontsize=9)

        # 3. Heatmap - Team pistol performance by map
        ax3 = axes2[1, 0]
        pivot_data = pistol_df.pivot_table(values='total_pistols', index='team', columns='map', fill_value=0)
        
        if not pivot_data.empty:
            pivot_data_int = pivot_data.astype(int)
            sns.heatmap(pivot_data_int, annot=True, fmt='d', cmap='YlOrRd', 
                       ax=ax3, cbar_kws={'label': 'Pistol Rounds Played'})
            ax3.set_title('Pistol Rounds Played: Team vs Map', fontweight='bold')
            ax3.set_xlabel('Map')
            ax3.set_ylabel('Team')

        # 4. Side preference in pistol rounds
        ax4 = axes2[1, 1]
        team_side_pref = team_overall.copy()
        team_side_pref['ct_pref'] = team_side_pref['ct_rate'] - team_side_pref['t_rate']
        team_side_pref = team_side_pref.sort_values('ct_pref')
        
        colors = [CT_COLOR if x > 0 else T_COLOR for x in team_side_pref['ct_pref']]
        bars = ax4.barh(team_side_pref['team'], team_side_pref['ct_pref'], color=colors, alpha=0.7)
        ax4.set_title('Pistol Round Side Performance Difference\n(CT% - T%)', fontweight='bold')
        ax4.set_xlabel('CT Win Rate - T Win Rate (%)')
        ax4.set_ylabel('Team')
        ax4.axvline(x=0, color='black', linestyle='-', alpha=0.8)
        
        for bar in bars:
            width = bar.get_width()
            label_x = width + (2 if width > 0 else -2)
            ax4.text(label_x, bar.get_y() + bar.get_height()/2.,
                     f'{width:.1f}%', ha='left' if width > 0 else 'right', 
                     va='center', fontweight='bold', fontsize=9)

    plt.tight_layout()
    plt.show()

    # ====================================================================================
    # PLOT 3: Round 2 Conversion Analysis
    # ====================================================================================
    
    print(f"\nCalculating Round 2 performance after pistol wins...")
    
    round2_data = df_filtered[df_filtered['round_number'] == 2].copy()
    pistol_to_round2_analysis = []
    
    for team_full, team_short in target_teams.items():
        team_pistol_wins = team_pistol_data[team_pistol_data['team_name_win'] == team_full]
        
        pistol_wins_count = 0
        round2_wins_after_pistol = 0
        
        for _, pistol_round in team_pistol_wins.iterrows():
            demo_file = pistol_round['demo_file']
            round2_match = round2_data[round2_data['demo_file'] == demo_file]
            
            if not round2_match.empty:
                pistol_wins_count += 1
                if team_full in round2_match['team_name_win'].values:
                    round2_wins_after_pistol += 1
        
        if pistol_wins_count > 0:
            conversion_rate = (round2_wins_after_pistol / pistol_wins_count) * 100
            pistol_to_round2_analysis.append({
                'team': team_short,
                'team_full': team_full,
                'pistol_wins': pistol_wins_count,
                'round2_wins_after_pistol': round2_wins_after_pistol,
                'conversion_rate': conversion_rate
            })
    
    if pistol_to_round2_analysis:
        fig3, axes3 = plt.subplots(2, 2, figsize=(16, 12))
        fig3.suptitle('Round 2 Performance After Pistol Round Wins', fontsize=16, fontweight='bold')
        
        conversion_df = pd.DataFrame(pistol_to_round2_analysis)
        
        # 1. Round 2 conversion rate
        ax1 = axes3[0, 0]
        conversion_sorted = conversion_df.sort_values('conversion_rate', ascending=True)
        colors = ['#DC2626' if x < 50 else '#059669' if x > 70 else '#D97706' for x in conversion_sorted['conversion_rate']]
        
        bars = ax1.barh(conversion_sorted['team'], conversion_sorted['conversion_rate'], color=colors, alpha=0.8)
        ax1.set_title('Round 2 Win Rate After Pistol Win', fontweight='bold')
        ax1.set_xlabel('Round 2 Win Rate After Pistol Win (%)')
        ax1.set_ylabel('Team')
        ax1.axvline(x=50, color='black', linestyle='--', alpha=0.5)
        ax1.axvline(x=70, color='green', linestyle='--', alpha=0.5)
        ax1.grid(axis='x', alpha=0.3)
        
        for bar in bars:
            width = bar.get_width()
            ax1.text(width + 1, bar.get_y() + bar.get_height()/2.,
                     f'{width:.1f}%', ha='left', va='center', fontweight='bold', fontsize=9)

        # 2. Sample sizes
        ax2 = axes3[0, 1]
        sample_sorted = conversion_df.sort_values('pistol_wins', ascending=True)
        bars2 = ax2.barh(sample_sorted['team'], sample_sorted['pistol_wins'], color='#3B82F6', alpha=0.7)
        ax2.set_title('Sample Size: Pistol Rounds Won', fontweight='bold')
        ax2.set_xlabel('Number of Pistol Rounds Won')
        ax2.set_ylabel('Team')
        
        for bar in bars2:
            width = bar.get_width()
            ax2.text(width + 0.1, bar.get_y() + bar.get_height()/2.,
                     f'{int(width)}', ha='left', va='center', fontweight='bold', fontsize=9)

        # 3. Scatter plot
        ax3 = axes3[1, 0]
        scatter = ax3.scatter(conversion_df['pistol_wins'], conversion_df['conversion_rate'], 
                             c=conversion_df['conversion_rate'], cmap='RdYlGn', s=100, alpha=0.8)
        
        for _, row in conversion_df.iterrows():
            ax3.annotate(row['team'], (row['pistol_wins'], row['conversion_rate']), 
                       xytext=(5, 5), textcoords='offset points', fontsize=9, fontweight='bold')
        
        ax3.set_title('Reliability vs Performance', fontweight='bold')
        ax3.set_xlabel('Sample Size (Pistol Wins)')
        ax3.set_ylabel('Round 2 Conversion Rate (%)')
        ax3.axhline(y=50, color='black', linestyle='--', alpha=0.5)
        ax3.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax3, label='Conversion Rate (%)')

        # 4. Summary table
        ax4 = axes3[1, 1]
        ax4.axis('tight')
        ax4.axis('off')
        
        table_data = [['Team', 'Pistol\nWins', 'R2 Wins\nAfter', 'Conversion\nRate']]
        for _, row in conversion_df.sort_values('conversion_rate', ascending=False).iterrows():
            table_data.append([
                row['team'],
                f"{int(row['pistol_wins'])}",
                f"{int(row['round2_wins_after_pistol'])}",
                f"{row['conversion_rate']:.1f}%"
            ])
        
        cell_colors = [['lightgray'] * 4]
        for _, row in conversion_df.sort_values('conversion_rate', ascending=False).iterrows():
            rate = row['conversion_rate']
            if rate >= 70:
                row_color = ['lightgreen', 'white', 'white', 'lightgreen']
            elif rate >= 50:
                row_color = ['lightyellow', 'white', 'white', 'lightyellow']
            else:
                row_color = ['lightcoral', 'white', 'white', 'lightcoral']
            cell_colors.append(row_color)
        
        table = ax4.table(cellText=table_data, cellColours=cell_colors, loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 2)
        ax4.set_title('Round 2 Performance Summary', fontweight='bold', pad=20)

        plt.tight_layout()
        plt.show()

        # Print Round 2 conversion stats
        print(f"\nROUND 2 CONVERSION ANALYSIS:")
        print(f"{'Team':<12} {'Pistol Wins':<12} {'R2 After Pistol':<15} {'Conversion %':<12}")
        print("-" * 60)
        for _, row in conversion_df.sort_values('conversion_rate', ascending=False).iterrows():
            print(f"{row['team']:<12} {int(row['pistol_wins']):<12} {int(row['round2_wins_after_pistol']):<15} {row['conversion_rate']:<12.1f}")

    # Print detailed pistol stats
    print(f"\nDetailed Team Pistol Round Statistics:")
    print(f"{'Team':<12} {'Total':<6} {'CT Wins':<8} {'T Wins':<7} {'CT%':<6} {'T%':<6}")
    print("-" * 60)
    
    for _, row in team_overall.iterrows():
        print(f"{row['team']:<12} {int(row['total_pistols']):<6} {int(row['ct_wins']):<8} {int(row['t_wins']):<7} "
              f"{row['ct_rate']:<6.1f} {row['t_rate']:<6.1f}")

else:
    print("No pistol round data found for target teams.")

# ====================================================================================
# PLOT 4: Team Win Rates by Side per Map
# ====================================================================================

print(f"\n{'='*60}")
print("TEAM WIN RATES BY SIDE PER MAP ANALYSIS")
print(f"{'='*60}")

team_data = df_filtered[~df_filtered['team_name_win'].isin(['CT', 'TERRORIST'])].copy()
team_map_data = team_data[team_data['team_name_win'].isin(target_teams.keys())].copy()

if len(team_map_data) > 0:
    team_side_map_stats = []
    
    for team_full, team_short in target_teams.items():
        team_rounds = team_map_data[team_map_data['team_name_win'] == team_full]
        
        if len(team_rounds) > 0:
            for map_name in team_rounds['map_name'].unique():
                map_rounds = team_rounds[team_rounds['map_name'] == map_name]
                
                ct_wins = len(map_rounds[map_rounds['round_winner'] == 'CT'])
                t_wins = len(map_rounds[map_rounds['round_winner'] == 'T'])
                total_rounds = len(map_rounds)
                
                if total_rounds >= 5:  # Minimum sample size
                    team_side_map_stats.append({
                        'team': team_short,
                        'team_full': team_full,
                        'map': map_name.replace('de_', '').capitalize(),
                        'ct_wins': ct_wins,
                        't_wins': t_wins,
                        'total_rounds': total_rounds,
                        'ct_rate': (ct_wins / total_rounds * 100),
                        't_rate': (t_wins / total_rounds * 100)
                    })
    
    if team_side_map_stats:
        team_stats_df = pd.DataFrame(team_side_map_stats)
        
        fig4, axes4 = plt.subplots(2, 2, figsize=(20, 16))
        fig4.suptitle('Team Performance by Side per Map', fontsize=18, fontweight='bold')

        # 1. CT Win Rate Heatmap
        ax1 = axes4[0, 0]
        ct_pivot = team_stats_df.pivot_table(values='ct_rate', index='team', columns='map', fill_value=0)
        
        if not ct_pivot.empty:
            sns.heatmap(ct_pivot, annot=True, fmt='.1f', cmap='Blues', 
                       ax=ax1, cbar_kws={'label': 'CT Win Rate (%)'}, vmin=0, vmax=100)
            ax1.set_title('CT Side Win Rate by Team and Map', fontweight='bold', fontsize=14)

        # 2. T Win Rate Heatmap  
        ax2 = axes4[0, 1]
        t_pivot = team_stats_df.pivot_table(values='t_rate', index='team', columns='map', fill_value=0)
        
        if not t_pivot.empty:
            sns.heatmap(t_pivot, annot=True, fmt='.1f', cmap='Reds', 
                       ax=ax2, cbar_kws={'label': 'T Win Rate (%)'}, vmin=0, vmax=100)
            ax2.set_title('T Side Win Rate by Team and Map', fontweight='bold', fontsize=14)

        # 3. Side Preference Analysis
        ax3 = axes4[1, 0]
        team_stats_df['side_preference'] = team_stats_df['ct_rate'] - team_stats_df['t_rate']
        team_preference = team_stats_df.groupby('team').agg({
            'side_preference': 'mean',
            'total_rounds': 'sum'
        }).reset_index().sort_values('side_preference')
        
        colors = [CT_COLOR if x > 0 else T_COLOR for x in team_preference['side_preference']]
        bars = ax3.barh(team_preference['team'], team_preference['side_preference'], color=colors, alpha=0.8)
        ax3.set_title('Team Side Preference (Average CT% - T%)', fontweight='bold', fontsize=14)
        ax3.set_xlabel('Side Preference (CT Favored ← → T Favored)')
        ax3.axvline(x=0, color='black', linestyle='-', alpha=0.8)
        ax3.grid(axis='x', alpha=0.3)
        
        for bar in bars:
            width = bar.get_width()
            label_x = width + (1 if width > 0 else -1)
            ax3.text(label_x, bar.get_y() + bar.get_height()/2.,
                     f'{width:.1f}%', ha='left' if width > 0 else 'right', 
                     va='center', fontweight='bold', fontsize=10)

        # 4. Summary Table
        ax4 = axes4[1, 1]
        ax4.axis('tight')
        ax4.axis('off')
        
        team_summary = team_stats_df.groupby('team').agg({
            'ct_rate': 'mean',
            't_rate': 'mean', 
            'total_rounds': 'sum'
        }).reset_index().sort_values('total_rounds', ascending=False)
        
        table_data = [['Team', 'Avg CT%', 'Avg T%', 'Total\nRounds', 'Side\nPreference']]
        for _, row in team_summary.iterrows():
            pref = row['ct_rate'] - row['t_rate']
            pref_text = f"CT +{pref:.1f}" if pref > 0 else f"T +{abs(pref):.1f}"
            
            table_data.append([
                row['team'],
                f"{row['ct_rate']:.1f}%",
                f"{row['t_rate']:.1f}%", 
                f"{int(row['total_rounds'])}",
                pref_text
            ])
        
        cell_colors = [['lightgray'] * 5]
        for _, row in team_summary.iterrows():
            overall_rate = (row['ct_rate'] + row['t_rate']) / 2
            if overall_rate >= 60:
                base_color = 'lightgreen'
            elif overall_rate >= 45:
                base_color = 'lightyellow'  
            else:
                base_color = 'lightcoral'
            cell_colors.append([base_color, 'white', 'white', 'white', 'white'])
        
        table = ax4.table(cellText=table_data, cellColours=cell_colors, loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1.2, 2.5)
        ax4.set_title('Team Performance Summary', fontweight='bold', fontsize=14, pad=20)

        plt.tight_layout()
        plt.show()

        # Print detailed breakdown
        print(f"\nDETAILED TEAM-MAP-SIDE BREAKDOWN:")
        print(f"{'Team':<12} {'Map':<10} {'CT Rate':<8} {'T Rate':<8} {'Total':<6} {'Side Pref':<10}")
        print("-" * 65)
        
        for _, row in team_stats_df.sort_values(['team', 'map']).iterrows():
            pref = "CT" if row['ct_rate'] > row['t_rate'] else "T"
            diff = abs(row['ct_rate'] - row['t_rate'])
            
            print(f"{row['team']:<12} {row['map']:<10} {row['ct_rate']:<8.1f} {row['t_rate']:<8.1f} "
                  f"{row['total_rounds']:<6} {pref} +{diff:.1f}")

        econ_data = df_filtered.copy()

# ====================================================================================
# PLOT 5: Economy Analysis (by SIDE, not team — dataset lacks team_name_t/ct)
# ====================================================================================

# Build per-side rows using what we have
econ_side = pd.concat([
    pd.DataFrame({
        'map_name': df_filtered['map_name'],
        'side': 'T',
        'round_type': df_filtered['t_round_type'],
        'won': (df_filtered['round_winner'] == 'T')
    }),
    pd.DataFrame({
        'map_name': df_filtered['map_name'],
        'side': 'CT',
        'round_type': df_filtered['ct_round_type'],
        'won': (df_filtered['round_winner'] == 'CT')
    })
], ignore_index=True)

# Keep only relevant economy categories
econ_side = econ_side[econ_side['round_type'].isin(['force', 'low_gun', 'full_gun'])].copy()

# Winrates by SIDE × MAP × ROUND_TYPE
econ_stats = (
    econ_side.groupby(['map_name','side','round_type'])
    .agg(rounds=('won','size'), win_rate=('won','mean'))
    .reset_index()
)
econ_stats['win_rate'] = econ_stats['win_rate'] * 100

# -----------------------
# A) Force-buy winrates by map (T vs CT)
# -----------------------
force_stats = econ_stats[econ_stats['round_type']=='force'].copy()
if not force_stats.empty:
    plt.figure(figsize=(12,6))
    sns.barplot(data=force_stats, x='map_name', y='win_rate', hue='side')
    plt.title('Force Buy Win Rates by Map (Side-corrected)', fontweight='bold')
    plt.ylabel('Win Rate (%)')
    plt.xlabel('Map')
    plt.xticks(rotation=45)
    plt.ylim(0,100)
    plt.legend(title='Side')
    plt.tight_layout()
    plt.show()
else:
    print("No 'force' rounds present.")

# -----------------------
# B) Force vs Low_gun (per map, per side)
# -----------------------
cmp_pivot = econ_stats.pivot_table(
    index=['map_name','side'],
    columns='round_type',
    values='win_rate'
).reset_index()

if {'force','low_gun'}.issubset(cmp_pivot.columns):
    plt.figure(figsize=(8,8))
    ax = plt.gca()
    sns.scatterplot(data=cmp_pivot, x='force', y='low_gun', style='side', s=90, alpha=0.85)
    ax.plot([0,100],[0,100],'k--',alpha=0.6)
    ax.set_xlim(0,100); ax.set_ylim(0,100)
    ax.set_title('Force vs Low_gun Win Rates (by Map & Side)', fontweight='bold')
    ax.set_xlabel('Force Buy Win Rate (%)')
    ax.set_ylabel('Low_gun Win Rate (%)')
    # annotate points lightly
    for _, r in cmp_pivot.dropna(subset=['force','low_gun']).iterrows():
        ax.annotate(f"{r['map_name']} ({r['side']})", (r['force'], r['low_gun']),
                    xytext=(4,4), textcoords='offset points', fontsize=8)
    plt.tight_layout(); plt.show()

# -----------------------
# C) Low_gun vs Full_gun (per map, per side)
# -----------------------
if {'low_gun','full_gun'}.issubset(cmp_pivot.columns):
    plt.figure(figsize=(8,8))
    ax = plt.gca()
    sns.scatterplot(data=cmp_pivot, x='low_gun', y='full_gun', style='side', s=90, alpha=0.85)
    ax.plot([0,100],[0,100],'k--',alpha=0.6)
    ax.set_xlim(0,100); ax.set_ylim(0,100)
    ax.set_title('Low_gun vs Full_gun Win Rates (by Map & Side)', fontweight='bold')
    ax.set_xlabel('Low_gun Win Rate (%)')
    ax.set_ylabel('Full_gun Win Rate (%)')
    for _, r in cmp_pivot.dropna(subset=['low_gun','full_gun']).iterrows():
        ax.annotate(f"{r['map_name']} ({r['side']})", (r['low_gun'], r['full_gun']),
                    xytext=(4,4), textcoords='offset points', fontsize=8)
    plt.tight_layout(); plt.show()


# ====================================================================================
# SUMMARY STATISTICS
# ====================================================================================

print(f"\n{'='*60}")
print("SUMMARY STATISTICS")
print(f"{'='*60}")

print(f"\nTotal rounds analyzed: {len(df_filtered):,}")
print(f"Total maps: {df_filtered['map_name'].nunique()}")

print(f"\nOverall win rates:")
overall_wins = df_filtered['round_winner'].value_counts()
overall_pct = (overall_wins / overall_wins.sum() * 100)
print(f"CT: {overall_pct['CT']:.1f}% ({overall_wins['CT']:,} rounds)")
print(f"T:  {overall_pct['T']:.1f}% ({overall_wins['T']:,} rounds)")

print(f"\nMost CT-sided maps:")
ct_sided = (win_rates_pct['CT'] - win_rates_pct['T']).sort_values(ascending=False)
for map_name, advantage in ct_sided.head(3).items():
    print(f"{map_name}: {advantage:.1f}% CT advantage")

print(f"\nMost T-sided maps:")
for map_name, advantage in ct_sided.tail(3).items():
    print(f"{map_name}: {abs(advantage):.1f}% T advantage")

print(f"\n{'='*60}")
print("DETAILED MAP BREAKDOWN")
print(f"{'='*60}")

for map_name in sorted(df_filtered['map_name'].unique()):
    map_data = df_filtered[df_filtered['map_name'] == map_name]
    total_rounds = len(map_data)
    ct_wins = len(map_data[map_data['round_winner'] == 'CT'])
    t_wins = len(map_data[map_data['round_winner'] == 'T'])
    
    print(f"\n{map_name.upper()}")
    print(f"  Total rounds: {total_rounds}")
    print(f"  CT wins: {ct_wins} ({ct_wins/total_rounds*100:.1f}%)")
    print(f"  T wins:  {t_wins} ({t_wins/total_rounds*100:.1f}%)")
    
    win_methods = map_data['win_method'].value_counts().head(3)
    print(f"  Top win methods:")
    for method, count in win_methods.items():
        print(f"    {method}: {count} ({count/total_rounds*100:.1f}%)")

print(f"\n{'='*60}")
print("✅ ANALYSIS COMPLETE")
print(f"{'='*60}")
print("📊 All 4 plots have been generated and displayed")
print(f"{'='*60}")