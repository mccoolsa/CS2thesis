import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import warnings
warnings.filterwarnings('ignore')

class CorrectedCS2MatchPredictor:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_scaler = None
        self.team_encoder = LabelEncoder()
        self.map_encoder = LabelEncoder()
        self.feature_columns = []
        self.team_stats = {}
        self.map_stats = {}
        
        # Define the 8 teams we're focusing on
        self.target_teams = [
            'Vitality', 'Astralis', 'Aurora', 'Virtus.pro', 
            'Team Spirit', 'Team Liquid', 'The MongolZ', 'MOUZ'
        ]
        
    
        self.map_ct_bias = {
            'de_ancient': 0.527,   
            'de_dust2': 0.507,     
            'de_inferno': 0.507,   
            'de_mirage': 0.536,    
            'de_nuke': 0.579,      
            'de_overpass': 0.599, 
            'de_train': 0.591     
        }
        
    
        self.team_elo_ratings = {
            'Vitality': 2129,
            'MOUZ': 1937,
            'Team Spirit': 1969,
            'Aurora': 1792,
            'The MongolZ': 1813,
            'Team Liquid': 1651,
            'Virtus.pro': 1553,
            'Astralis': 1679
        }
        
    def load_data(self, player_stats_path='player_statistics.csv', 
                  round_analysis_path='round_analysis.csv'):
        """Load and preprocess the CS2 data"""
        print("Loading data...")
        
        # Load datasets
        self.player_stats = pd.read_csv(player_stats_path)
        self.round_analysis = pd.read_csv(round_analysis_path)
        
        print(f"Loaded {len(self.player_stats)} player records and {len(self.round_analysis)} round records")
        
        # Clean and preprocess data
        self._preprocess_data()
        
    def _preprocess_data(self):
        """Preprocess and clean the data"""
        print("Preprocessing data...")
        
        # Handle missing values
        self.player_stats = self.player_stats.fillna(0)
        self.round_analysis = self.round_analysis.fillna(0)
        
        # Convert boolean columns in round_analysis
        bool_cols = ['bomb_planted', 'bomb_defused', 'bomb_exploded']
        for col in bool_cols:
            if col in self.round_analysis.columns:
                self.round_analysis[col] = self.round_analysis[col].astype(int)
        
        # Extract team information from demo_file or use team_name_win
        if 'team_name_win' in self.round_analysis.columns:
            self._extract_team_mapping()
        else:
            self.player_stats['team'] = self._extract_team_from_filename(self.player_stats['demo_file'])
        
        # Filter for target teams only
        self._filter_target_teams()
        
        # Create match-level aggregations
        self._create_match_features()
        
    def _extract_team_mapping(self):
        """Extract team mapping from round analysis data"""
        team_mapping = {}
        
        for demo_file in self.round_analysis['demo_file'].unique():
            demo_rounds = self.round_analysis[self.round_analysis['demo_file'] == demo_file]
            winning_teams = demo_rounds['team_name_win'].unique()
            
            teams_in_match = []
            for team in winning_teams:
                if pd.notna(team) and team != '':
                    teams_in_match.append(team)
            
            filename_teams = self._extract_teams_from_filename(demo_file)
            teams_in_match.extend(filename_teams)
            
            teams_in_match = list(set([t for t in teams_in_match if t in self.target_teams]))
            
            if len(teams_in_match) >= 2:
                team_mapping[demo_file] = teams_in_match[:2]
            elif len(teams_in_match) == 1:
                other_team = self._infer_opposing_team(teams_in_match[0], demo_file)
                if other_team:
                    team_mapping[demo_file] = [teams_in_match[0], other_team]
        
        self.player_stats['team'] = self.player_stats['demo_file'].map(
            lambda x: team_mapping.get(x, ['Unknown'])[0] if x in team_mapping else 'Unknown'
        )
        
        self.team_mapping = team_mapping
        
    def _extract_teams_from_filename(self, filename):
        """Extract team names from filename"""
        teams_found = []
        filename_upper = filename.upper()
        
        for team in self.target_teams:
            team_variations = [team.upper(), team.replace(' ', '').upper(), team.replace('.', '').upper()]
            for variation in team_variations:
                if variation in filename_upper:
                    teams_found.append(team)
                    break
        
        return teams_found
    
    def _extract_team_from_filename(self, demo_files):
        """Extract team from demo filename - fallback method"""
        teams = []
        for filename in demo_files:
            found_teams = self._extract_teams_from_filename(filename)
            teams.append(found_teams[0] if found_teams else 'Unknown')
        return teams
    
    def _infer_opposing_team(self, known_team, demo_file):
        """Try to infer the opposing team from various sources"""
        filename_teams = self._extract_teams_from_filename(demo_file)
        for team in filename_teams:
            if team != known_team and team in self.target_teams:
                return team
        return None
    
    def _filter_target_teams(self):
        """Filter data to only include matches with target teams"""
        target_matches = set()
        
        if hasattr(self, 'team_mapping'):
            for demo_file, teams in self.team_mapping.items():
                if any(team in self.target_teams for team in teams):
                    target_matches.add(demo_file)
        else:
            for demo_file in self.player_stats['demo_file'].unique():
                if any(team in demo_file.upper() for team in [t.upper() for t in self.target_teams]):
                    target_matches.add(demo_file)
        
        print(f"Found {len(target_matches)} matches with target teams")
        
        self.player_stats = self.player_stats[self.player_stats['demo_file'].isin(target_matches)]
        self.round_analysis = self.round_analysis[self.round_analysis['demo_file'].isin(target_matches)]
        
    def _create_match_features(self):
        """Create match-level features with team-specific aggregations"""
        print("Creating enhanced match features...")
        
        player_agg_cols = {
            'kills': 'sum', 'deaths': 'sum', 'assists': 'sum', 'headshots': 'sum',
            'headshot_percentage': 'mean', 'adr': 'mean', 'adr_taken': 'mean',
            'adr_diff': 'mean', 'kpr': 'mean', 'kast': 'mean', 'swing': 'mean',
            'economy_eff': 'mean', 'equipment_ef': 'mean', 'equipment_ef_scaled': 'mean',
            'clutch_win_rate': 'mean', 'clutch_loss_rate': 'mean', 'rounds_won': 'sum',
            'rounds_lost': 'sum', 'mks': 'sum', 'flash_assists': 'sum',
            'traded_deaths': 'sum', 'equipment_lost': 'mean', 'opkpr': 'mean',
            'opdpr': 'mean', 'fapr': 'mean'
        }
        
        # Filter for existing columns
        existing_cols = {col: func for col, func in player_agg_cols.items() 
                        if col in self.player_stats.columns}
        
        print(f"Using {len(existing_cols)} aggregation columns")
        
        try:
            # Group by demo_file, map_name, team for team-specific stats
            grouped = self.player_stats.groupby(['demo_file', 'map_name', 'team']).agg(existing_cols)
            
            if hasattr(grouped.columns, 'levels'):
                new_column_names = [f"{col[0]}_{col[1]}" if len(col) > 1 and col[1] 
                                  else col[0] for col in grouped.columns]
                grouped.columns = new_column_names
            
            match_player_stats = grouped.reset_index()
            print(f"Match player stats shape: {match_player_stats.shape}")
            
        except Exception as e:
            print(f"Error in groupby aggregation: {e}")
            match_player_stats = self.player_stats[['demo_file', 'map_name', 'team']].drop_duplicates()
        
        # Create temporal features
        temporal_features = self._create_temporal_features()
        
        # Merge data
        if len(temporal_features) > 0 and len(match_player_stats) > 0:
            try:
                self.match_data = pd.merge(match_player_stats, temporal_features, 
                                         on=['demo_file', 'map_name'], how='inner')
                print(f"Successfully merged data: {len(self.match_data)} records")
            except Exception as e:
                print(f"Merge failed: {e}")
                self.match_data = match_player_stats.copy()
                self._add_default_temporal_features()
        else:
            self.match_data = match_player_stats.copy()
            self._add_default_temporal_features()
        
        # Calculate team performance statistics
        self._calculate_team_performance_stats()
        
    def _calculate_team_performance_stats(self):
        """Calculate team performance statistics using provided ELO ratings"""
        print("Calculating team performance statistics...")
        
        # Use provided ELO ratings for all teams
        for team in self.target_teams:
            team_data = self.match_data[self.match_data['team'] == team]
            elo_rating = self.team_elo_ratings.get(team, 1250)
            
            if len(team_data) > 0:
                self.team_stats[team] = {
                    'avg_kills': team_data['kills'].mean() if 'kills' in team_data.columns else 20,
                    'avg_deaths': team_data['deaths'].mean() if 'deaths' in team_data.columns else 20,
                    'avg_adr': team_data['adr'].mean() if 'adr' in team_data.columns else 75,
                    'avg_kast': team_data['kast'].mean() if 'kast' in team_data.columns else 0.7,
                    'matches_played': len(team_data),
                    'skill_rating': elo_rating
                }
            else:
                # Default stats for teams with no data
                self.team_stats[team] = {
                    'avg_kills': 20, 'avg_deaths': 20, 'avg_adr': 75, 'avg_kast': 0.7,
                    'matches_played': 0, 'skill_rating': elo_rating
                }
        
        # Map statistics (only for maps in our list)
        for map_name in self.match_data['map_name'].unique():
            if map_name in self.map_ct_bias:
                map_data = self.match_data[self.match_data['map_name'] == map_name]
                if len(map_data) > 0:
                    self.map_stats[map_name] = {
                        'ct_win_rate': self.map_ct_bias[map_name],  # Use your research data
                        'matches_played': len(map_data)
                    }
        
    def _add_default_temporal_features(self):
        """Add default temporal features - NO ARTIFICIAL CT BIAS"""
        np.random.seed(42)
        self.match_data['ct_max_streak'] = np.random.randint(1, 6, len(self.match_data))
        self.match_data['t_max_streak'] = np.random.randint(1, 6, len(self.match_data))
        self.match_data['ct_pistol_wins'] = np.random.randint(0, 3, len(self.match_data))
        self.match_data['t_pistol_wins'] = np.random.randint(0, 3, len(self.match_data))
        self.match_data['ct_avg_equipment'] = np.random.uniform(3000, 5000, len(self.match_data))
        self.match_data['t_avg_equipment'] = np.random.uniform(3000, 5000, len(self.match_data))
        
        # Generate match winners based ONLY on your map bias data
        match_winners = []
        for _, row in self.match_data.iterrows():
            map_name = row['map_name']
            ct_prob = self.map_ct_bias.get(map_name, 0.5)  # Use YOUR empirical data
            match_winners.append('CT' if np.random.random() < ct_prob else 'T')
        
        self.match_data['match_winner'] = match_winners
        
    def _create_temporal_features(self):
        """Create temporal features using actual round_analysis columns"""
        temporal_features = []
        
        for demo_file in self.round_analysis['demo_file'].unique():
            match_rounds = self.round_analysis[
                self.round_analysis['demo_file'] == demo_file
            ].sort_values('round_number')
            
            if len(match_rounds) == 0:
                continue
                
            features = {
                'demo_file': demo_file,
                'map_name': match_rounds['map_name'].iloc[0]
            }
            
            # Calculate features from round data
            if 'round_winner' in match_rounds.columns:
                ct_wins = (match_rounds['round_winner'] == 'CT').astype(int)
                t_wins = (match_rounds['round_winner'] == 'T').astype(int)
                
                features['ct_max_streak'] = self._max_consecutive(ct_wins)
                features['t_max_streak'] = self._max_consecutive(t_wins)
                
                # Pistol round performance
                if 'round_number' in match_rounds.columns:
                    pistol_rounds = match_rounds[match_rounds['round_number'].isin([1, 13])]  # CS2 halftime
                    features['ct_pistol_wins'] = (pistol_rounds['round_winner'] == 'CT').sum()
                    features['t_pistol_wins'] = (pistol_rounds['round_winner'] == 'T').sum()
                else:
                    features['ct_pistol_wins'] = np.random.randint(0, 3)
                    features['t_pistol_wins'] = np.random.randint(0, 3)
                
                # CORRECT CS2 MATCH WINNER LOGIC: FIRST TO 13 ROUNDS
                ct_rounds = ct_wins.sum()
                t_rounds = t_wins.sum()
                
                if ct_rounds >= 13:
                    features['match_winner'] = 'CT'
                elif t_rounds >= 13:
                    features['match_winner'] = 'T'
                elif ct_rounds + t_rounds > 0:
                    # For incomplete matches, use majority
                    features['match_winner'] = 'CT' if ct_rounds > t_rounds else 'T'
                else:
                    features['match_winner'] = 'CT'
                    
            else:
                # Default values using YOUR map bias data (no arbitrary 55/45)
                features.update({
                    'ct_max_streak': np.random.randint(1, 6),
                    't_max_streak': np.random.randint(1, 6),
                    'ct_pistol_wins': np.random.randint(0, 3),
                    't_pistol_wins': np.random.randint(0, 3)
                })
                
                # Use YOUR research data for match outcomes
                map_name = features['map_name']
                ct_prob = self.map_ct_bias.get(map_name, 0.5)
                np.random.seed(hash(demo_file) % 2**31)
                features['match_winner'] = 'CT' if np.random.random() < ct_prob else 'T'
            
            # Economy features
            if 'ct_equipment_value' in match_rounds.columns:
                features['ct_avg_equipment'] = match_rounds['ct_equipment_value'].mean()
                features['t_avg_equipment'] = match_rounds['t_equipment_value'].mean()
            else:
                features['ct_avg_equipment'] = np.random.uniform(3000, 5000)
                features['t_avg_equipment'] = np.random.uniform(3000, 5000)
            
            temporal_features.append(features)
        
        return pd.DataFrame(temporal_features) if temporal_features else pd.DataFrame()
    
    def _max_consecutive(self, binary_series):
        """Calculate maximum consecutive wins"""
        if len(binary_series) == 0:
            return 0
        max_streak = 0
        current_streak = 0
        for win in binary_series:
            if win:
                current_streak += 1
                max_streak = max(max_streak, current_streak)
            else:
                current_streak = 0
        return max_streak
    
    def prepare_features(self):
        """Prepare features with team-specific modeling"""
        print("Preparing enhanced features for ML...")
        
        if len(self.match_data) == 0:
            print("No match data available!")
            return pd.DataFrame(), pd.Series()
        
        # Create team matchup features
        self._create_team_matchup_features()
        
        # Exclude problematic features (data leakage)
        exclude_cols = [
            'demo_file', 'map_name', 'team', 'match_winner',
            'rounds_won', 'rounds_lost'  # Remove direct outcome predictors
        ]
        
        # Get numerical features
        numerical_cols = []
        for col in self.match_data.columns:
            if col not in exclude_cols:
                try:
                    if pd.api.types.is_numeric_dtype(self.match_data[col]):
                        numerical_cols.append(col)
                        print(f"Added feature: {col}")
                except Exception as e:
                    print(f"Error checking column {col}: {e}")
        
        # Add engineered features
        engineered_features = self._create_engineered_features()
        for feature_name, feature_values in engineered_features.items():
            self.match_data[feature_name] = feature_values
            numerical_cols.append(feature_name)
            print(f"Added engineered feature: {feature_name}")
        
        self.feature_columns = numerical_cols
        print(f"Using {len(numerical_cols)} total features")
        
        # Prepare feature matrix
        X = self.match_data[self.feature_columns].fillna(0)
        
        # Apply robust scaling
        scaler = RobustScaler()
        X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)
        self.feature_scaler = scaler
        
        # Prepare target
        if 'match_winner' in self.match_data.columns:
            y = self.match_data['match_winner']
        else:
            # Use map-based probabilities (no arbitrary bias)
            np.random.seed(42)
            winners = []
            for _, row in self.match_data.iterrows():
                map_name = row.get('map_name', 'de_mirage')
                ct_prob = self.map_ct_bias.get(map_name, 0.5)
                winners.append('CT' if np.random.random() < ct_prob else 'T')
            y = pd.Series(winners)
        
        print(f"Class distribution: {y.value_counts().to_dict()}")
        
        return X_scaled, y
    
    def _create_team_matchup_features(self):
        """Create features for team vs team matchups"""
        print("Creating team matchup features...")
        team_encoded = self.team_encoder.fit_transform(self.match_data['team'].fillna('Unknown'))
        self.match_data['team_encoded'] = team_encoded
        
    def _create_engineered_features(self):
        """Create meaningful engineered features"""
        engineered_features = {}
        
        # KD Ratio
        if 'kills' in self.match_data.columns and 'deaths' in self.match_data.columns:
            deaths_safe = self.match_data['deaths'].replace(0, 1)
            engineered_features['kd_ratio'] = self.match_data['kills'] / deaths_safe
        
        # Performance efficiency
        if 'adr' in self.match_data.columns and 'economy_eff' in self.match_data.columns:
            economy_safe = self.match_data['economy_eff'].replace(0, 1)
            engineered_features['performance_efficiency'] = self.match_data['adr'] / economy_safe
        
        # Clutch performance
        if 'clutch_win_rate' in self.match_data.columns and 'clutch_loss_rate' in self.match_data.columns:
            engineered_features['clutch_differential'] = (
                self.match_data['clutch_win_rate'] - self.match_data['clutch_loss_rate']
            )
        
        # Team skill impact
        if 'team_encoded' in self.match_data.columns:
            team_max = self.match_data['team_encoded'].max()
            if team_max > 0:
                engineered_features['team_strength'] = self.match_data['team_encoded'] / team_max
        
        return engineered_features
    
    def train_model(self, X, y):
        """Train model with strong regularization"""
        print("Training model with improved regularization...")
        
        if len(X) == 0:
            return None, None, None, None, None, None
        
        # Split data
        try:
            X_temp, X_val, y_temp, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
            X_train, X_test, y_train, y_test = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp)
        except ValueError:
            X_temp, X_val, y_temp, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
            X_train, X_test, y_train, y_test = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42)
        
        print(f"Training: {len(X_train)}, Test: {len(X_test)}, Validation: {len(X_val)}")
        
        # Models with strong regularization
        models = {
            'RandomForest': RandomForestClassifier(
                n_estimators=50, random_state=42, max_depth=3, min_samples_split=15,
                min_samples_leaf=8, max_features=0.4, class_weight='balanced'
            ),
            'LogisticRegression': LogisticRegression(
                random_state=42, C=0.01, class_weight='balanced', max_iter=1000
            )
        }
        
        best_score = 0
        best_model = None
        
        for name, model in models.items():
            print(f"Testing {name}...")
            
            try:
                cv_folds = min(3, len(np.unique(y_train)))
                cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
                cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy')
                
                print(f"{name} CV Score: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
                
                model.fit(X_train, y_train)
                test_score = model.score(X_test, y_test)
                print(f"{name} Test Score: {test_score:.3f}")
                
                if cv_scores.mean() > best_score:
                    best_score = cv_scores.mean()
                    best_model = model
                    self.best_model_name = name
                    
            except Exception as e:
                print(f"Error training {name}: {e}")
                continue
        
        if best_model is None:
            print("No model trained successfully!")
            return None, None, None, None, None, None
            
        self.model = best_model
        
        # Validation results
        val_score = self.model.score(X_val, y_val)
        val_predictions = self.model.predict(X_val)
        
        print(f"\nBest Model: {self.best_model_name}")
        print(f"Validation Score: {val_score:.3f}")
        print("\nValidation Classification Report:")
        print(classification_report(y_val, val_predictions))
        
        # Feature importance
        if hasattr(self.model, 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'feature': self.feature_columns,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            print("\nTop 10 Most Important Features:")
            print(feature_importance.head(10))
        
        return X_train, X_test, X_val, y_train, y_test, y_val
    
    def predict_match(self, team1, team2, map_name, team1_side='CT'):
        """Predict with ONLY your empirical data + small ELO adjustment"""
        if self.model is None:
            raise ValueError("Model not trained yet. Call run_full_pipeline() first.")
        
        print(f"\nPredicting match: {team1} vs {team2} on {map_name}")
        
        # Determine team sides
        team_ct = team1 if team1_side == 'CT' else team2
        team_t = team2 if team1_side == 'CT' else team1
        
        # Get team ELO ratings
        ct_rating = self.team_elo_ratings.get(team_ct, 1250)
        t_rating = self.team_elo_ratings.get(team_t, 1250)
        rating_diff = ct_rating - t_rating
        
        # ONLY use your empirical map bias (no additional bias)
        base_ct_prob = self.map_ct_bias.get(map_name, 0.5)
        
        # Small ELO adjustment: 100 rating points = ~1.5% probability shift
        rating_adjustment = np.tanh(rating_diff / 400) * 0.06  # Max 6% adjustment
        
        final_ct_prob_raw = base_ct_prob + rating_adjustment
        
        # Apply confidence bounds (30-70%)
        MIN_PROB, MAX_PROB = 0.30, 0.70
        final_ct_prob = np.clip(final_ct_prob_raw, MIN_PROB, MAX_PROB)
        final_t_prob = 1.0 - final_ct_prob
        
        # Determine winner
        predicted_winner = team_ct if final_ct_prob > final_t_prob else team_t
        winner_probability = max(final_ct_prob, final_t_prob)
        
        # Confidence level
        prob_margin = abs(final_ct_prob - final_t_prob)
        if prob_margin > 0.25:
            confidence = "High"
        elif prob_margin > 0.15:
            confidence = "Medium" 
        else:
            confidence = "Low"
        
        print(f"Prediction: {predicted_winner}")
        print(f"Map CT Bias (Your Data): {base_ct_prob:.1%}")
        print(f"ELO Adjustment: {rating_adjustment:+.1%}")
        print(f"Final: CT {final_ct_prob:.1%} vs T {final_t_prob:.1%}")
        
        return predicted_winner, {
            'CT': final_ct_prob, 'T': final_t_prob, 'winner_probability': winner_probability,
            'team_ct': team_ct, 'team_t': team_t, 'confidence': confidence,
            'rating_difference': rating_diff
        }
    
    def run_full_pipeline(self):
        """Run the complete corrected ML pipeline"""
        print("=== CORRECTED CS2 Match Prediction Model ===")
        print("Corrections applied:")
        print("- Using ONLY your empirical map CT bias data")
        print("- Removed all artificial CT bias sources")
        print("- Small ELO weight for team strength")
        print(f"Target Teams: {', '.join(self.target_teams)}")
        
        self.load_data()
        X, y = self.prepare_features()
        
        if len(X) == 0:
            print("No data available for training!")
            return
        
        self.train_model(X, y)
        
        print("\n=== Training Complete ===")


def corrected_interactive_prediction_system():
    """Corrected interactive prediction system"""
    predictor = CorrectedCS2MatchPredictor()
    predictor.run_full_pipeline()
    
    print("\n" + "="*60)
    print("🎯 CORRECTED CS2 MATCH PREDICTION SYSTEM 🎯")
    print("="*60)
    print("✅ CS2 first to 13 rounds")
    print("✅ Uses your empirical map CT bias data")
    print("✅ No artificial CT bias")
    print("✅ Confidence bounds (30-70%)")
    print("="*60)
    
    while True:
        try:
            print(f"\nAvailable teams:")
            for i, team in enumerate(predictor.target_teams, 1):
                team_rating = predictor.team_elo_ratings.get(team, 1250)
                print(f"  {i}. {team} (ELO: {team_rating})")
            
            print("\nEnter 'quit' to exit")
            
            # Team selection logic (same as before)
            team1_input = input("\nEnter Team 1 (name or number): ").strip()
            if team1_input.lower() == 'quit':
                break
                
            team1 = None
            if team1_input.isdigit():
                team1_idx = int(team1_input) - 1
                if 0 <= team1_idx < len(predictor.target_teams):
                    team1 = predictor.target_teams[team1_idx]
                else:
                    print("Invalid team number!")
                    continue
            else:
                matches = [team for team in predictor.target_teams if team1_input.lower() in team.lower()]
                if len(matches) == 1:
                    team1 = matches[0]
                elif len(matches) > 1:
                    print(f"Multiple matches: {matches}")
                    continue
                else:
                    print(f"Team '{team1_input}' not found!")
                    continue
            
            team2_input = input("Enter Team 2 (name or number): ").strip()
            if team2_input.lower() == 'quit':
                break
                
            team2 = None
            if team2_input.isdigit():
                team2_idx = int(team2_input) - 1
                if 0 <= team2_idx < len(predictor.target_teams):
                    team2 = predictor.target_teams[team2_idx]
                else:
                    print("Invalid team number!")
                    continue
            else:
                matches = [team for team in predictor.target_teams if team2_input.lower() in team.lower()]
                if len(matches) == 1:
                    team2 = matches[0]
                elif len(matches) > 1:
                    print(f"Multiple matches: {matches}")
                    continue
                else:
                    print(f"Team '{team2_input}' not found!")
                    continue
            
            if team1 == team2:
                print("Teams cannot be the same!")
                continue
            
            # Side selection
            side_input = input(f"Which team starts CT? (1 for {team1}, 2 for {team2}, Enter for {team1}): ").strip()
            team1_side = 'T' if side_input == '2' else 'CT'
            
            # Map selection with your empirical data
            maps_with_bias = [
                ("de_mirage", "53.6% CT"), ("de_dust2", "50.7% CT"), ("de_inferno", "50.7% CT"),
                ("de_ancient", "52.7% CT"), ("de_nuke", "57.9% CT"), ("de_overpass", "59.9% CT"),
                ("de_train", "59.1% CT")
            ]
            
            print("\nMaps (with YOUR empirical CT win rates):")
            for i, (map_name, ct_bias) in enumerate(maps_with_bias, 1):
                print(f"  {i}. {map_name} ({ct_bias})")
            
            map_input = input("Enter map (name or number): ").strip()
            
            if map_input.isdigit():
                map_idx = int(map_input) - 1
                if 0 <= map_idx < len(maps_with_bias):
                    map_name = maps_with_bias[map_idx][0]
                else:
                    map_name = "de_mirage"
            else:
                map_name = map_input if map_input else "de_mirage"
            
            # Make prediction
            print("\n" + "-"*60)
            predicted_winner, probabilities = predictor.predict_match(team1, team2, map_name, team1_side)
            
            # Enhanced output
            print(f"\n🎯 CORRECTED PREDICTION 🎯")
            print(f"Match: {team1} vs {team2}")
            print(f"Map: {map_name}")
            print(f"Predicted Winner: {predicted_winner}")
            print(f"{probabilities['team_ct']} (CT): {probabilities['CT']:.1%}")
            print(f"{probabilities['team_t']} (T): {probabilities['T']:.1%}")
            print(f"Confidence: {probabilities['confidence']}")
            
            if 'rating_difference' in probabilities:
                rating_diff = probabilities['rating_difference']
                if abs(rating_diff) > 50:
                    stronger_team = probabilities['team_ct'] if rating_diff > 0 else probabilities['team_t']
                    print(f"ELO Advantage: {stronger_team} (+{abs(rating_diff)} ELO)")
            
            print("-"*60)
            
            another = input("\nMake another prediction? (y/n): ").strip().lower()
            if another not in ['y', 'yes', '']:
                break
                
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")
            continue
    
    print("\nThanks for using the Corrected CS2 Prediction System! 🎯")

# Main execution
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == '--interactive':
        corrected_interactive_prediction_system()
    else:
        predictor = CorrectedCS2MatchPredictor()
        predictor.run_full_pipeline()