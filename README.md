## CS2 ML Project 
This is a prototype project using CS2 demo files using library demoparser2. 
It is recommended that the demo files exceed 500+ for accurate results.

## Files Involved

- check_file.py: Script to sort the files within demo folder chronologically (avoid overlap)
- PlayerStatistics.py: Parses the demo files using ticks, calculates all relevant statistics from the match. Also calculates swing rating, economy efficiency, ADR differential, etc.
- RoundAnalysis.py: Round-by-round analysis on outcome, equipment value, status of round end, etc. Helpful in determining strongest/weakest teams.
- MLModel.py: Interactive ML Model using LR, RF, K-fold CV, GBC. Outputs results of match based on teams involved, map played, who starts CT side. 
- Cs_gnn_python.py: Graph Neural Network script (irrelevant for the MLModel.py performance - mainly for exploration & capabilities of GNN's in this space).
- TeamOveralls.py & PlayerStatsMainVisuals.py both output visualizations. GNN also produces a graph displaying MSE, MAE, Training Loss & Prediction Correlation 

# Note: 
- import/install relevant packages in each file
- change demo file_path in .py files relative to your folder
- ensure .csv output is located in script folder

# Steps:
- Run check_file.py 
- Run RoundAnalysis.py
- Run PlayerStatistics.py
- Run cs_gnn_python.py 
- Use python MLModel.py --interactive in terminal
- 
