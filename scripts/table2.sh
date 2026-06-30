LOG="table2"
script="figure_generation/table2.py"
mkdir -p logs
nohup python -u $script > "logs/log_${LOG}.txt" 2>&1 &
