LOG="table1"
script="figure_generation/table1.py"
mkdir -p logs
nohup python -u $script > "logs/log_${LOG}.txt" 2>&1 &
