LOG="table1-4"
script="figure_generation/table1-4.py"
mkdir -p logs
nohup python -u $script > "logs/log_${LOG}.txt" 2>&1 &
