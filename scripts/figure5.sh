LOG="figure5"
script="figure_generation/figure5.py"
mkdir -p logs
nohup python -u $script > "logs/log_${LOG}.txt" 2>&1 &