LOG="figure6"
script="figure6.py"
mkdir -p logs
nohup python -u $script > "logs/log_${LOG}.txt" 2>&1 &