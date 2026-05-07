LOG="figure2-4"
script="figure2-4.py"
mkdir -p logs
nohup python -u $script > "logs/log_${LOG}.txt" 2>&1 &