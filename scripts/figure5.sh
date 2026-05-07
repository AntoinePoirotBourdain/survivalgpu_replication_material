LOG="figure5"
script="figure5.py"
mkdir -p logs
nohup python -u $script > "logs/log_${LOG}.txt" 2>&1 &