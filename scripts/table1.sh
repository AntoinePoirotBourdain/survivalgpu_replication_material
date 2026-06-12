LOG="table1"
script="table1.py"
mkdir -p logs
nohup python -u $script > "logs/log_${LOG}.txt" 2>&1 &
