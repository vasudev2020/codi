# # Evaluate models
# sbatch -J codi-eval -p compute --mem 50000 -t 2-0 --gres=gpu:a100:1 --cpus-per-task=8 --wrap="python3 -u Evaluate.py --classifier sklearn --benchmark probing"

# # More test on WC task
# sbatch -J codi-eval -p compute --mem 50000 -t 2-0 --gres=gpu:a100:1 --cpus-per-task=8 --wrap="python3 -u WCTest.py"

# # Evaluate Baseline models 
# sbatch -J baseline -p compute --mem 80000 -t 2-0 --gres=gpu:a100:1 --cpus-per-task=8 --wrap="python -u Baseline.py"
# # sbatch -J baseline -p compute --mem 80000 -t 2-0 --gres=gpu:a100:1 --cpus-per-task=8 --wrap="python3 -u Baseline.py"