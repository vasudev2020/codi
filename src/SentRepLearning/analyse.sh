
# sbatch -J analyse -p compute --mem 50000 -t 2-0 --gres=gpu --cpus-per-task=8 --output="codi-glove-norm" --wrap="python3 -u Analyse.py --printnorm --model codi"
# sbatch -J analyse -p compute --mem 80000 -t 2-0 --gres=gpu:a100:1 --cpus-per-task=8 --output="codi-roberta-norm" --wrap="python3 -u Analyse.py --printnorm --model codi_init_modelroberta"
sbatch -J analyse -p compute --mem 80000 -t 2-0 --gres=gpu:a100:1 --cpus-per-task=8 --output="codi-llama-norm" --wrap="python3 -u Analyse.py --printnorm --model codi_init_modelllama"

# sbatch -J analyse -p compute --mem 50000 -t 2-0 --gres=gpu --cpus-per-task=8 --output="codi-glove-comp" --wrap="python3 -u Analyse.py --companalyse --model codi"
# sbatch -J analyse -p compute --mem 80000 -t 2-0 --gres=gpu:a100:1 --cpus-per-task=8 --output="codi-roberta-comp" --wrap="python3 -u Analyse.py --companalyse --model codi_init_modelroberta"
sbatch -J analyse -p compute --mem 80000 -t 2-0 --gres=gpu:a100:1 --cpus-per-task=8 --output="codi-llama-comp" --wrap="python3 -u Analyse.py --companalyse --model codi_init_modelllama"

# sbatch -J analyse -p compute --mem 50000 -t 2-0 --gres=gpu --cpus-per-task=8 --output="codi-glove-scores" --wrap="python3 -u Analyse.py --printscore --model codi"
# sbatch -J analyse -p compute --mem 80000 -t 2-0 --gres=gpu:a100:1 --cpus-per-task=8 --output="codi-roberta-scores" --wrap="python3 -u Analyse.py --printscore --model codi_init_modelroberta"
sbatch -J analyse -p compute --mem 80000 -t 2-0 --gres=gpu:a100:1 --cpus-per-task=8 --output="codi-llama-scores" --wrap="python3 -u Analyse.py --printscore --model codi_init_modelllama"

# sbatch -J analyse -p compute --mem 50000 -t 2-0 --gres=gpu --cpus-per-task=8 --wrap="python3 -u Analyse.py --catfreq"

