gpu=gpu:a100:1 #'gpu:rtx2080ti:1'
mem=80000
size=100000

# # Parse the data before training
# sbatch -J codi -p long --mem $mem -t 10-0 --gres=$gpu --cpus-per-task=8 --mail-type=ALL --wrap="python3 -u Preparse.py --dataset wiki --size $size"

# # Train models
# for init_model in 'glove' 'roberta' 'llama'; do
for init_model in 'glove' 'roberta'; do
    sbatch -J codi -p compute --mem $mem -t 2-0 --gres=$gpu --cpus-per-task=8 --output=$init_model'-'$size'.out' --wrap="python3 -u IterTrain.py --size $size --init_model $init_model"
done

# # Train for ablation study
# for init_model in 'glove' 'roberta'; do
# for init_model in 'llama'; do
#     sbatch -J codi -p compute --mem $mem -t 2-0 --gres=$gpu --cpus-per-task=4 --output=$init_model'-'$size'-111110.out' --wrap="python3 -u IterTrain.py --size $size --init_model $init_model --aux_mask 111110"
#     sbatch -J codi -p compute --mem $mem -t 2-0 --gres=$gpu --cpus-per-task=4 --output=$init_model'-'$size'-111101.out' --wrap="python3 -u IterTrain.py --size $size --init_model $init_model --aux_mask 111101"
#     sbatch -J codi -p compute --mem $mem -t 2-0 --gres=$gpu --cpus-per-task=4 --output=$init_model'-'$size'-111011.out' --wrap="python3 -u IterTrain.py --size $size --init_model $init_model --aux_mask 111011"
#     sbatch -J codi -p compute --mem $mem -t 2-0 --gres=$gpu --cpus-per-task=4 --output=$init_model'-'$size'-110111.out' --wrap="python3 -u IterTrain.py --size $size --init_model $init_model --aux_mask 110111"
#     sbatch -J codi -p compute --mem $mem -t 2-0 --gres=$gpu --cpus-per-task=4 --output=$init_model'-'$size'-101111.out' --wrap="python3 -u IterTrain.py --size $size --init_model $init_model --aux_mask 101111"
#     sbatch -J codi -p compute --mem $mem -t 2-0 --gres=$gpu --cpus-per-task=4 --output=$init_model'-'$size'-011111.out' --wrap="python3 -u IterTrain.py --size $size --init_model $init_model --aux_mask 011111"
#     sbatch -J codi -p compute --mem $mem -t 2-0 --gres=$gpu --cpus-per-task=4 --output=$init_model'-'$size'-000111.out' --wrap="python3 -u IterTrain.py --size $size --init_model $init_model --aux_mask 000111"
#     sbatch -J codi -p compute --mem $mem -t 2-0 --gres=$gpu --cpus-per-task=8 --output=$init_model'-'$size'-syn.out' --wrap="python3 -u IterTrain.py --size $size --init_model $init_model --aux_mask 000111"
# done