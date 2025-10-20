gpu=gpu:a100:1

# # Parse the probing task dataset before generating its representation
# for ds in subj_number obj_number bigram_shift odd_man_out coordination_inversion top_constituents tree_depth past_present sentence_length word_content; do
#     sbatch -J codi -p compute --mem 50000 -t 2-0 --gres=gpu --cpus-per-task=8 --wrap="python3 -u Preparse.py --dataset $ds"
# done

# # Generate representations of probing task dataset for the evaluation
# for modelname in 'glove' 'roberta' 'codi' 'codi_init_modelroberta'; do
#     for ds in subj_number obj_number coordination_inversion top_constituents tree_depth past_present sentence_length word_content; do
#         sbatch -J codi-gen -p compute --mem 50000 -t 2-0 --gres=gpu --cpus-per-task=8 --wrap="python3 -u GenSentReps.py --dataset $ds --modelname $modelname"
#     done
# done

# # Generate representations of probing task dataset for ablation study
# for ds in subj_number obj_number coordination_inversion top_constituents tree_depth past_present sentence_length word_content; do
#     sbatch -J codi-gen -p compute --mem 80000 -t 2-0 --gres=$gpu --wrap="sh ablation.sh $ds"
# done

