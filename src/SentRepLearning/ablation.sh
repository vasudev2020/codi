# for modelname in 'codi_aux_mask011111' 'codi_aux_mask101111' 'codi_aux_mask110111' 'codi_aux_mask111011' 'codi_aux_mask111101' 'codi_aux_mask111110' 'codi_aux_mask000111'; do
for modelname in 'codi_aux_mask011111_init_modelllama' 'codi_aux_mask101111_init_modelllama' 'codi_aux_mask110111_init_modelllama' 'codi_aux_mask111011_init_modelllama' 'codi_aux_mask111101_init_modelllama' 'codi_aux_mask111110_init_modelllama' 'codi_aux_mask000111_init_modelllama'; do
    python3 -u GenSentReps.py --dataset $1 --modelname $modelname
done
