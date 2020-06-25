#================================================== PREPROCESS ==================================================
onmt_preprocess -train_src 'data/processed/train/squad_id_split0.9_cased_source.txt' -train_tgt 'data/processed/train/squad_id_split0.9_cased_target.txt' \
    -valid_src 'data/processed/val/squad_id_split0.9_cased_source.txt' -valid_tgt 'data/processed/val/squad_id_split0.9_cased_target.txt' \
    -save_data 'data/processed/onmt/squad_id_split0.9_cased_copy' \
    -overwrite \
    -dynamic_dict \
    -src_vocab_size 50000 \
    -tgt_vocab_size 30000 \
    -src_seq_length 60 \
    -tgt_seq_length 20

#================================================== EMBEDDING ==================================================
python src/onmt/embeddings_to_torch.py -emb_file_both 'models/word-embedding/ft_to_gl_300_id.vec' \
    -dict_file 'data/processed/onmt/squad_id_split0.9_cased_copy.vocab.pt' \
    -output_file 'data/processed/onmt/embeddings_cased_copy'

#================================================== TRAIN ==================================================
onmt_train -data 'data/processed/onmt/squad_id_split0.9_cased_copy' -save_model 'models/checkpoints/onmt/transformer_012' \
    -world_size 1 -gpu_ranks 0 \
    -seed 42 \
    -pre_word_vecs_enc 'data/processed/onmt/embeddings_cased_copy.enc.pt' \
    -pre_word_vecs_dec 'data/processed/onmt/embeddings_cased_copy.dec.pt' \
    -fix_word_vecs_enc \
    -fix_word_vecs_dec \
    -feat_merge mlp \
    -keep_checkpoint 5 \
    -layers 2 -rnn_size 300 -word_vec_size 300 -transformer_ff 256 -heads 2 \
    -encoder_type transformer -decoder_type transformer -position_encoding \
    -train_steps 120600  -max_generator_batches 2 -dropout 0.1 \
    -batch_size 256 -batch_type tokens -normalization tokens -accum_count 2 \
    -optim adam -adam_beta2 0.998 -decay_method noam -warmup_steps 4020 -learning_rate 1 \
    -max_grad_norm 1 -param_init 0  -param_init_glorot \
    -label_smoothing 0.1 -valid_steps 4020 -save_checkpoint_steps 4020 \
    -copy_attn \
    -copy_attn_force

#================================================== TRANSLATE ==================================================
#-------------------------------------------------- SQUAD --------------------------------------------------
onmt_translate -model 'models/checkpoints/onmt/transformer_012_step_120600.pt' \
    -src 'data/processed/test/squad_id_split0.9_cased_source.txt' -output 'reports/txts/onmt/transformer_012_step_120600_pred.txt' -replace_unk \
    -seed 42 \
    -beam_size 5 \
    -max_length 20
#-------------------------------------------------- TYDIQA --------------------------------------------------
onmt_translate -model 'models/checkpoints/onmt/transformer_012_step_120600.pt' \
    -src 'data/processed/test/tydiqa_id_split0.9_cased_source.txt' -output 'reports/txts/onmt/transformer_012_step_120600_pred_tydiqa.txt' -replace_unk \
    -seed 42 \
    -beam_size 5 \
    -max_length 20

#================================================== EVALUATE ==================================================
#-------------------------------------------------- SQUAD --------------------------------------------------
python src/onmt/run_evaluation.py \
    --source_file='data/processed/test/squad_id_split0.9_cased_source.txt' \
    --target_file='data/processed/test/squad_id_split0.9_cased_target.txt' \
    --prediction_file='reports/txts/onmt/transformer_012_step_120600_pred.txt' \
    --log_file='reports/txts/onmt/eval_log_transformer_012_step_120600.txt'

#-------------------------------------------------- TYDIQA --------------------------------------------------
python src/onmt/run_evaluation.py \
    --source_file='data/processed/test/tydiqa_id_split0.9_cased_source.txt' \
    --target_file='data/processed/test/tydiqa_id_split0.9_cased_target.txt' \
    --prediction_file='reports/txts/onmt/transformer_012_step_120600_pred_tydiqa.txt' \
    --log_file='reports/txts/onmt/eval_log_transformer_012_step_120600_tydiqa.txt'
