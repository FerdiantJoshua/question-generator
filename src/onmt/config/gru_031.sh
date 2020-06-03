#================================================== PREPROCESS ==================================================
onmt_preprocess -train_src 'data/processed/train/squad_id_split0.9_cased_source.txt' -train_tgt 'data/processed/train/squad_id_split0.9_cased_target.txt' \
    -valid_src 'data/processed/val/squad_id_split0.9_cased_source.txt' -valid_tgt 'data/processed/val/squad_id_split0.9_cased_target.txt' \
    -save_data 'data/processed/onmt/squad_id_split0.9_cased' \
    -overwrite \
    -dynamic_dict \
    -src_vocab_size 60000 \
    -tgt_vocab_size 60000 \
    -src_seq_length 60 \
    -tgt_seq_length 20

#================================================== EMBEDDING ==================================================
python src/onmt/embeddings_to_torch.py -emb_file_both 'models/word-embedding/ft_to_gl_300_id.vec' \
    -dict_file 'data/processed/onmt/squad_id_split0.9_cased.vocab.pt' \
    -output_file 'data/processed/onmt/embeddings_cased'

#================================================== TRAIN ==================================================
onmt_train -data 'data/processed/onmt/squad_id_split0.9_cased' -save_model 'models/checkpoints/onmt/gru_027' \
    -world_size 1 -gpu_ranks 0 \
    -train_from 'models/checkpoints/onmt/gru_027_step_40125.pt' \
    -seed 42 \
    -save_checkpoint_steps 8025 \
    -word_vec_size 300 \
    -pre_word_vecs_enc 'data/processed/onmt/embeddings_cased.enc.pt' \
    -pre_word_vecs_dec 'data/processed/onmt/embeddings_cased.dec.pt' \
    -fix_word_vecs_enc \
    -fix_word_vecs_dec \
    -keep_checkpoint 3 \
    -optim 'adam' \
    -learning_rate 0.001 \
    -learning_rate_decay 0.95 \
    -start_decay_steps 16050 \
    -rnn_type GRU \
    -encoder_type brnn \
    -layers 2 \
    -global_attention mlp \
    -rnn_size 256 \
    -train_steps 40125 \
    -valid_steps 3210 \
    -batch_size 64 \
    -dropout 0.3 \
    -copy_attn \
    -copy_attn_force \
    -coverage_attn

#================================================== TRANSLATE ==================================================
onmt_translate -model 'models/checkpoints/onmt/gru_027_step_40125.pt' \
    -src 'data/processed/test/squad_id_split0.9_cased_source.txt' -output 'reports/txts/onmt/gru_027_step_40125_pred.txt' -replace_unk \
    -beam_size 5 \
    -max_length 20

#================================================== EVALUATE ==================================================
python src/onmt/run_evaluation.py \
    --source_file='data/processed/test/squad_id_split0.9_cased_source.txt' \
    --target_file='data/processed/test/squad_id_split0.9_cased_target.txt' \
    --prediction_file='reports/txts/onmt/gru_027_step_40125_pred.txt' \
    --log_file='reports/txts/onmt/eval_log_gru_027_step_40125.txt'
