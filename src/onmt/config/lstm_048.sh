#================================================== PREPROCESS ==================================================
onmt_preprocess -train_src 'data/processed/train/squad_id_split0.9_uncased_source.txt' -train_tgt 'data/processed/train/squad_id_split0.9_uncased_target.txt' \
    -valid_src 'data/processed/val/squad_id_split0.9_uncased_source.txt' -valid_tgt 'data/processed/val/squad_id_split0.9_uncased_target.txt' \
    -save_data 'data/processed/onmt/squad_id_split0.9_uncased_2' \
    -overwrite \
    -dynamic_dict \
    -src_vocab_size 45000 \
    -tgt_vocab_size 28000 \
    -src_seq_length 60 \
    -tgt_seq_length 20

#================================================== EMBEDDING ==================================================
python src/onmt/embeddings_to_torch.py -emb_file_both 'models/word-embedding/ft_to_gl_300_id.vec' \
    -dict_file 'data/processed/onmt/squad_id_split0.9_uncased_2.vocab.pt' \
    -output_file 'data/processed/onmt/embeddings_uncased_2'

#================================================== TRAIN ==================================================
onmt_train -data 'data/processed/onmt/squad_id_split0.9_uncased_2' -save_model 'models/checkpoints/onmt/lstm_048' \
    -world_size 1 -gpu_ranks 0 \
    -seed 42 \
    -save_checkpoint_steps 8025 \
    -word_vec_size 300 \
    -pre_word_vecs_enc 'data/processed/onmt/embeddings_uncased_2.enc.pt' \
    -pre_word_vecs_dec 'data/processed/onmt/embeddings_uncased_2.dec.pt' \
    -fix_word_vecs_enc \
    -fix_word_vecs_dec \
    -keep_checkpoint 3 \
    -optim 'adam' \
    -learning_rate 0.001 \
    -learning_rate_decay 0.5 \
    -start_decay_steps 8025 \
    -rnn_type LSTM \
    -encoder_type brnn \
    -layers 2 \
    -global_attention general \
    -rnn_size 600 \
    -train_steps 16050 \
    -valid_steps 3210 \
    -batch_size 64 \
    -dropout 0.3 \
    -copy_attn \
    -copy_attn_force \
    -coverage_attn

#================================================== TRANSLATE ==================================================
onmt_translate -model 'models/checkpoints/onmt/lstm_048_step_16050.pt' \
    -src 'data/processed/test/squad_id_split0.9_uncased_source.txt' -output 'reports/txts/onmt/lstm_048_step_16050_pred.txt' -replace_unk \
    -seed 42 \
    -beam_size 5 \
    -max_length 20

#================================================== EVALUATE ==================================================
python src/onmt/run_evaluation.py \
    --source_file='data/processed/test/squad_id_split0.9_uncased_source.txt' \
    --target_file='data/processed/test/squad_id_split0.9_uncased_target.txt' \
    --prediction_file='reports/txts/onmt/lstm_048_step_16050_pred.txt' \
    --log_file='reports/txts/onmt/eval_log_lstm_048_step_16050.txt'