echo "This shell example is not intended to be executed. Please read and use all the provided commands in this file manually."
exit 1

# If you do not wish to use copy and coverage mechanism, remove these lines:
# PREPROCESS:
    -dynamic_dict
# TRAIN:
    -copy_attn \
    -copy_attn_force \
    -coverage_attn
#================================================== PREPROCESS ==================================================
onmt_preprocess -train_src 'data/processed/train/squad_id_cased_source.txt' -train_tgt 'data/processed/train/squad_id_cased_target.txt' \
    -valid_src 'data/processed/val/squad_id_cased_source.txt' -valid_tgt 'data/processed/val/squad_id_cased_target.txt' \
    -dynamic_dict \
    -save_data 'data/processed/onmt/squad_id_cased' \
    -overwrite \
    -src_vocab_size 50000 \
    -tgt_vocab_size 30000 \
    -src_seq_length 60 \
    -tgt_seq_length 20

#================================================== EMBEDDING ==================================================
python src/onmt/embeddings_to_torch.py -emb_file_both 'models/word-embedding/ft_to_gl_300_id.vec' \
    -dict_file 'data/processed/onmt/squad_id_cased.vocab.pt' \
    -output_file 'data/processed/embeddings_cased'

#================================================== TRAIN ==================================================
#-------------------------------------------------- RNN --------------------------------------------------
onmt_train -data 'data/processed/onmt/squad_id_cased' -save_model 'models/checkpoints/onmt/lstm_013' \
    -world_size 1 -gpu_ranks 0 \
    -save_checkpoint_steps 7195 \
    -word_vec_size 300 \
    -pre_word_vecs_enc 'data/processed/embeddings_cased.enc.pt' \
    -pre_word_vecs_dec 'data/processed/embeddings_cased.dec.pt' \
    -fix_word_vecs_enc \
    -fix_word_vecs_dec \
    -keep_checkpoint 5 \
    -learning_rate 1 \
    -learning_rate_decay 0.5 \
    -start_decay_steps 14390 \
    -encoder_type brnn \
    -layers 2 \
    -global_attention general \
    -rnn_size 600 \
    -train_steps 35975 \
    -valid_steps 2878 \
    -batch_size 64 \
    -dropout 0.3 \
    -copy_attn \
    -copy_attn_force \
    -coverage_attn
#-------------------------------------------------- TRANSFORMER --------------------------------------------------
onmt_train -data 'data/processed/onmt/squad_id_cased' -save_model 'models/checkpoints/onmt/transformer_004' \
    -world_size 1 -gpu_ranks 0 \
    -pre_word_vecs_enc 'data/processed/onmt/embeddings_cased.enc.pt' \
    -pre_word_vecs_dec 'data/processed/onmt/embeddings_cased.dec.pt' \
    -fix_word_vecs_enc \
    -fix_word_vecs_dec \
    -feat_merge mlp \
    -keep_checkpoint 20 \
    -layers 2 -rnn_size 300 -word_vec_size 300 -transformer_ff 256 -heads 2 \
    -encoder_type transformer -decoder_type transformer -position_encoding \
    -train_steps 200000  -max_generator_batches 2 -dropout 0.1 \
    -batch_size 256 -batch_type tokens -normalization tokens  -accum_count 2 \
    -optim adam -adam_beta2 0.998 -decay_method noam -warmup_steps 8000 -learning_rate 2 \
    -max_grad_norm 0 -param_init 0  -param_init_glorot \
    -label_smoothing 0.1 -valid_steps 1200 -save_checkpoint_steps 3000 \
    -copy_attn \
    -copy_attn_force \
    -coverage_attn

#================================================== TRANSLATE ==================================================
onmt_translate -model 'models/checkpoints/onmt/lstm_013_step_35975.pt' \
    -src 'data/processed/test/squad_id_cased_source.txt' -output 'reports/txts/onmt/pred_lstm_013_step_35975.txt' -replace_unk \
    -beam_size 5 \
    -max_length 22 \
    -verbose

#================================================== EVALUATE ==================================================
python src/onmt/run_evaluation.py \
    --prediction_file='reports/txts/onmt/pred_lstm_013_step_35975.txt' \
    --log_file='reports/txts/onmt/eval_log_lstm_013_step_35975.txt'

#================================================== FREE INFERENCE ==================================================
python src/onmt/run_free_generation.py \
    --preprocess_output_path='reports/txts/onmt/free_input.txt' \
    --model_path='models/checkpoints/onmt/lstm_013_step_35975.pt' \
    --manual_ne_postag
