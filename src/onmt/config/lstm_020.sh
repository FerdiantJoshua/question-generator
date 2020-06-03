#================================================== PREPROCESS ==================================================
onmt_preprocess -train_src '/raid/ayu/ferdiant/data/processed/train/squad_id_cased_source.txt' -train_tgt '/raid/ayu/ferdiant/data/processed/train/squad_id_cased_target.txt' \
    -valid_src '/raid/ayu/ferdiant/data/processed/val/squad_id_cased_source.txt' -valid_tgt '/raid/ayu/ferdiant/data/processed/val/squad_id_cased_target.txt' \
    -save_data '/raid/ayu/ferdiant/data/processed/onmt/squad_id_cased' \
    -overwrite \
    -src_vocab_size 60000 \
    -tgt_vocab_size 60000 \
    -src_seq_length 60 \
    -tgt_seq_length 20

#================================================== EMBEDDING ==================================================
python src/onmt/embeddings_to_torch.py -emb_file_both 'models/word-embedding/ft_to_gl_300_id.vec' \
    -dict_file '/raid/ayu/ferdiant/data/processed/onmt/squad_id_cased.vocab.pt' \
    -output_file '/raid/ayu/ferdiant/data/processed/embeddings_cased'

#================================================== TRAIN ==================================================
onmt_train -data '/raid/ayu/ferdiant/data/processed/onmt/squad_id_cased' -save_model 'models/checkpoints/onmt/lstm_020' \
    -world_size 1 -gpu_ranks 0 \
    -save_checkpoint_steps 7195 \
    -word_vec_size 300 \
    -train_from 'models/checkpoints/onmt/lstm_020_step_28780.pt' \
    -pre_word_vecs_enc '/raid/ayu/ferdiant/data/processed/embeddings_cased.enc.pt' \
    -pre_word_vecs_dec '/raid/ayu/ferdiant/data/processed/embeddings_cased.dec.pt' \
    -fix_word_vecs_enc \
    -fix_word_vecs_dec \
    -keep_checkpoint 5 \
    -learning_rate 1 \
    -learning_rate_decay 0.5 \
    -start_decay_steps 14390 \
    -rnn_type LSTM \
    -encoder_type brnn \
    -layers 2 \
    -global_attention general \
    -rnn_size 600 \
    -train_steps 34536 \
    -valid_steps 2878 \
    -batch_size 64 \
    -dropout 0.3

#================================================== TRANSLATE ==================================================
onmt_translate -model 'models/checkpoints/onmt/lstm_020_step_34536.pt' \
    -src '/raid/ayu/ferdiant/data/processed/test/squad_id_cased_source.txt' -output 'reports/txts/onmt/lstm_020_step_34536_pred.txt' -replace_unk \
    -beam_size 5 \
    -max_length 20

#================================================== EVALUATE ==================================================
python src/onmt/run_evaluation.py \
    --source_file='/raid/ayu/ferdiant/data/processed/test/squad_id_cased_source.txt' \
    --target_file='/raid/ayu/ferdiant/data/processed/test/squad_id_cased_target.txt' \
    --prediction_file='reports/txts/onmt/lstm_020_step_34536_pred.txt' \
    --log_file='reports/txts/onmt/eval_log_lstm_020_step_34536.txt'