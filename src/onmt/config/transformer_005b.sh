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
onmt_train -data '/raid/ayu/ferdiant/data/processed/onmt/squad_id_cased' -save_model 'models/checkpoints/onmt/transformer_005' \
    -world_size 1 -gpu_ranks 0 \
    -seed 42 \
    -pre_word_vecs_enc '/raid/ayu/ferdiant/data/processed/embeddings_cased.enc.pt' \
    -pre_word_vecs_dec '/raid/ayu/ferdiant/data/processed/embeddings_cased.dec.pt' \
    -fix_word_vecs_enc \
    -fix_word_vecs_dec \
    -feat_merge mlp \
    -keep_checkpoint 20 \
    -layers 2 -rnn_size 300 -word_vec_size 300 -transformer_ff 300 -heads 2 \
    -encoder_type transformer -decoder_type transformer -position_encoding \
    -train_steps 101920  -max_generator_batches 2 -dropout 0.3 \
    -batch_size 64 -accum_count 2 \
    -optim adam -adam_beta2 0.998 -decay_method noam -warmup_steps 7195 -learning_rate 1 \
    -max_grad_norm 1 \
    -valid_steps 7195 -save_checkpoint_steps 7195

#================================================== TRANSLATE ==================================================
onmt_translate -model 'models/checkpoints/onmt/transformer_005_step_86340.pt' \
    -src '/raid/ayu/ferdiant/data/processed/test/squad_id_cased_source.txt' -output 'reports/txts/onmt/transformer_005_step_86340_pred.txt' -replace_unk \
    -beam_size 5 \
    -max_length 20

#================================================== EVALUATE ==================================================
python src/onmt/run_evaluation.py \
    --source_file='/raid/ayu/ferdiant/data/processed/test/squad_id_cased_source.txt' \
    --target_file='/raid/ayu/ferdiant/data/processed/test/squad_id_cased_target.txt' \
    --prediction_file='reports/txts/onmt/transformer_005_step_86340_pred.txt' \
    --log_file='reports/txts/onmt/eval_log_transformer_005_step_86340.txt'