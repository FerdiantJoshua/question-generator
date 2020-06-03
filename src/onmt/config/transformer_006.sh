#================================================== PREPROCESS ==================================================
onmt_preprocess -train_src 'data/processed/train/squad_id_split0.9_cased_source.txt' -train_tgt 'data/processed/train/squad_id_split0.9_cased_target.txt' \
    -valid_src 'data/processed/val/squad_id_split0.9_cased_source.txt' -valid_tgt 'data/processed/val/squad_id_split0.9_cased_target.txt' \
    -save_data 'data/processed/onmt/squad_id_split0.9_cased' \
    -overwrite \
    -src_vocab_size 60000 \
    -tgt_vocab_size 60000 \
    -src_seq_length 60 \
    -tgt_seq_length 20

#================================================== EMBEDDING ==================================================
python src/onmt/embeddings_to_torch.py -emb_file_both 'models/word-embedding/ft_to_gl_300_id.vec' \
    -dict_file 'data/processed/onmt/squad_id_split0.9_cased.vocab.pt' \
    -output_file 'data/processed/onmt/embeddings_cased'

#================================================== TRAIN ==================================================
onmt_train -data 'data/processed/onmt/squad_id_split0.9_cased' -save_model 'models/checkpoints/onmt/transformer_006' \
    -world_size 1 -gpu_ranks 0 \
    -seed 42 \
    -train_from 'models/checkpoints/onmt/transformer_006_step_20100.pt' \
    -pre_word_vecs_enc 'data/processed/onmt/embeddings_cased.enc.pt' \
    -pre_word_vecs_dec 'data/processed/onmt/embeddings_cased.dec.pt' \
    -fix_word_vecs_enc \
    -fix_word_vecs_dec \
    -feat_merge mlp \
    -keep_checkpoint 3 \
    -layers 2 -rnn_size 300 -word_vec_size 300 -transformer_ff 512 -heads 4 \
    -encoder_type transformer -decoder_type transformer -position_encoding \
    -train_steps 80400  -max_generator_batches 2 -dropout 0.3 \
    -batch_size 512 -accum_count 2 \
    -optim adam -adam_beta2 0.998 -decay_method noam -warmup_steps 2010 -learning_rate 1 \
    -max_grad_norm 1 \
    -valid_steps 2010 -save_checkpoint_steps 2010

#================================================== TRANSLATE ==================================================
onmt_translate -model 'models/checkpoints/onmt/transformer_006_step_80400.pt' \
    -src 'data/processed/test/squad_id_split0.9_cased_source.txt' -output 'reports/txts/onmt/transformer_006_step_80400_pred.txt' -replace_unk \
    -seed 42 \
    -beam_size 5 \
    -max_length 20

#================================================== EVALUATE ==================================================
python src/onmt/run_evaluation.py \
    --source_file='data/processed/test/squad_id_split0.9_cased_source.txt' \
    --target_file='data/processed/test/squad_id_split0.9_cased_target.txt' \
    --prediction_file='reports/txts/onmt/transformer_006_step_80400_pred.txt' \
    --log_file='reports/txts/onmt/eval_log_transformer_006_step_80400.txt'