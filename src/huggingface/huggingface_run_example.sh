echo 'This shell example is not intended to be executed. Please read and use all the provided commands in this file manually.'
exit 1
#================================================== PREPROCESS ==================================================
python src/huggingface/prepare_data.py \
    --train_src='data/processed/train/squad_id_cased_source.txt' \
    --train_tgt='data/processed/train/squad_id_cased_target.txt' \
    --valid_src='data/processed/val/squad_id_cased_source.txt' \
    --valid_tgt='data/processed/val/squad_id_cased_target.txt' \
    --test_src='data/processed/test/squad_id_cased_source.txt' \
    --test_tgt='data/processed/test/squad_id_cased_target.txt' \
    --add_special_token

#================================================== TRAIN TOKENIZER ==================================================
python src/huggingface/prepare_tokenizer_corpus.py --merge \
    --squad_path='data/processed/train-v2.0-translated_fixed_enhanced.json'

python src/huggingface/train_tokenizer.py --resource_dir='data/processed/tokenizer/merged' \
    --language_name='contexts_questions' \
    --tokenizer_type='byte_bpe' \
    --model_type='gpt2'


#================================================== PREPARE CONFIG ==================================================
#-------------------------------------------------- CREATE NEW --------------------------------------------------
import json
from transformers import GPT2Config
with open('models/tokenizer/gpt2/contexts_questions/config.json', 'w') as f:
    json.dump(GPT2Config().to_dict(), f)

#-------------------------------------------------- COPY FROM EXISTING --------------------------------------------------
cp 'src/huggingface/config_examples/config_gpt2.example.json' 'models/tokenizer/gpt2/contexts_questions/config.json'

#================================================== TRAIN ==================================================
#-------------------------------------------------- GPT2 --------------------------------------------------
python src/huggingface/run_language_modeling.py \
    --overwrite_output_dir \
    --output_dir='models/finetuned/gpt2' \
    --model_type=gpt2 \
    --line_by_line \
    --config_name='models/tokenizer/gpt2/contexts_questions/config.json' \
    --tokenizer_name='models/tokenizer/gpt2/contexts_questions' \
    --do_train \
    --train_data_file='data/processed/huggingface/train/sentence_pairs_spec_tokens.txt' \
    --do_eval \
    --eval_data_file='data/processed/huggingface/val/sentence_pairs_spec_tokens.txt' \
    --per_gpu_train_batch_size 4 \
    --num_train_epochs 5 \
    --block_size 128

# .................................................. TO CONTINUE TRAINING ..................................................
    --model_name_or_path='models/finetuned/gpt2/checkpoint-1000' \

#-------------------------------------------------- BERT --------------------------------------------------
#@TODO

#================================================== INFERENCE ==================================================
python src/huggingface/run_generation.py \
    --model_type=gpt2 \
    --model_name_or_path='models/finetuned/gpt2/gpt2_00X' \
    --length=100 --stop_token='</s>' \
    --num_return_sequences=1 \
    --num_beams=5 \
    --temperature=0.1 \
    --no_repeat_ngram_size=2

#-------------------------------------------------- FOR INPUT FROM FILES --------------------------------------------------
python src/huggingface/separate_source_target.py --input_file_path='data/processed/huggingface/test/sentence_pairs_spec_tokens.txt'
python src/huggingface/run_generation.py \
    --model_type=gpt2 \
    --model_name_or_path='models/finetuned/gpt2/gpt2_00X' \
    --input_file_path='data/processed/huggingface/test/sentence_pairs_spec_tokens_source.txt' \
    --output_file_path='reports/txts/huggingface/gpt2_00X_epoch_3_pred.txt' \
    --length=100 --stop_token='</s>' \
    --num_return_sequences=1 \
    --num_beams=5 \
    --temperature=0.1 \
    --no_repeat_ngram_size=2

#================================================== EVALUATE ==================================================
python src/huggingface/run_evaluation.py \
    --target_file='data/processed/huggingface/test/sentence_pairs_spec_tokens_target.txt' \
    --prediction_file='reports/txts/huggingface/gpt2_00X_epoch_3_pred.txt'
