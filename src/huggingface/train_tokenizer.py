import os
from pathlib import Path

import argparse
from tokenizers import ByteLevelBPETokenizer, CharBPETokenizer, SentencePieceBPETokenizer, BertWordPieceTokenizer
from transformers import GPT2Tokenizer, XLMTokenizer

TOKENIZER_CLASSES = {
    'byte_bpe': ByteLevelBPETokenizer,
    'char_bpe': CharBPETokenizer,
    'sentence_piece': SentencePieceBPETokenizer,
    'bert_word_piece': BertWordPieceTokenizer,
}
MODEL_CLASSES = {
    'gpt2': GPT2Tokenizer,
    'xlm': XLMTokenizer,
}

LANGUAGE_LIST = (
    ('merged', 'contexts_questions'),
)
VOCAB_SIZE = 52000
MIN_FREQUENCY = 2


def get_tokenizer_construction_kwargs(tokenizer_type, save_dir, language_name):
    TOKENIZER_CONSTRUCTION_KWARGS = {
        'byte_bpe': {
            'vocab_file': f'{save_dir}/{language_name}-vocab.json',
            'merges_file': f'{save_dir}/{language_name}-merges.txt',
        },
        'char_bpe': {
            'vocab_file': f'{save_dir}/{language_name}-vocab.json',
            'merges_file': f'{save_dir}/{language_name}-merges.txt',
        },
        'sentence_piece': {
            'vocab_file': f'{save_dir}/{language_name}-vocab.json',
            'merges_file': f'{save_dir}/{language_name}-merges.txt',
        },
        'bert_word_piece': {
            'vocab_file': f'{save_dir}/{language_name}-vocab.txt',
        },
    }
    kwargs = TOKENIZER_CONSTRUCTION_KWARGS[tokenizer_type]
    kwargs.update({
        'unk_token': '<unk>',
        'bos_token': '<s>',
        'eos_token': '</s>',
        'pad_token': '<pad>',
        'sep_token': '<sep>',
    })
    return kwargs


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--resource_dir',
        default=None,
        type=str,
        required=True,
        help='Resource directory path. e.g.: train, test, etc)',
    )
    parser.add_argument(
        '--tokenizer_type',
        default=None,
        type=str,
        choices=TOKENIZER_CLASSES,
        required=True,
        help='Tokenizer type (directories will be created with this name',
    )
    parser.add_argument(
        '--model_type',
        default=None,
        type=str,
        choices=MODEL_CLASSES,
        required=True,
        help='Model type selected',
    )
    args = parser.parse_args()

    RESOURCE_DIR = f'./data/processed/tokenizer/{args.resource_dir}'
    BASIC_TOKENIZER_ROOT_DIR = f'./models/tokenizer/{args.tokenizer_type}'
    MODEL_TOKENIZER_ROOT_DIR = f'./models/tokenizer/{args.model_type}'

    for language in LANGUAGE_LIST:
        print(f'Training tokenizer for language: {language[1]} (as {language[0]})')
        paths = [str(x) for x in Path(f'{RESOURCE_DIR}/{language[0]}').glob('**/*.txt')]
        print(paths)
        basic_tokenizer = TOKENIZER_CLASSES[args.tokenizer_type]()
        basic_tokenizer.train(files=paths, vocab_size=VOCAB_SIZE, min_frequency=MIN_FREQUENCY, special_tokens=[
            '<s>',
            '<pad>',
            '<sep>',
            '</s>',
            '<unk>',
            '<mask>',
        ])
        basic_tokenizer_save_dir = f'{BASIC_TOKENIZER_ROOT_DIR}/{language[1]}'
        print('Basic tokenizer vocab (and optional merges) are saved in', basic_tokenizer_save_dir)
        os.makedirs(basic_tokenizer_save_dir)
        basic_tokenizer.save(basic_tokenizer_save_dir, language[1])

        model_tokenizer = MODEL_CLASSES[args.model_type](
            **get_tokenizer_construction_kwargs(args.tokenizer_type, basic_tokenizer_save_dir, language[1])
        )
        model_tokenizer_save_dir = f'{MODEL_TOKENIZER_ROOT_DIR}/{language[1]}'
        print('Model tokenizer vocab and merges are saved in', model_tokenizer_save_dir)
        os.makedirs(model_tokenizer_save_dir)
        model_tokenizer.save_pretrained(model_tokenizer_save_dir)
