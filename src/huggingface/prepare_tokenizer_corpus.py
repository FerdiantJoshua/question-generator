import argparse
import os
import time
from os.path import dirname
import pandas as pd
import sys
sys.path.insert(1, dirname(dirname(sys.path[0])))

from src.util.tokenizer import tokenize, normalize_string


def load_context_and_question(df_squad):
    j = 0
    contexts = []
    questions = []
    start_time = time.time()
    for taken_topic_idx in range(df_squad.shape[0]):
        for taken_context_idx in range(len(df_squad.iloc[taken_topic_idx]['paragraphs'])):
            i = 0
            context = df_squad.iloc[taken_topic_idx]['paragraphs'][taken_context_idx]['context']
            contexts.append(tokenize(normalize_string(context)))

            qas = df_squad.iloc[taken_topic_idx]['paragraphs'][taken_context_idx]['qas']
            while i < len(qas):
                question = qas[i]['question']
                questions.append(tokenize(normalize_string(question)))
                i += 1
                j += 1
                if j % 10000 == 0: print(f'{j:04d}: {time.time() - start_time}s')
    return contexts, questions


def save_file_individual(context_save_path, question_save_path, contexts, questions):
    with open(context_save_path, 'w') as f_out_src:
        for context in contexts:
            f_out_src.write(f'{" ".join(context)}\n')
    with open(question_save_path, 'w') as f_out_tgt:
        for question in questions:
            f_out_tgt.write(f'{" ".join(question)}\n')


def save_file_merged(save_path, contexts, questions):
    with open(save_path, 'w') as f_out:
        for context in contexts:
            f_out.write(f'{" ".join(context)}\n')
        for question in questions:
            f_out.write(f'{" ".join(question)}\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--squad_path',
        default=None,
        type=str,
        required=True,
        help='Train source file (e.g.: ./data/processed/train-v2.0-translated_fixed_enhanced.json)',
    )
    parser.add_argument(
        '--merge',
        action='store_true',
        help='Boolean parameter to merge source and target sentences to single file or not',
    )
    args = parser.parse_args()

    create_merged = args.merge

    SAVE_DIR_ROOT = f'./data/processed/tokenizer'
    if create_merged:
        os.makedirs(f'{SAVE_DIR_ROOT}/merged', exist_ok=True)
    else:
        os.makedirs(f'{SAVE_DIR_ROOT}/source', exist_ok=True)
        os.makedirs(f'{SAVE_DIR_ROOT}/target', exist_ok=True)

    print(args.squad_path)
    df_squad = pd.read_json(args.squad_path)
    contexts, questions = load_context_and_question(df_squad)

    if create_merged:
        save_file_merged(f'{SAVE_DIR_ROOT}/merged/contexts_and_questions.txt', contexts, questions)
    else:
        save_file_individual(f'{SAVE_DIR_ROOT}/source/contexts.txt',
                             f'{SAVE_DIR_ROOT}/target/questions.txt', contexts, questions)
