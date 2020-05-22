import argparse
from os.path import dirname, exists
from os import system
import pandas as pd
import sys
sys.path.insert(1, dirname(dirname(sys.path[0])))

from src.preprocess import do_preprocess, shuffle, split_by_k
from src.util.file_handler import create_data_file

BASE_PATH = './'


def delete_unfound_answers(df_squad):
    total_questions_before = 0
    total_questions = 0
    for taken_topic_idx in range(df_squad.shape[0]):
        for taken_context_idx in range(len(df_squad.iloc[taken_topic_idx]['paragraphs'])):
            i = 0
            qas = df_squad.iloc[taken_topic_idx]['paragraphs'][taken_context_idx]['qas']
            while i < len(qas):
                total_questions_before += 1
                indonesian_answer = qas[i].get('indonesian_answers') or qas[i].get('indonesian_plausible_answers')
                if indonesian_answer[0]['answer_start'] < 0:
                    qas.pop(i)
                else:
                    i += 1
                    total_questions += 1
    print(f'Left: {total_questions}. Deleted: {total_questions_before-total_questions}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--lower',
        action='store_true',
        help='Boolean parameter to lower the dataset (and add case feature) or not',
    )
    parser.add_argument(
        '--src_max_len',
        default='60',
        type=int,
        required=False,
        help='Max length of source input',
    )
    parser.add_argument(
        '--tgt_max_len',
        default='20',
        type=int,
        required=False,
        help='Max length of source input',
    )
    args = parser.parse_args()
    assert exists(args.embedding_file), f'Embedding file at {args.embedding_file} is not found'

    squad_train_dataset_path = f'{BASE_PATH}data/processed/train-v2.0-translated_fixed_enhanced.json'
    squad_test_dataset_path = f'{BASE_PATH}data/processed/dev-v2.0-translated_fixed_enhanced.json'

    df_squad = pd.read_json(squad_train_dataset_path)
    df_squad_test = pd.read_json(squad_test_dataset_path)
    print(df_squad.shape)
    print(df_squad.head())
    print(df_squad_test.shape)
    print(df_squad_test.head())

    print('Deleting unfound answer...')
    print('Train')
    delete_unfound_answers(df_squad)
    print('Test')
    delete_unfound_answers(df_squad_test)

    inputs, features, targets = do_preprocess(df_squad, args.lower, args.src_max_len, args.tgt_max_len)
    inputs_test, features_test, targets_test = do_preprocess(df_squad_test, args.lower, args.src_max_len, args.tgt_max_len)

    inputs, features, targets = shuffle(inputs, features, targets)
    inputs, features, targets, inputs_val, features_val, targets_val, = split_by_k(inputs, features, targets, k=0.8)
    print('Train feature shape:', features.shape)
    print('Val feature shape:', features_val.shape)

    file_name = 'squad_id_uncased' if args.lower else 'squad_id_cased'
    create_data_file([(inputs, features, targets), (inputs_val, features_val, targets_val),
                      (inputs_test, features_test, targets_test)], dir_name='onmt', file_name=file_name,
                     lower=args.lower)
    print('Train data')
    system(f'wc -l data/train/{file_name}_source.txt')
    print()
    system(f'head -5 data/train/{file_name}_source.txt')
    print()
    system(f'head -5 data/train/{file_name}_target.txt')
    print('\nTest data')
    system(f'wc -l data/test/{file_name}_source.txt')
    print()
    system(f'head -5 data/test/{file_name}_source.txt')
    print()
    system(f'head -5 data/test/{file_name}_target.txt')

if __name__ == '__main__':
    main()
