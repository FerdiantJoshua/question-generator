import argparse
import os
from os.path import dirname
import sys

sys.path.insert(1, dirname(dirname(sys.path[0])))

from src.util.file_handler import FEAT_SEP
from src.util import file_handler


def extract_token_feature(data, return_answer=True):
    data = data.split()
    if len(data[0].split(FEAT_SEP)) < 2:
        if return_answer:
            raise ValueError('Data doesn\'t contain additional features. please set --no_answer_feature for this type of dataset')
        else:
            return data, None
    context = []
    answer = []
    for features in data:
        features = features.split(FEAT_SEP)
        context.append(features[0])
        if int(features[1]):
            answer.append(features[0])
    return context, answer


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--train_src',
        default=None,
        type=str,
        required=True,
        help='Train source file (e.g.: ./data/processed/train/squad_id_cased_source.txt)',
    )
    parser.add_argument(
        '--train_tgt',
        default=None,
        type=str,
        required=True,
        help='Train target file (e.g.: ./data/processed/train/squad_id_cased_source.txt)',
    )
    parser.add_argument(
        '--valid_src',
        default=None,
        type=str,
        required=False,
        help='Valid source file (e.g.: ./data/processed/valid/squad_id_cased_source.txt)',
    )
    parser.add_argument(
        '--valid_tgt',
        default=None,
        type=str,
        required=False,
        help='Valid target file (e.g.: ./data/processed/valid/squad_id_cased_source.txt)',
    )
    parser.add_argument(
        '--test_src',
        default=None,
        type=str,
        required=False,
        help='Test source file (e.g.: ./data/processed/test/squad_id_cased_source.txt)',
    )
    parser.add_argument(
        '--test_tgt',
        default=None,
        type=str,
        required=False,
        help='Test target file (e.g.: ./data/processed/test/squad_id_cased_source.txt)',
    )
    parser.add_argument(
        '--save_dir_root',
        default='./data/processed/huggingface',
        type=str,
        required=False,
        help='Save dir root file (e.g.: ./data/processed/huggingface)',
    )
    parser.add_argument(
        '--no_answer_feature',
        action='store_true',
        help='Boolean parameter to add answer feature or not',
    )
    parser.add_argument(
        '--add_special_tokens',
        action='store_true',
        help='Boolean parameter to add special tokens (<s>, <sep>, </s>) or not',
    )
    args = parser.parse_args()

    data_path_tuples = [('train', args.train_src, args.train_tgt),
                        ('val', args.valid_src, args.valid_tgt),
                        ('test', args.test_src, args.test_tgt)]
    for data_path_tuple in data_path_tuples:
        save_dir = f'{args.save_dir_root}/{data_path_tuple[0]}'
        print(f'Data are saved in {save_dir}')
        os.makedirs(save_dir, exist_ok=True)
        source_data = file_handler.load_txt(data_path_tuple[1])
        target_data = file_handler.load_txt(data_path_tuple[2])
        assert len(source_data) == len(target_data), \
            f'Total number of lines of source and target data must be same! Found {len(source_data)} and {len(target_data)}'

        save_file_name = 'sentence_pairs_spec_tokens.txt' if args.add_special_tokens else 'sentence_pairs.txt'
        with open(f'{save_dir}/{save_file_name}', 'w') as f_out:
            for i in range(len(source_data)):
                if not source_data[i].strip():
                    continue

                src_context, src_answer = extract_token_feature(source_data[i].strip(), not args.no_answer_feature)
                if args.no_answer_feature:
                    if args.add_special_tokens:
                        f_out.write(f'<s> {" ".join(src_context)} <sep> {target_data[i].strip()} </s>\n')
                    else:
                        f_out.write(f'{" ".join(src_context)}\t{target_data[i].strip()}\n')
                else:
                    if args.add_special_tokens:
                        f_out.write(f'<s> {" ".join(src_context)} <sep> {" ".join(src_answer)} <sep> {target_data[i].strip()} </s>\n')
                    else:
                        f_out.write(f'{" ".join(src_context)}\t{" ".join(src_answer)}\t{target_data[i].strip()}\n')
