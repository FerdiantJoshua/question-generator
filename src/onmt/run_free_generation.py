import argparse
import os
from os.path import dirname
import sys

sys.path.insert(1, dirname(dirname(sys.path[0])))

from src.preprocess.prepare_free_input import prepare_featured_input


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--preprocess_input_path',
        default='',
        type=str,
        required=False,
        help='Path to the preprocessed input file, if you have already had one. \
             Do not provide --preprocess_output_path if you use this argument.',
    )
    parser.add_argument(
        '--preprocess_output_path',
        default='',
        type=str,
        required=False,
        help='Preprocessed paragraph will be saved under this path, if you want to provide new input. \
             Do not provide --preprocess_input_path if you use this argument',
    )
    parser.add_argument(
        '--manual_ne_postag',
        default=False,
        action='store_true',
        help='Whether to input the NE and Pos Tag manually by hand, or provided by 3rd party Prosa.ai API. \
             This argument is not used if you provide --preprocess_input_path',
    )
    parser.add_argument(
        '--uncased',
        action='store_true',
        help='Whether to lower the provided input or nont. \
             This argument is not used if you provide --preprocess_input_path',
    )
    parser.add_argument(
        '--pred_output_path',
        default='reports/txts/onmt/pred_free_input.txt',
        type=str,
        required=False,
        help='Predicted questions will be saved under this path',
    )
    parser.add_argument(
        '--model_path',
        type=str,
        required=True,
        help='ONMT model path.',
    )
    parser.add_argument(
        '--beam_size',
        default=2,
        type=int,
        required=False,
        help='Beam size used while inferencing',
    )
    parser.add_argument(
        '--max_length',
        default=22,
        type=int,
        required=False,
        help='Max prediction length',
    )
    args = parser.parse_args()
    if (args.preprocess_input_path != '' and args.preprocess_output_path != '') or \
            (args.preprocess_input_path == '' and args.preprocess_output_path == ''):
        raise ValueError(
            'Provide ONLY either --preprocess_input_path (if you have a file with preprocessed input) \
            or --preprocess_output_path (if you want to input the paragraph manually)'
        )
    
    if args.preprocess_output_path:
        article = input('Enter the paragraph you would like to create the questions: ')    
        prepare_featured_input(article, output_file_name=args.preprocess_output_path, manual_ne_postag=args.manual_ne_postag, lower=args.uncased, seed=42)
        preprocessed_file_path = args.preprocess_output_path
    else:
        print('--preprocess_input_path parameter is provided. Will ignore --manual_ne_postag, and --uncased parameter')
        preprocessed_file_path = args.preprocess_input_path

    print('Generating questions...')
    os.system(
        f'onmt_translate -model {args.model_path} \
            -src {preprocessed_file_path} \
            -output {args.pred_output_path} -replace_unk \
            -beam_size {args.beam_size} \
            -max_length {args.max_length}'
    )

    with open(args.pred_output_path, 'r') as f_in:
        predictions = f_in.readlines()
    for prediction in predictions:
        print(prediction.strip())

if __name__ == '__main__':
    main()
