import argparse
import logging
import os
import sys

MODEL_LIST = [
    'lstm_032_step_16050.pt', 'lstm_036_step_16050.pt', 'lstm_038_step_16050.pt',
    'lstm_040_step_16050.pt', 'lstm_042_step_16050.pt', 'lstm_044_step_16050.pt',
    'gru_033_step_32100.pt', 'gru_037_step_32100.pt', 'gru_039_step_32100.pt',
    'gru_041_step_32100.pt', 'gru_043_step_32100.pt', 'gru_045_step_32100.pt',
    'transformer_011_step_120600.pt', 'transformer_012_step_120600.pt',
    'transformer_013_step_120600.pt', 'transformer_014_step_120600.pt'
]
DATA_LIST = ['squad', 'tydiqa']
UNCASED = [13, 14, 36, 37, 38, 39, 40, 41]

if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(levelname)s [%(name)s.%(funcName)s:%(lineno)d] %(message)s',
        datefmt='%H:%M:%S',
        stream=sys.stdout
    )
    logger = logging.getLogger('bulk_inference')

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--log_dir',
        type=str,
        required=True,
        help='Path directory which contains all eval_logs'
    )
    parser.add_argument(
        '--n_beams',
        type=str,
        required=True,
        help='Number of beam used in beam search. Can pass several values to infer in several beam width. (e.g. 2,4,5)'
    )
    parser.add_argument(
        '--dry_run',
        action='store_true',
        help='Boolean parameters to test creating the files'
    )
    args = parser.parse_args()

    log_dir = args.log_dir
    n_beams = list(map(int, args.n_beams.split(',')))

    logger.info(f'All outputs will be saved in {log_dir}. Now creating dir...')
    os.makedirs(log_dir, exist_ok=True)

    for model in MODEL_LIST:
        model_type = '_'.join(model.split('_')[:2])
        exp_num = int(model_type.split('_')[1])
        case = 'uncased' if exp_num in UNCASED else 'cased'
        for data in DATA_LIST:
            for n_beam in n_beams:
                logger.info(f'Evaluating {model} on dataset {data} with beam {n_beam}...')
                output_path = f'{log_dir}/{model_type}_pred_{data}_b{n_beam}.txt'
                eval_log_path = f'{log_dir}/eval_log_{model_type}_{data}_b{n_beam}.txt'
                if args.dry_run:
                    with open(output_path, 'w') as f_out:
                        f_out.write(f'data/processed/test/{data}_id_split0.9_{case}_source.txt')
                    with open(eval_log_path, 'w') as f_out:
                        f_out.writelines('\n'.join([f'data/processed/test/{data}_id_split0.9_{case}_source.txt',
                                          f'data/processed/test/{data}_id_split0.9_{case}_target.txt',
                                          output_path]))
                else:
                    os.system(f"onmt_translate -model 'models/checkpoints/onmt/{model}' \
                        -src 'data/processed/test/{data}_id_split0.9_{case}_source.txt' \
                        -output {output_path} \
                        -replace_unk \
                        -seed 42 \
                        -beam_size {n_beam} \
                        -max_length 20"
                    )
                    os.system(f"python src/onmt/run_evaluation.py \
                        --source_file='data/processed/test/{data}_id_split0.9_{case}_source.txt' \
                        --target_file='data/processed/test/{data}_id_split0.9_{case}_target.txt' \
                        --prediction_file={output_path} \
                        --log_file={eval_log_path}"
                    )
