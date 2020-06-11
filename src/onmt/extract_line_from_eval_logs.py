import argparse
import glob

UNCASED = [13, 14, 36, 37, 38, 39, 40, 41]
CASED = [11, 12, 32, 33, 42, 43, 44, 45]
COPY = [12, 13, 32, 33, 36, 37, 38, 39, 42, 43]
COVERAGE = [32, 33, 36, 37]


def output_line_conditional(rnn_type, exp_number, line):
    output = rnn_type
    if exp_number in UNCASED:
        output += '_uncased'
    elif exp_number in CASED:
        output += '_cased'
    if exp_number in COPY:
        output += '_copy'
    if exp_number in COVERAGE:
        output += '_coverage'

    output += f':{line.split("Prediction: ")[-1].strip()}'

    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--eval_log_dir',
        type=str,
        required=True,
        help='Path directory which contains all eval_logs'
    )
    parser.add_argument(
        '--line_number',
        type=int,
        required=True,
        help='Line of each eval_log file which will be extracted'
    )
    args = parser.parse_args()
    path = args.eval_log_dir
    line_number = args.line_number

    outputs = []

    files = list(glob.glob(f'{path}/eval_log*'))
    for file in files:
        splitted = file.split('_')
        rnn_type = splitted[2]
        exp_number = int(splitted[3])
        with open(file, 'r') as f_in:
            for i, line in enumerate(f_in):
                if i == line_number-1:
                    outputs.append(output_line_conditional(rnn_type, exp_number, line))

    outputs.sort()
    print(f'Printing all line-{line_number} from eval_logs in directory {path}')
    print('-' * 50)
    for output in outputs:
        print(output)
