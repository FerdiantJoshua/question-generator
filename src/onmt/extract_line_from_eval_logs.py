import argparse
import ast
import glob
import os

UNCASED = [13, 14, 36, 37, 38, 39, 40, 41]
CASED = [11, 12, 32, 33, 42, 43, 44, 45]
COPY = [12, 13, 32, 33, 36, 37, 38, 39, 42, 43]
COVERAGE = [32, 33, 36, 37]


def output_line_conditional(rnn_type, exp_number, line, is_eval=False):
    output = rnn_type
    if exp_number in UNCASED:
        output += '_uncased'
    elif exp_number in CASED:
        output += '_cased'
    if exp_number in COPY:
        output += '_copy'
    if exp_number in COVERAGE:
        output += '_coverage'

    if is_eval:
        score = ast.literal_eval(line.strip().split('evaluations: ')[-1])
        for key in score: score[key] = round(score[key] * 100, 2)
        output += f":{score['Bleu_1']}\t{score['Bleu_2']}\t{score['Bleu_3']}\t{score['Bleu_4']}\t{score['ROUGE_L']}"
    else:
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
        '--line_numbers',
        type=str,
        required=True,
        help='Line of each eval_log file which will be extracted'
    )
    parser.add_argument(
        '--only_print_eval',
        action='store_true',
        help='Boolean parameters to set if the to be extracted line is an eval result'
    )
    args = parser.parse_args()
    path = args.eval_log_dir
    line_numbers = list(map(int, args.line_numbers.split(',')))

    input_text = ''
    answer_text = ''
    target_text = ''
    outputs = []

    files = list(glob.glob(f'{path}/eval_log*'))
    if len(files) == 0: raise FileNotFoundError(f'No eval_log* files found in directory {path}!')
    for line_number in line_numbers:
        for file in files:
            splitted = os.path.split(file)[-1].split('_')
            rnn_type = splitted[2]
            exp_number = int(splitted[3])
            with open(file, 'r') as f_in:
                for i, line in enumerate(f_in):
                    if not args.only_print_eval:
                        if i == line_number - 4 and exp_number == 11: input_text = line.strip()
                        elif i == line_number - 3 and exp_number == 11: answer_text = line.strip()
                        elif i == line_number - 2 and exp_number == 11: target_text = line.strip()
                    if i == line_number-1:
                        outputs.append(output_line_conditional(rnn_type, exp_number, line, args.only_print_eval))

        outputs.sort()
        print(f'Printing all line-{line_number} from eval_logs in directory {path}')
        print('-' * 50)
        if not args.only_print_eval:
            print(input_text)
            print(answer_text)
            print(target_text)
        for output in outputs:
            print(output)
