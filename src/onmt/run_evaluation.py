import argparse
from os.path import dirname
import sys
import time

from nlgeval import NLGEval

sys.path.insert(1, dirname(dirname(sys.path[0])))
from src.util.file_handler import FEAT_SEP


def calculate_eval_score(nlgeval, references, hypothesis, ndigits=4):
    result_dict = nlgeval.compute_individual_metrics(references, hypothesis)
    for key, val in result_dict.items():
        result_dict[key] = round(val, ndigits)
    result_dict['Bleu_avg'] = (result_dict['Bleu_1'] + result_dict['Bleu_2'] + result_dict['Bleu_3'] + result_dict['Bleu_4'])/4
    return result_dict


def dict_mean(dict_list):
    mean_dict = {}
    for key in dict_list[0].keys():
        mean_dict[key] = round(sum(d[key] for d in dict_list) / len(dict_list), 4)
    return mean_dict


def merge_and_print_to_file(src_file, tgt_file, pred_file, output_log_file, nlgeval):
    with open(src_file, 'r') as f_in:
        src = f_in.readlines()
    with open(tgt_file, 'r') as f_in:
        target = f_in.readlines()
    with open(pred_file, 'r') as f_in:
        predictions = f_in.readlines()

    with open(output_log_file, 'w') as f_out:
        evaluations = []
        start_time = time.time()
        f_out.write(f'Total data: {len(src)}\n\n')
        for i in range(len(src)):
            answer = ''
            orig_text = ''
            tokens = src[i].split()
            for token in tokens:
                word, is_answer = token.split(FEAT_SEP)[:2]
                orig_text += word + ' '
                if int(is_answer): answer += word + ' '
            f_out.write(f'Data: {src[i]}')
            f_out.write(f'Text: {orig_text}\n')
            f_out.write(f'Answer: {answer}\n')
            f_out.write(f'Target: {target[i]}')
            f_out.write(f'Prediction: {predictions[i]}')
            eval_score = calculate_eval_score(nlgeval, [target[i]], predictions[i])
            evaluations.append(eval_score)
            f_out.write(f'Eval score: {eval_score}\n\n')
            if i % 500 == 0:
                print(f'{i}: {time.time()-start_time:.2f}s')
        averaged_eval = dict_mean(evaluations)
        f_out.write(f'Average evaluations: {averaged_eval}')
    return averaged_eval


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--source_file',
        default='data/processed/test/squad_id_cased_source.txt',
        type=str,
        required=False,
        help='Source file containing original input, each data are separated by line',
    )
    parser.add_argument(
        '--target_file',
        default='data/processed/test/squad_id_cased_target.txt',
        type=str,
        required=False,
        help='Target file containing ground truth values, each data are separated by line',
    )
    parser.add_argument(
        '--prediction_file',
        default='reports/txts/onmt/pred.txt',
        type=str,
        required=False,
        help='Prediction file containing model predictions, each data are separated by line',
    )
    parser.add_argument(
        '--log_file',
        default='reports/txts/onmt/evaluation_log.txt',
        type=str,
        required=False,
        help='Complete evaluation log file will be saved under this path',
    )
    args = parser.parse_args()

    nlgeval = NLGEval(metrics_to_omit=['CIDEr', 'EmbeddingAverageCosineSimilairty', 'EmbeddingAverageCosineSimilarity',
                                       'GreedyMatchingScore', 'SkipThoughtCS',
                                       'VectorExtremaCosineSimilarity'])
    averaged_eval = merge_and_print_to_file(args.source_file, args.target_file, args.prediction_file, args.log_file, nlgeval)
    print(averaged_eval)
