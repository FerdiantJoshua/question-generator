# !pip install git+https://github.com/Maluuba/nlg-eval.git@master
# !nlg-eval --setup
import argparse
import time

from nlgeval import NLGEval


def calculate_eval_score(nlgeval, references, hypothesis, ndigits=4):
    result_dict = nlgeval.compute_individual_metrics(references, hypothesis)
    for key, val in result_dict.items():
        result_dict[key] = round(val, ndigits)
    result_dict['Bleu_avg'] = (result_dict['Bleu_1'] + result_dict['Bleu_2'] + result_dict['Bleu_3'] + result_dict['Bleu_4']) / 4
    return result_dict


def dict_mean(dict_list):
    mean_dict = {}
    for key in dict_list[0].keys():
        mean_dict[key] = round(sum(d[key] for d in dict_list) / len(dict_list), 4)
    return mean_dict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--target_file',
        default=None,
        type=str,
        required=True,
        help='Target file containing ground truth values, separated by lines',
    )
    parser.add_argument(
        '--prediction_file',
        default=None,
        type=str,
        required=True,
        help='Prediction file containing model predictions, separated by lines',
    )
    args = parser.parse_args()

    nlgeval = NLGEval(metrics_to_omit=['CIDEr', 'EmbeddingAverageCosineSimilairty', 'EmbeddingAverageCosineSimilarity',
                                       'GreedyMatchingScore', 'SkipThoughtCS',
                                       'VectorExtremaCosineSimilarity'])
    calculate_eval_score(nlgeval,
                         ['aku adalah anak gembala wo is picik herder'],
                         'beta shi childof peternak aku adalah anak gembala')

    with open(args.prediction_file, 'r') as f_in:
        predictions = f_in.readlines()

    with open(args.target_file, 'r') as f_in:
        target = f_in.readlines()

    evaluations = []
    start_time = time.time()
    for i in range(len(predictions)):
        eval_score = calculate_eval_score(nlgeval, [target[i]], predictions[i])
        evaluations.append(eval_score)
        if i % 500 == 0:
            print(f'{i}: {time.time() - start_time:.2f}s')

    print(dict_mean(evaluations))


if __name__ == '__main__':
    main()
