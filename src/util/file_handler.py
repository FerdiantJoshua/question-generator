import os

import numpy as np

FEAT_SEP = u'￨'


def print_input_along_feature(input, feature):
    concated = np.concatenate((np.expand_dims(input, axis=0), np.array(feature.tolist())), axis=0).T
    result = []
    for concated_token in concated:
        result.append(FEAT_SEP.join(concated_token))
    return ' '.join(result)


def create_data_file(input_features, dir_name, file_name, print_features=True, lower=False):
    i = 0
    for type_ in ['train', 'val', 'test']:
        save_dir = f'data/{dir_name}/{type_}'
        print(f'Data are saved in {save_dir}')
        os.makedirs(save_dir, exist_ok=True)
        source_f_out = open(f'{save_dir}/{file_name}_source.txt', 'w', encoding='utf-8')
        target_f_out = open(f'{save_dir}/{file_name}_target.txt', 'w', encoding='utf-8')
        src_tgt_pairs = input_features[i]
        for j in range(len(src_tgt_pairs[0])):
            if print_features:
                source_line = print_input_along_feature(src_tgt_pairs[0][j], src_tgt_pairs[1][j]) + '\n'
            else:
                source_line = ' '.join(src_tgt_pairs[0][j]) + '\n'
            target_line = ' '.join(src_tgt_pairs[2][j]) + '\n'

            if lower:
                source_line = source_line.lower()
                target_line = target_line.lower()
            source_f_out.write(source_line)
            target_f_out.write(target_line)
        source_f_out.close()
        target_f_out.close()
        i += 1
