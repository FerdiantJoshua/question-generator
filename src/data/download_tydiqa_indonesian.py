import json
import os

import pandas as pd


def extract_indonesian(path, type_='train'):
    n = 0
    combined = []
    with open(f'{path}/tydiqa-v1.0-{type_}-indonesian.jsonl', 'r') as f_in:
        for line in f_in:
            if '"language":"indonesian"' in line:
                combined.append(line)
                n += 1
    print(f'Total: {n}')

    with open(f'{path}/tydiqa-v1.0-{type_}-indonesian.jsonl', 'w') as f_out:
        f_out.writelines(json.dumps({'data': combined}))


if __name__ == '__main__':
    SAVE_DIR_PATH = 'data/raw/TyDiQA'

    os.system('wget "https://storage.googleapis.com/tydiqa/v1.0/tydiqa-v1.0-dev.jsonl.gz"')
    os.system('gunzip "tydiqa-v1.0-dev.jsonl.gz"')
    os.system(f'mv "tydiqa-v1.0-dev.jsonl" "{SAVE_DIR_PATH}/TyDiQA/tydiqa-v1.0-dev-indonesian.jsonl"')

    os.system('wget "https://storage.googleapis.com/tydiqa/v1.0/tydiqa-v1.0-train.jsonl.gz"')
    os.system('gunzip "tydiqa-v1.0-train.jsonl.gz"')
    os.system(f'mv "tydiqa-v1.0-train.jsonl" "{SAVE_DIR_PATH}/TyDiQA/tydiqa-v1.0-train-indonesian.jsonl"')

    extract_indonesian(SAVE_DIR_PATH, 'dev')
    extract_indonesian(SAVE_DIR_PATH, 'train')

    df_tydiqa = pd.read_json(f'{SAVE_DIR_PATH}/tydiqa-v1.0-train-indonesian.jsonl', encoding='utf8')
    df_tydiqa_test = pd.read_json(f'{SAVE_DIR_PATH}/tydiqa-v1.0-dev-indonesian.jsonl', encoding='utf8')
    print(df_tydiqa.shape)
    print(df_tydiqa_test.shape)
    df_tydiqa.head()
    df_tydiqa_test.head()
    print(json.loads(df_tydiqa.iloc[0]['data']))
    print(json.loads(df_tydiqa_test.iloc[0]['data']))
