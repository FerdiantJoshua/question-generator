import os

import pandas as pd


if __name__ == '__main__':
    SAVE_DIR_PATH = 'data/raw/TyDiQA'

    os.system('wget https://storage.googleapis.com/tydiqa/v1.1/tydiqa-goldp-v1.1-train.json')
    os.system('wget https://storage.googleapis.com/tydiqa/v1.1/tydiqa-goldp-v1.1-dev.tgz')

    os.system('tar -xvf "tydiqa-goldp-v1.1-dev.tgz"')

    df1 = pd.read_json('tydiqa-goldp-v1.1-dev/tydiqa-goldp-dev-indonesian.json')
    df2 = pd.read_json('tydiqa-goldp-v1.1-train.json')
    print(df2)

    df2['language'] = df2.data.apply(lambda x: x['paragraphs'][0]['qas'][0]['id'].split('-')[0])

    df2_temp = df2[df2.language=='indonesian']
    df2_temp.drop(columns=['language']).to_json('tydiqa-goldp-v1.1-train-indonesian.json')

    os.system(f'mv "tydiqa-goldp-v1.1-train-indonesian.json" "{SAVE_DIR_PATH}/."')
    os.system(f'mv "tydiqa-goldp-v1.1-dev/tydiqa-goldp-dev-indonesian.json" "{SAVE_DIR_PATH}/tydiqa-goldp-v1.1-dev-indonesian.json"')
    os.system('rm -rf tydiqa-goldp-v1.1-dev/')
    os.system('rm -rf tydiqa-goldp-v1.1-dev.tgz')
    os.system('rm -rf tydiqa-goldp-v1.1-train.json')
