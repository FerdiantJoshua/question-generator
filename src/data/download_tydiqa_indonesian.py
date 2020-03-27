import json

!wget "https://storage.googleapis.com/tydiqa/v1.0/tydiqa-v1.0-dev.jsonl.gz"
!gunzip "tydiqa-v1.0-dev.jsonl.gz"

!wget "https://storage.googleapis.com/tydiqa/v1.0/tydiqa-v1.0-train.jsonl.gz"
!gunzip "tydiqa-v1.0-train.jsonl.gz"

def extract_indonesian(path, type_='train'):
    n = 0
    combined = []
    with open(f'{path}/tydiqa-v1.0-{type_}.jsonl', 'r') as f_in:
      for line in f_in:
          if '"language":"indonesian"' in line:
              combined.append(line)
              n += 1
    print(f'Total: {n}')


    with open(f'{path}/tydiqa-v1.0-{type_}-indonesian.jsonl', 'w') as f_out:
        f_out.writelines(json.dumps({'data': combined}))

extract_indonesian('/content', 'dev')
extract_indonesian('/content', 'train')

!cp "tydiqa-v1.0-dev-indonesian.jsonl" "/content/drive/My Drive/TA/data/raw/TyDiQA/."
!cp "tydiqa-v1.0-train-indonesian.jsonl" "/content/drive/My Drive/TA/data/raw/TyDiQA/."

df_tydiqa = pd.read_json('/content/drive/My Drive/TA/data/raw/TyDiQA/tydiqa-v1.0-train-indonesian.jsonl', encoding='utf8')
df_tydiqa_test = pd.read_json('/content/drive/My Drive/TA/data/raw/TyDiQA/tydiqa-v1.0-dev-indonesian.jsonl', encoding='utf8')
print(df_tydiqa.shape)
print(df_tydiqa_test.shape)
df_tydiqa.head()
df_tydiqa_test.head()
print(json.loads(df_tydiqa.iloc[0]['data']))
print(json.loads(df_tydiqa_test.iloc[0]['data']))