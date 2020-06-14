import json
import random

import numpy as np

from src.preprocess.call_external_api import get_ner, get_pos_tag
from src.preprocess.features import NONE_NER_POS, is_end_punctuations, create_ner_tensor, create_postags_tensor
from src.util.file_handler import print_input_along_feature
from src.util.tokenizer import tokenize, normalize_string


def sentenize(tokenized_input, entities, postags):
    final_sentences = []
    final_entities = []
    final_postags = []

    temp_sent = []
    temp_entities = []
    temp_postag = []
    for i in range(len(tokenized_input)):
        temp_sent.append(tokenized_input[i])
        temp_entities.append(entities[i])
        temp_postag.append(postags[i])
        if is_end_punctuations(tokenized_input[i]):
            final_sentences.append(temp_sent.copy())
            final_entities.append(temp_entities.copy())
            final_postags.append(temp_postag.copy())
            temp_sent = []
            temp_entities = []
            temp_postag = []
    return final_sentences, final_entities, final_postags


def get_entities_position_group(entities):
    entities_position_group = []
    entity_position_group = []
    prev_ne = NONE_NER_POS
    i = -1
    for i in range(len(entities)):
        if entities[i] != NONE_NER_POS and prev_ne == NONE_NER_POS:
            entity_position_group.append(i)
        elif entities[i] != prev_ne and prev_ne != NONE_NER_POS:
            entity_position_group.append(i - 1)
            entities_position_group.append(tuple(entity_position_group))
            entity_position_group = []
        prev_ne = entities[i]
    if i >= 0 and prev_ne != NONE_NER_POS:
        entity_position_group.append(i)
        entities_position_group.append(tuple(entity_position_group))
    return entities_position_group


def get_random_answer_loc(tokenized_input, entities, entity_chance=0.8, seed=42):
    random.seed(seed)
    entities_position_group = get_entities_position_group(entities)
    get_from_entity = random.random()
    if len(entities_position_group) > 0 and get_from_entity <= entity_chance:
        answer_loc = entities_position_group[random.randint(0, len(entities_position_group) - 1)]
    else:
        start_idx = random.randint(0, len(tokenized_input) - 2)
        length = random.randint(2, 6)
        end_idx = min(start_idx + length, len(tokenized_input))
        answer_loc = (start_idx, end_idx)
    is_answer = ['0' if i not in range(answer_loc[0], answer_loc[1] + 1) else '1' for i in range(len(tokenized_input))]
    return is_answer


def prepare_featured_input(input_text, output_file_name='free_input.txt', manual_ne_postag=False, lower=False, seed=42):
    is_answer_sents = []
    is_cased_sents = []
    if manual_ne_postag:
        entities = json.loads(input('Enter the named entities (list of dicts):').replace('\'', '"'))
        postags = json.loads(input('Enter the postags (list of lists of lists):').replace('\'', '"'))
    else:
        try:
            entities = get_ner(input_text)['entities']
            postags = get_pos_tag(input_text)['postags']
        except TimeoutError as e:
            print('Unable to invoke the NE and/or Pos Tag API. Please check your VPN or your internet connection:', e)
            exit(1)
    tokenized_input = tokenize(normalize_string(input_text))
    entities = create_ner_tensor(tokenized_input, entities, ner_textdict=None, return_in_tensor=False)
    postags = create_postags_tensor(tokenized_input, postags, postags_textdict=None, return_in_tensor=False)
    tokenized_sents, entity_sents, postag_sents = sentenize(tokenized_input, entities, postags)
    for i in range(len(tokenized_sents)):
        is_answer_sents.append(get_random_answer_loc(tokenized_sents[i], entity_sents[i], seed=seed))
        is_cased = []
        for j in range(len(tokenized_sents[i])):
            is_cased.append(
                '1' if j < len(tokenized_sents[i]) and any(c.isupper() for c in tokenized_sents[i][j]) \
                    else '0'
            )
        is_cased_sents.append(is_cased)

    tokenized_sents = np.array(tokenized_sents)

    # YES, DIRTY CODE. But have no choice to force the numpy to keep the input as array-of-list instead of pure array
    is_answer_sents = np.array(is_answer_sents+[[]])[:-1]
    is_cased_sents = np.array(is_cased_sents+[[]])[:-1]
    entity_sents = np.array(entity_sents+[[]])[:-1]
    postag_sents = np.array(postag_sents+[[]])[:-1]

    is_answer_sents = np.expand_dims(is_answer_sents, axis=-1)
    is_cased_sents = np.expand_dims(is_cased_sents, axis=-1)
    entity_sents = np.expand_dims(entity_sents, axis=-1)
    postag_sents = np.expand_dims(postag_sents, axis=-1)

    if lower:
        features = np.concatenate((is_answer_sents, is_cased_sents, entity_sents, postag_sents), axis=-1)
    else:
        features = np.concatenate((is_answer_sents, entity_sents, postag_sents), axis=-1)
    with open(output_file_name, 'w', encoding='utf-8') as f_out:
        for i in range(len(tokenized_sents)):
            if lower:
                f_out.write((print_input_along_feature(tokenized_sents[i], features[i]) + '\n').lower())
            else:
                f_out.write((print_input_along_feature(tokenized_sents[i], features[i]) + '\n'))

def sent_tokenize(input_text):
    tokenized_sents = []
    tokenized_input = tokenize(normalize_string(input_text))
    sentence = []
    for token in tokenized_input:
        sentence.append(token)
        if is_end_punctuations(token):
            tokenized_sents.append(sentence.copy())
            sentence = []
    return tokenized_sents


def prepare_simple_input(input_text, file_name='free_input.txt'):
    tokenized_sents = sent_tokenize(input_text)
    with open(file_name, 'w') as f_out:
        for sentence in tokenized_sents:
            f_out.write(' '.join(sentence) + '\n')
