from fuzzywuzzy import fuzz

from src.util.tokenizer import tokenize

WORD_SIMILARITY_THRESHOLD = 80
FULL_MATCH = 100
MAX_N_GRAM = 5
NONE_NER_POS_TOKEN = 0
NONE_NER_POS = '<none>'


def convert_charloc_to_wordloc(tokenized_context, tokenized_words, char_loc):
    if len(tokenized_words) == 0:
        return -2

    pointer_loc = 0
    i = 0
    j = 0
    while i < len(tokenized_context) and j < min(2, len(tokenized_words)):
        if char_loc-pointer_loc <= 5:
            if tokenized_context[i].isnumeric():
                similarity = fuzz.ratio(tokenized_context[i], tokenized_words[j])
            else:
                similarity = fuzz.partial_ratio(tokenized_context[i], tokenized_words[j])
            # print(f'{tokenized_context[i]} vs {tokenized_words[j]} = {similarity}')
            if similarity >= WORD_SIMILARITY_THRESHOLD:
                j += 1
        pointer_loc += len(tokenized_context[i]) + 1
        i += 1
    if j >= min(2, len(tokenized_words)):
        return i-j
    else:
        return -1


def is_end_punctuations(token):
    return token in '.!?'


def get_sentence_location_from_answer_word_index(tokenized_context, answer_word_idx):
    start_idx = answer_word_idx-1
    end_idx = answer_word_idx
    while start_idx > -1 and not is_end_punctuations(tokenized_context[start_idx]):
        start_idx -= 1
    while end_idx < len(tokenized_context)-1 and not is_end_punctuations(tokenized_context[end_idx]):
        end_idx += 1
    return start_idx+1, end_idx


def create_ner_tensor(tokenized_context, entities, ner_textdict, return_in_tensor=True):
    ner_tensor = [NONE_NER_POS_TOKEN if return_in_tensor else NONE_NER_POS for _ in range(len(tokenized_context))]

    if len(entities) == 0:
        return ner_tensor

    pointer_loc = 0
    i = 0
    j = 0
    k = 0
    entities_name = tokenize(entities[j]['name'])
    while i < len(tokenized_context) and entities_name != None:
        pointer_loc += len(tokenized_context[i]) + 1
        if entities[j]['begin_offset'] - pointer_loc <= 0:
            similarity = fuzz.partial_ratio(tokenized_context[i], entities_name[k])
            # print(f'{tokenized_context[i]} vs {entities_name[k]} = {similarity}')
            if similarity >= WORD_SIMILARITY_THRESHOLD:
                ner_tensor[i] = ner_textdict.word2index[entities[j]['type']] if return_in_tensor else \
                    entities[j]['type']
                k += 1
                if k == len(entities_name):
                    j += 1
                    k = 0
                    entities_name = None if j == len(entities) else tokenize(entities[j]['name'])
            i += 1

    return ner_tensor


def flatten(list):
    new_list = []
    for list_ in list:
        new_list.extend(list_)
    return new_list


def calc_n_gram_similarity(n, token_1, postags, j):
    n_gram = ''
    if j + n < len(postags):
        for k in range(n):
            n_gram += postags[j + k][0]
        # print(f'{n}-gram: {token_1} vs {n_gram} = {fuzz.ratio(token_1, n_gram)}')
        return fuzz.ratio(token_1, n_gram)
    else:
        return 0


def create_postags_tensor(tokenized_context, postags_, postags_textdict, return_in_tensor=True):
    pos_tensor = [NONE_NER_POS_TOKEN if return_in_tensor else NONE_NER_POS for _ in range(len(tokenized_context))]

    if len(postags_) == 0:
        return pos_tensor

    average_sim = []
    j = 0
    postags = flatten(postags_)
    for i in range(len(tokenized_context)):
        n = 1
        found = False
        iter_limit = MAX_N_GRAM - max(0, i + 5 - len(tokenized_context))
        prev_n_gram_similarity = 0
        while n <= iter_limit and not found:
            n_gram_similarity = calc_n_gram_similarity(n, tokenized_context[i], postags, j)
            if n_gram_similarity == 0:
                pos_tensor[i] = NONE_NER_POS_TOKEN if return_in_tensor else NONE_NER_POS
                found = True
            elif n_gram_similarity != FULL_MATCH and n < iter_limit:
                if n_gram_similarity > prev_n_gram_similarity:
                    prev_n_gram_similarity = n_gram_similarity
                elif n_gram_similarity <= prev_n_gram_similarity:
                    j -= 1
                    pos_tensor[i] = postags_textdict.word2index[postags[j - n + 1][1]] if return_in_tensor else \
                        postags[j - n + 1][1]
                    # print(f'\t{tokenized_context[i]} {postags[j-n+1][1]}')
                    j += n
                    found = True
            elif n_gram_similarity >= WORD_SIMILARITY_THRESHOLD:
                pos_tensor[i] = postags_textdict.word2index[postags[j - n + 1][1]] if return_in_tensor else \
                    postags[j - n + 1][1]
                # print(f'\t{tokenized_context[i]} {postags[j-n+1][1]}')
                j += n
                found = True
            else:
                pos_tensor[i] = NONE_NER_POS_TOKEN if return_in_tensor else NONE_NER_POS
                found = True
            n += 1
        average_sim.append(n_gram_similarity if n_gram_similarity > prev_n_gram_similarity else prev_n_gram_similarity)
    # average_sim = sum(average_sim)/len(average_sim)
    # print(f'Average similarity: {average_sim:.2f}%')
    return pos_tensor
