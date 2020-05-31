import numpy as np
import re
import time

from .features import convert_charloc_to_wordloc, get_sentence_location_from_answer_word_index,\
                      create_ner_tensor, create_postags_tensor
from ..util.tokenizer import NON_ASCII_REGEX, unicode_to_ascii, normalize_string, tokenize

SENTENCE_MAX_LENGTH = 60
QUESTION_MAX_LENGTH = 20


def do_preprocess(df_squad, lower=False, sentence_max_length=SENTENCE_MAX_LENGTH, question_max_length=QUESTION_MAX_LENGTH):
    inputs = []
    is_answers = []
    is_caseds = []
    ners = []
    poss = []

    targets = []

    deleted = 0

    start_time = time.time()
    for taken_topic_idx in range(df_squad.shape[0]):
        for taken_context_idx in range(len(df_squad.iloc[taken_topic_idx]['paragraphs'])):
            context = df_squad.iloc[taken_topic_idx]['paragraphs'][taken_context_idx]['context']
            tokenized_context = tokenize(normalize_string(context))
            count_tobe_removed_chars = len(re.findall(NON_ASCII_REGEX, unicode_to_ascii(context))) * 1.5  # With assumption every nonascii is followed by space

            try:
                entities = df_squad.iloc[taken_topic_idx]['paragraphs'][taken_context_idx]['entities']
                postags = df_squad.iloc[taken_topic_idx]['paragraphs'][taken_context_idx]['postags']
            except KeyError:
                print(f'Entities/postags not found, (topic:{taken_topic_idx}, context:{taken_context_idx})')
                continue
            entities = create_ner_tensor(tokenized_context, entities, None, return_in_tensor=False)
            postags = create_postags_tensor(tokenized_context, postags, None, return_in_tensor=False)

            qas = df_squad.iloc[taken_topic_idx]['paragraphs'][taken_context_idx]['qas']
            i = 0
            while i < len(qas):
                qa = qas[i]

                indonesian_answer = qa.get('indonesian_answers') or qa.get('indonesian_plausible_answers')
                tokenized_answers = tokenize(normalize_string(indonesian_answer[0]['text']))
                answer_start = indonesian_answer[0]['answer_start'] - count_tobe_removed_chars
                answer_idx = convert_charloc_to_wordloc(tokenized_context, tokenized_answers, answer_start)
                if answer_idx < 0:
                    print(f'Not found, (topic:{taken_topic_idx}, context:{taken_context_idx}, qas:{i})')
                    deleted += 1
                    qas.pop(i)
                    continue

                sent_start_idx, sent_end_idx = get_sentence_location_from_answer_word_index(tokenized_context, answer_idx)
                tokenized_sentence = tokenized_context[sent_start_idx:sent_end_idx+1]
                if sent_end_idx-sent_start_idx+1 > sentence_max_length:
                    # print(f'Sentence too long, (topic:{taken_topic_idx}, context:{taken_content_idx})')
                    i += 1
                    continue

                ner = [entities[i] for i in range(sent_start_idx, sent_end_idx+1)]
                pos = [postags[i] for i in range(sent_start_idx, sent_end_idx+1)]

                answer_idx_range = (answer_idx-sent_start_idx, answer_idx+len(tokenized_answers)-sent_start_idx)

                is_answer = []
                is_cased = []
                for j in range(len(tokenized_sentence)):
                    is_answer.append(
                        '1' if j in range(answer_idx_range[0], answer_idx_range[1]) \
                        else '0'
                    )
                    is_cased.append(
                        '1' if j<len(tokenized_sentence) and any(c.isupper() for c in tokenized_sentence[j]) \
                        else '0'
                    )

                indonesian_question = qa['question']
                tokenized_questions = tokenize(normalize_string(indonesian_question))
                if len(tokenized_questions) > question_max_length-2:
                    # print(f'Question too long, skipped (topic:{taken_topic_idx}, context:{taken_content_idx})')
                    i += 1
                    continue

                i += 1
                inputs.append(tokenized_sentence)
                is_answers.append(is_answer)
                is_caseds.append(is_cased)
                ners.append(ner)
                poss.append(pos)
                targets.append(tokenized_questions)
    end_time = time.time()

    inputs = np.array(inputs)
    is_answers = np.expand_dims(is_answers, axis=-1)
    is_caseds = np.expand_dims(is_caseds, axis=1)
    ners = np.expand_dims(ners, axis=-1)
    poss = np.expand_dims(poss, axis=-1)
    if lower:
        features = np.concatenate((is_answers, is_caseds, ners, poss), axis=-1)
    else:
        features = np.concatenate((is_answers, ners, poss), axis=1)
    targets = np.array(targets)

    print(f'Not found answers: {deleted}')
    print(f'Execution time: {end_time-start_time}')
    return inputs, features, targets


def shuffle(*args, seed=42):
    indices = np.arange(args[0].shape[0])
    np.random.seed(seed)
    np.random.shuffle(indices)
    result = []
    for i in range(len(args)):
        result.append(args[i][indices])
    return tuple(result)


def split_by_k(*args, k=0.9):
    assert 0 < k < 1, 'k must be between 0 and 1!'
    group_1 = []
    group_2 = []
    split_idx = int(k * len(args[0]))
    for i in range(len(args)):
        group_1.append(args[i][:split_idx])
        group_2.append(args[i][split_idx:])
    group_1.extend(group_2)
    return tuple(group_1)
