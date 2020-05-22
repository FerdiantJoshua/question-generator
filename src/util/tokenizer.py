import re
from unidecode import unidecode

NON_ASCII_REGEX = re.compile(r"[^\x00-\x7F\u2013]")
# Complete punctuation from string.punctuation: !"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~
PUNCTUATIONS = '!"#$%&\'()*+/;<=>@?[\\]^_`{|}~'
PUNCUATIONS_REGEX = re.compile(r"([%s])" % PUNCTUATIONS)
REAL_SEPARATOR_REGEX = re.compile(r"(([\.,:][^a-zA-Z0-9])|([\.,:]$))")


def unicode_to_ascii(s):
    return unidecode(s)


def normalize_string(s, lower=False):
    s = unicode_to_ascii(s)
    if lower:
        s = s.lower()
    return s


def tokenize(s):
    s = re.sub(PUNCUATIONS_REGEX, r" \1 ", s)
    s = re.sub(REAL_SEPARATOR_REGEX, r" \1", s)
    s = s.split()
    return s
