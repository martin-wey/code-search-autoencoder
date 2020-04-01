import re
import spacy
import numpy as np
from spacy.tokenizer import Tokenizer
from spacy.tokens import Doc
from gensim.parsing.preprocessing import remove_stopwords, preprocess_string

# NLP utils functions

camel_case_infix_re = re.compile(r'''[A-Z](?:[a-z]+|[A-Z]*(?=[A-Z]|$))''')
underscore_infix_re = re.compile(r'''_''')


def camel_case_tokenizer(nlp):
    return Tokenizer(nlp.vocab, infix_finditer=camel_case_infix_re.finditer)


def underscore_tokenizer(nlp):
    return Tokenizer(nlp.vocab, infix_finditer=underscore_infix_re.finditer)


def initialize_nlp(corpus):
    """Initialize spacy with a given corpus name"""
    return spacy.load(corpus, disable=["tagger", "parser", "ner"])


def initialize_tokenizer(nlp, type):
    """Initialize spacy's tokenizer according to the type provided"""
    if type == 'underscore':
        nlp.tokenizer = underscore_tokenizer(nlp)
    elif type == 'camel':
        nlp.tokenizer = camel_case_tokenizer(nlp)
    return nlp


rm_list = ['_', 'clf', 'self', '', None]
def tokenize_list(nlp, data, lower=False):
    data_tokenized = []
    for item in data:
        data_tokenized.append(
            ' '.join(
                token.text.translate({ord('_'): None}) for token in nlp.tokenizer(item)
                if token.text not in rm_list
             )
        )
    if lower:
        return map(lambda x: x.lower(), data_tokenized)
    return data_tokenized


def strings_to_list(data):
    tkn_list = []
    for item in data:
        for x in item.split():
            tkn_list.append(x)
    return tkn_list


def clean_text_data(stemmer, data):
    data = ' '.join(data)
    data = remove_stopwords(data)
    data = stemmer.stem_sentence(data)
    data = preprocess_string(data)
    return data
