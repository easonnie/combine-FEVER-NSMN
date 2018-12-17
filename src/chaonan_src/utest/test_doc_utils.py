#!/usr/bin/env python


from chaonan_src._utils.doc_utils import get_default_tfidf_ranker_args
from chaonan_src._utils.doc_utils import DocIDTokenizer


def test_get_default_tfidf_ranker_args():
    args = get_default_tfidf_ranker_args()
    assert has_attr(args, 'ngram')
    assert has_attr(args, 'hash_size')
    assert has_attr(args, 'num_workers')

def test_doc_id_tokenizer():
    DocIDTokenizer()
