#!/usr/bin/env python3
# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Rank documents with TF-IDF scores"""
import argparse
import logging
import numpy as np
import scipy.sparse as sp

from multiprocessing.pool import ThreadPool
from functools import partial

import math
from drqa_yixin.retriever import utils

from sentence_retrieval.build_tfidf_yixin import get_count_matrix, get_tfidf_matrix, get_doc_freqs
from drqa_yixin.tokenizers import tokenizer


logger = logging.getLogger(__name__)


class TfidfDocRanker(object):
    """Loads a pre-weighted inverted index of token/document terms.
    Scores new queries by taking sparse dot products.
    """

    def __init__(self, doc_dict, doc_freqs, matrix, tfidf_path=None, num_gram=2,
                 hash_size=int(math.pow(2, 24)), strict=True):
        """
        Args:
            tfidf_path: path to saved model file
            strict: fail on empty queries or continue (and return empty result)
        """
        # Load from disk
        # tfidf_path = tfidf_path or DEFAULTS['tfidf_path']
        logger.info('Loading %s' % tfidf_path)
        # matrix, metadata = utils.load_sparse_csr(tfidf_path)
        self.doc_mat = matrix
        self.ngrams = num_gram
        self.hash_size = hash_size
        # self.tokenizer = tokenizers.get_class(metadata['tokenizer'])()
        self.doc_freqs = doc_freqs
        self.doc_dict = doc_dict
        self.num_docs = len(self.doc_dict[0])
        self.strict = strict

    def get_doc_index(self, doc_id):
        """Convert doc_id --> doc_index"""
        return self.doc_dict[0][doc_id]

    def get_doc_id(self, doc_index):
        """Convert doc_index --> doc_id"""
        return self.doc_dict[1][doc_index]

    def closest_docs(self, query, k=1):
        """Closest docs by dot product between query and documents
        in tfidf weighted word vector space.
        """
        spvec = self.text2spvec(query)
        res = spvec * self.doc_mat

        if len(res.data) <= k:
            o_sort = np.argsort(-res.data)
        else:
            o = np.argpartition(-res.data, k)[0:k]
            o_sort = o[np.argsort(-res.data[o])]

        doc_scores = res.data[o_sort]
        doc_ids = [self.get_doc_id(i) for i in res.indices[o_sort]]
        return doc_ids, doc_scores

    def batch_closest_docs(self, queries, k=1, num_workers=None):
        """Process a batch of closest_docs requests multithreaded.
        Note: we can use plain threads here as scipy is outside of the GIL.
        """
        with ThreadPool(num_workers) as threads:
            closest_docs = partial(self.closest_docs, k=k)
            results = threads.map(closest_docs, queries)
        return results

    def parse(self, query):
        """Parse the query into tokens (either ngrams or tokens)."""
        # tokens = self.tokenizer.tokenize(query)
        tokens = tokenizer.Tokens([(w,) for w in query.split(' ')], [])
        return tokens.ngrams(n=self.ngrams, uncased=True,
                             filter_fn=utils.filter_ngram)

    def text2spvec(self, query):
        """Create a sparse tfidf-weighted word vector from query.

        tfidf = log(tf + 1) * log((N - Nt + 0.5) / (Nt + 0.5))
        """
        # Get hashed ngrams
        words = self.parse(utils.normalize(query))
        wids = [utils.hash(w, self.hash_size) for w in words]

        if len(wids) == 0:
            if self.strict:
                raise RuntimeError('No valid word in: %s' % query)
            else:
                logger.warning('No valid word in: %s' % query)
                return sp.csr_matrix((1, self.hash_size))

        # Count TF
        wids_unique, wids_counts = np.unique(wids, return_counts=True)
        tfs = np.log1p(wids_counts)

        # Count IDF
        Ns = self.doc_freqs[wids_unique]
        idfs = np.log((self.num_docs - Ns + 0.5) / (Ns + 0.5))
        idfs[idfs < 0] = 0

        # TF-IDF
        data = np.multiply(tfs, idfs)

        # One row, sparse csr matrix
        indptr = np.array([0, len(wids_unique)])
        spvec = sp.csr_matrix(
            (data, wids_unique, indptr), shape=(1, self.hash_size)
        )

        return spvec


class OnlineTfidfDocRanker(TfidfDocRanker):
    """Loads a pre-weighted inverted index of token/document terms.
    Scores new queries by taking sparse dot products.
    """

    def __init__(self, args, hash_size, num_gram, lines,
                 freqs=None, strict=True):
        """
        Args:
            tfidf_path: path to saved model file
            strict: fail on empty queries or continue (and return empty result)
        """
        # Load from disk
        logging.info('Counting words...')
        count_matrix, doc_dict = get_count_matrix(
            args, 'memory', {'lines': lines}
        )

        logger.info('Making tfidf vectors...')
        tfidf = get_tfidf_matrix(count_matrix)

        if freqs is None:
            logger.info('Getting word-doc frequencies...')
            freqs = get_doc_freqs(count_matrix)

        metadata = {
            'doc_freqs': freqs,
            'hash_size': hash_size,
            'ngram': num_gram,
            'doc_dict': doc_dict
        }

        self.doc_mat = tfidf
        self.ngrams = metadata['ngram']
        self.hash_size = metadata['hash_size']
        self.doc_freqs = metadata['doc_freqs'].squeeze()
        self.doc_dict = metadata['doc_dict']
        self.num_docs = len(self.doc_dict[0])
        self.strict = strict


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ngram', type=int, default=2,
                        help=('Use up to N-size n-grams '
                              '(e.g. 2 = unigrams + bigrams)'))
    parser.add_argument('--hash-size', type=int, default=int(math.pow(2, 24)),
                        help='Number of buckets to use for hashing ngrams')
    # parser.add_argument('--tokenizer', type=str, default='simple',
    #                     help=("String option specifying tokenizer type to use "
    #                           "(e.g. 'corenlp')"))
    parser.add_argument('--num-workers', type=int, default=None,
                        help='Number of CPU processes (for tokenizing, etc)')
    args = parser.parse_args()

    # ranker = OnlineTfidfDocRanker(args, args.hash_size, args.ngram,
    #                               ["It said that because the ads had been shared widely on social media , they therefore would have been seen by a large number of people , including some children who did not actively follow the Poundland accounts .",
    #                                "The ads were irresponsible and likely to cause serious or widespread offence , said the ASA , which also revealed it had received 85 complaints about the Poundland campaign .",
    #                                "Thousands of women who work in Tesco stores could receive back pay totalling £20,000 if the legal challenge demanding parity with men who work in the company's warehouses is successful.",
    #                                "Lawyers say hourly-paid female store staff earn less than men even though the value of the work is comparable.",
    #                                "Paula Lee, of Leigh Day solicitors told the BBC it was time for Tesco to tackle the problem of equal pay for work of equal worth.",
    #                                "Her firm has been contacted by more than 1,000 Tesco staff and will this week take the initial legal steps for 100 of them.",
    #                                "The most common rate for women is £8 an hour whereas for men the hourly rate can be as high as £11 an hour, she added.",
    #                                "Since 1984 workers doing jobs that require comparable skills, have similar levels of responsibility and are of comparable worth to the employer, should also be rewarded equally, according to the law.",
    #                                "Thus if you are a cleaner, lugging mops and buckets up and down staircases, you may have a case for being paid the same as co-workers collecting rubbish bins.",
    #                                "It doesn't matter whether the cleaner or the shop floor worker is male or female, they may still have a case to see their pay upped to match colleagues doing other jobs."
    #                                "But in practice many of the poorer paid jobs have been done by women."])

    # ranker = OnlineTfidfDocRanker(args, args.hash_size, args.ngram,
    #                               ["A A A B",
    #                                "B C A C",
    #                                "B C A C",
    #                                "B B A C",
    #                                "D D F A",
    #                                "D C F A",
    #                                "D E A C"])
    #
    # # print(ranker.closest_docs("female met the employer and had the responsibility to collect rubbish bins",k=5))
    # print(ranker.closest_docs("A A B B", k=5))

    ranker = OnlineTfidfDocRanker(args, args.hash_size, args.ngram,
                                  [
                                    "It said that because the ads had been shared widely on social media , B Lawyers say hourly-paid female store",
                                   "the the the The ads were irresponsible and likely to cause serious or widespread",
                                   "Thousands of women who work in Tesco stores could receive back pay",
                                   "Paula Lee , of Leigh Day solicitors told the BBC it was time for",
                                   "Her firm has been contacted by more than 1000 Tesco staff and will this responsibility and are of comparable worth to the employer",
                                   "Since 1984 workers doing jobs that require comparable skills , have",
                                   "It does n't matter whether the cleaner or the shop floor"])

    print(ranker.closest_docs("the the the female met the employer and had the responsibility to collect rubbish bins", k=5))