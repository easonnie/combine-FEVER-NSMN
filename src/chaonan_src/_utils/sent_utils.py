

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