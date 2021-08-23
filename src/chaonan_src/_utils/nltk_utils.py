#!/usr/bin/env python
"""NLTK utils
"""

import pickle
import nltk
from nltk.corpus import brown, gutenberg


__all__ = ['get_nltk_freq_words']
__author__ = ['chaonan99']


def get_nltk_freq_words():
    """Use Brown corpus frequent words
    More corpora: http://www.nltk.org/book/ch02.html
    """
    freq_dict = nltk.FreqDist(brown.words())

    for fileid in gutenberg.fileids():
        freq_dict.update(nltk.FreqDist(gutenberg.words(fileid)))

    freq_words = [k for k, v in freq_dict.items() if v > 10]
    return freq_words, freq_dict


def main():
    freq_words, freq_dict = get_nltk_freq_words()
    from IPython import embed; embed(); import os; os._exit(1)
    pickle.dump(freq_dict, open("../../../results/chaonan99/freq_dict.pkl", "wb"))
    save_path = "../../../results/chaonan99/freq_words.txt"
    with open(save_path, 'w') as f:
        f.write('\n'.join(freq_words))

if __name__ == '__main__':
    main()