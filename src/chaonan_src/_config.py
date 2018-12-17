#!/usr/bin/env python

import os


def get_dir_name():
    return os.path.dirname(os.path.abspath(__file__))


def relative_path(path):
    return os.path.join(get_dir_name(), path)


freq_words_path = relative_path('../../results/chaonan99/freq_words.txt')

old_result_path = relative_path('../../results_old')


def main():
    pass


if __name__ == '__main__':
    main()