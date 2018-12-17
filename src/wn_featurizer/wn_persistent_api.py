import json

import nltk
from nltk.corpus import wordnet as wn
from nltk.corpus import wordnet_ic
import config

from functools import reduce
import numpy as np

from sqlitedict import SqliteDict

nltk_path = '/playpen/home/yicheng/data/nltk_data'
nltk.data.path.append(nltk_path)

inf = float('inf')
ic = wordnet_ic.ic('ic-brown.dat')

stemmer = nltk.SnowballStemmer('english')

LEVEL_INF = 17  # diameter of wordnet

from functools import partial

from tqdm import tqdm


def memoize(f):  # python tricks ahead
    """ Memoization decorator for a function taking a single argument """

    class memodict(dict):
        def __missing__(self, key):
            ret = self[key] = f(key)
            return ret

    return memodict().__getitem__


def memoize_reflexive(f):
    class memodict(dict):
        def __missing__(self, key):
            a, b = key
            ret = self[(a, b)] = self[(b, a)] = f(a, b)
            return ret

    return lambda a, b: memodict().__getitem__((a, b))


def memoize_two_args(f):
    class memodict(dict):
        def __missing__(self, key):
            a, b = key
            ret = self[(a, b)] = f(a, b)
            return ret

    return lambda a, b: memodict().__getitem__((a, b))


def memoize_three_args(f):
    class memodict(dict):
        def __missing__(self, key):
            a, b, c = key
            ret = self[(a, b, c)] = f(a, b, c)
            return ret

    return lambda a, b, c: memodict().__getitem__((a, b, c))


def convert_to_wn_pos(pos):
    if pos.startswith("J"):
        return wn.ADJ
    elif pos.startswith("V"):
        return wn.VERB
    elif pos.startswith("N"):
        return wn.NOUN
    elif pos.startswith("R"):
        return wn.ADV
    else:
        return ""


def get_wn_pos(parse):
    # print(parse)
    base_parse = [s.rstrip(" ").rstrip(")")
                  for s in parse.split("(") if ")" in s]
    # print(base_parse)

    pos = [pair.split(" ")[0] for pair in base_parse]
    # print(pos)
    # input()
    wn_pos = [convert_to_wn_pos(p) for p in pos]
    # print(wn_pos)
    # input()
    return wn_pos


def wn_pos_tag(sent):
    sent_with_pos = nltk.pos_tag(sent)

    output = [(w, convert_to_wn_pos(p)) for (w, p) in sent_with_pos]

    return output


@memoize
def get_stem(word):
    return stemmer.stem(word)


@memoize
def get_hyponyms(synset):
    if synset is None:
        return set([])
    hyponyms = set([])
    hyponyms |= set(synset.hyponyms())

    for h in synset.hyponyms():
        hyponyms |= set(get_hyponyms(h))

    return hyponyms


@memoize
def get_hypernyms(synset):
    if synset is None:
        return set([])
    hypernyms = set([])
    hypernyms |= set(synset.hypernyms())

    for h in synset.hypernyms():
        hypernyms |= set(get_hypernyms(h))

    return hypernyms


@memoize
def get_hyponym_stems(item):
    word, pos = item
    stems = set([])

    for synset in wn.synsets(word, pos=pos):
        for lemma in synset.lemma_names():
            for syn_lemma in wn.synsets(lemma, pos=pos):
                try:
                    syn_lemma_hypos = get_hyponyms(syn_lemma)
                except RecursionError:
                    # print(syn_lemma)
                    continue

                for nym in syn_lemma_hypos:
                    stems |= set(nym.lemma_names())
                    stems |= set([get_stem(ln) for ln in nym.lemma_names()])

    return (stems - get_stem_set(word))


def get_hyponym_stems_p(item, hyponym_stems_dict):
    word, pos = item
    item_key = word + '<(-.-)>' + pos

    if item_key in hyponym_stems_dict:
        return hyponym_stems_dict[item_key]

    stems = set([])
    for synset in wn.synsets(word, pos=pos):
        for lemma in synset.lemma_names():
            for syn_lemma in wn.synsets(lemma, pos=pos):
                try:
                    syn_lemma_hypos = get_hyponyms(syn_lemma)
                except RecursionError:
                    # print(syn_lemma)
                    continue

                for nym in syn_lemma_hypos:
                    stems |= set(nym.lemma_names())
                    stems |= set([get_stem(ln) for ln in nym.lemma_names()])

    result_set = (stems - get_stem_set(word))
    hyponym_stems_dict[item_key] = result_set
    return result_set


def get_hypernym_stems_p(item, hypernym_stems_dict):
    word, pos = item
    item_key = word + '<(-.-)>' + pos
    if item_key in hypernym_stems_dict:
        return hypernym_stems_dict[item_key]

    stems = set([])

    for synset in wn.synsets(word, pos=pos):
        for lemma in synset.lemma_names():
            for syn_lemma in wn.synsets(lemma, pos=pos):
                try:
                    syn_lemma_hypers = get_hypernyms(syn_lemma)
                except RecursionError:
                    # print(syn_lemma)
                    continue

                for nym in syn_lemma_hypers:
                    stems |= set(nym.lemma_names())
                    stems |= set([get_stem(ln) for ln in nym.lemma_names()])

    result_set = (stems - get_stem_set(word))
    hypernym_stems_dict[item_key] = result_set
    return result_set


@memoize
def get_hypernym_stems(item):
    word, pos = item
    stems = set([])

    for synset in wn.synsets(word, pos=pos):
        for lemma in synset.lemma_names():
            for syn_lemma in wn.synsets(lemma, pos=pos):
                try:
                    syn_lemma_hypers = get_hypernyms(syn_lemma)
                except RecursionError:
                    # print(syn_lemma)
                    continue

                for nym in syn_lemma_hypers:
                    stems |= set(nym.lemma_names())
                    stems |= set([get_stem(ln) for ln in nym.lemma_names()])

    return (stems - get_stem_set(word))


hyper = lambda s: s.hypernyms()
hypo = lambda s: s.hyponyms()

# @memoize_two_args
# def get_hyper_from_lv(item, level):
#     word, pos = item
#     stems = set([])
#
#     for synset in wn.synsets(word, pos=pos):
#         for lemma in synset.lemma_names():
#             for syn_lemma in wn.synsets(lemma, pos=pos):
#                 for h in syn_lemma.closure(hyper, LEVEL_INF):
#                     for h_lemma in h.lemma_names():
#                         stems.add(h_lemma)
#                         stems.add(get_stem(h_lemma))
#
#     return (stems - get_hyper_up_to_lv(word, level))
db_name_list = [
    'hypernym_stems_dict', 'hyponym_stems_dict', 'hyper_lvl_dict',
    'hypo_lvl_dict', 'ant_dict', 'em_dict', 'em_lemmas_dict']


def persistence_load(db_path=config.WN_FEATURE_CACHE_PATH):
    p_dict = {
        'hypernym_stems_dict': dict(),
        'hyponym_stems_dict': dict(),
        'hyper_lvl_dict': dict(),
        'hypo_lvl_dict': dict(),
        'ant_dict': dict(),
        'em_lemmas_dict': dict(),
    }

    # if em_dict:
    #     p_dict['em_dict'] = dict()

    for dict_name in p_dict.keys():
        print("Loading Persistent WN Feature Dict:", dict_name)
        if dict_name != 'em_dict':
            in_db_dict = SqliteDict(str(db_path / dict_name),
                                    autocommit=False, tablename='the_table',
                                    flag='c')
            for key, v in tqdm(in_db_dict.items()):
                p_dict[dict_name][key] = v
            in_db_dict.close()
        elif dict_name == 'em_dict':
            in_db_dict = SqliteDict(str(db_path / dict_name),
                                    autocommit=False, tablename='the_table', flag='c')
            for key, v in tqdm(in_db_dict.items()):
                p_dict[dict_name][key] = v
            in_db_dict.close()

    return p_dict


def persistence_update(p_dict, db_path=config.WN_FEATURE_CACHE_PATH):
    for dict_name in p_dict.keys():
        print("Updating Persistent WN Feature Dict:", dict_name)
        if dict_name != 'em_dict':
            in_db_dict = SqliteDict(str(db_path / dict_name),
                                    autocommit=False, tablename='the_table',
                                    flag='c')
            for key, v in tqdm(p_dict[dict_name].items()):
                if key not in in_db_dict:
                    in_db_dict[key] = v
            in_db_dict.commit()
            in_db_dict.close()

        elif dict_name == 'em_dict':
            in_db_dict = SqliteDict(str(db_path / dict_name),
                                    autocommit=False, tablename='the_table', flag='c')
            for key, v in tqdm(p_dict[dict_name].items()):
                if key not in in_db_dict:
                    in_db_dict[key] = v
            in_db_dict.commit()
            in_db_dict.close()


@memoize_two_args
def get_hyper_up_to_lv(item, level):
    word, pos = item
    stems = set([])

    for synset in wn.synsets(word, pos=pos):
        for lemma in synset.lemma_names():
            for syn_lemma in wn.synsets(lemma, pos=pos):
                for h in syn_lemma.closure(hyper, level):
                    for h_lemma in h.lemma_names():
                        stems.add(h_lemma)
                        stems.add(get_stem(h_lemma))

    return stems


def get_hyper_up_to_lv_p(item, level, hyper_lvl_dict):
    word, pos = item
    stems = set([])

    item_key = word + '<(-.-)>' + pos + '#' + str(level)
    if item_key in hyper_lvl_dict:
        return hyper_lvl_dict[item_key]

    for synset in wn.synsets(word, pos=pos):
        for lemma in synset.lemma_names():
            for syn_lemma in wn.synsets(lemma, pos=pos):
                for h in syn_lemma.closure(hyper, level):
                    for h_lemma in h.lemma_names():
                        stems.add(h_lemma)
                        stems.add(get_stem(h_lemma))

    hyper_lvl_dict[item_key] = stems

    return stems


# @memoize_two_args
# def get_hypo_from_lv(item, level):
#     word, pos = item
#     stems = set([])
#
#     for synset in wn.synsets(word, pos=pos):
#         for lemma in synset.lemma_names():
#             for syn_lemma in wn.synsets(lemma, pos=pos):
#                 for h in syn_lemma.closure(hypo, LEVEL_INF):
#                     for h_lemma in h.lemma_names():
#                         stems.add(h_lemma)
#                         stems.add(get_stem(h_lemma))
#
#     return (stems - get_hypo_up_to_lv(word, level))


@memoize_two_args
def get_hypo_up_to_lv(item, level):
    word, pos = item
    stems = set([])

    for synset in wn.synsets(word, pos=pos):
        for lemma in synset.lemma_names():
            for syn_lemma in wn.synsets(lemma, pos=pos):
                for h in syn_lemma.closure(hypo, level):
                    for h_lemma in h.lemma_names():
                        stems.add(h_lemma)
                        stems.add(get_stem(h_lemma))

    return stems


def get_hypo_up_to_lv_p(item, level, hypo_lvl_dict):
    word, pos = item
    stems = set([])

    item_key = word + '<(-.-)>' + pos + '#' + str(level)
    if item_key in hypo_lvl_dict:
        return hypo_lvl_dict[item_key]

    for synset in wn.synsets(word, pos=pos):
        for lemma in synset.lemma_names():
            for syn_lemma in wn.synsets(lemma, pos=pos):
                for h in syn_lemma.closure(hypo, level):
                    for h_lemma in h.lemma_names():
                        stems.add(h_lemma)
                        stems.add(get_stem(h_lemma))

    hypo_lvl_dict[item_key] = stems

    return stems


@memoize
def get_stem_set(word):
    stems = set([word])

    stems.add(get_stem(word))
    return stems


@memoize
def get_antonym_stems(item):
    word, pos = item
    stems = set([])

    for synset in wn.synsets(word, pos=pos):
        for lemma in synset.lemma_names():
            for syn_lemma in wn.synsets(lemma, pos=pos):
                for l in syn_lemma.lemmas():
                    for antonym in l.antonyms():
                        stems.add(antonym.name())
                        stems.add(get_stem(antonym.name()))

    return stems


def get_antonym_stems_p(item, ant_dict):
    word, pos = item
    stems = set([])
    item_key = word + '<(-.-)>' + pos
    if item_key in ant_dict:
        return ant_dict[item_key]

    for synset in wn.synsets(word, pos=pos):
        for lemma in synset.lemma_names():
            for syn_lemma in wn.synsets(lemma, pos=pos):
                for l in syn_lemma.lemmas():
                    for antonym in l.antonyms():
                        stems.add(antonym.name())
                        stems.add(get_stem(antonym.name()))

    ant_dict[item_key] = stems

    return stems


@memoize_reflexive
def is_exact_match(token1, token2):
    # token1 = token1.lower()
    # token2 = token2.lower()

    if token1 == token2:
        return True

    token1_stem = get_stem(token1)

    for synsets in wn.synsets(token2):
        for lemma in synsets.lemma_names():
            if token1_stem == get_stem(lemma):
                return True

    token2_stem = get_stem(token2)

    for synsets in wn.synsets(token1):
        for lemma in synsets.lemma_names():
            if token2_stem == get_stem(lemma):
                return True

    if token1 == "n't" and token2 == "not":
        return True
    elif token1 == "not" and token2 == "n't":
        return True
    elif token1_stem == get_stem(token2):
        return True
    return False


def get_lemma_stem_dict(token, em_match_dict):
    token = token.lower()

    if token in em_match_dict:
        return em_match_dict[token]

    match_set = set()
    # match_set = {token}
    # match_set |= {get_stem(token)}

    for synsets in wn.synsets(token):
        for lemma in synsets.lemma_names():
            match_set.add(lemma)

    if token == "n't":
        match_set |= {"not"}

    em_match_dict[token] = match_set

    return match_set


def is_exact_match_p(token1, token2, em_dict):
    token1 = token1.lower()
    token2 = token2.lower()

    if token1 == token2:
        return True

    item_key_f = token1 + '###' + token2
    item_key_b = token2 + '###' + token1

    if (item_key_f in em_dict) or (item_key_b in em_dict):
        return em_dict[item_key_f]

    token1_stem = get_stem(token1)

    for synsets in wn.synsets(token2):
        for lemma in synsets.lemma_names():
            if token1_stem == get_stem(lemma):
                em_dict[item_key_f] = True
                em_dict[item_key_b] = True
                return True

    token2_stem = get_stem(token2)

    for synsets in wn.synsets(token1):
        for lemma in synsets.lemma_names():
            if token2_stem == get_stem(lemma):
                em_dict[item_key_f] = True
                em_dict[item_key_b] = True
                return True

    if token1 == "n't" and token2 == "not":
        em_dict[item_key_f] = True
        em_dict[item_key_b] = True
        return True
    elif token1 == "not" and token2 == "n't":
        em_dict[item_key_f] = True
        em_dict[item_key_b] = True
        return True
    elif token1_stem == get_stem(token2):
        em_dict[item_key_f] = True
        em_dict[item_key_b] = True
        return True

    em_dict[item_key_f] = False
    em_dict[item_key_b] = False
    return False


def compute_wn_features(s1, s2):
    sent1 = wn_pos_tag(s1)
    sent2 = wn_pos_tag(s2)

    s1_lemmas = reduce(lambda x, y: x | y,
                       map(lambda item: get_stem_set(item[0]), sent1))
    s1_antonym_lemmas = reduce(lambda x, y: x | y,
                               map(get_antonym_stems, sent1))
    s1_hypernyms = reduce(lambda x, y: x.union(y),
                          map(get_hypernym_stems, sent1))
    s1_hyponyms = reduce(lambda x, y: x.union(y),
                         map(get_hyponym_stems, sent1))
    s1_hyper_up_to_1 = reduce(lambda x, y: x | y,
                              map(lambda w: get_hyper_up_to_lv(w, 1), sent1))
    s1_hyper_up_to_2 = reduce(lambda x, y: x | y,
                              map(lambda w: get_hyper_up_to_lv(w, 2), sent1)) - \
                       s1_hyper_up_to_1
    s1_hypo_up_to_1 = reduce(lambda x, y: x | y,
                             map(lambda w: get_hypo_up_to_lv(w, 1), sent1))
    s1_hypo_up_to_2 = reduce(lambda x, y: x | y,
                             map(lambda w: get_hypo_up_to_lv(w, 2), sent1)) - \
                      s1_hypo_up_to_1
    s1_hypernyms -= (s1_hyper_up_to_1 | s1_hyper_up_to_2)
    s1_hyponyms -= (s1_hypo_up_to_1 | s1_hypo_up_to_2)

    s2_lemmas = reduce(lambda x, y: x | y, map(lambda item: get_stem_set(item[0]), sent2))
    s2_antonym_lemmas = reduce(lambda x, y: x | y,
                               map(get_antonym_stems, sent2))
    s2_hypernyms = reduce(lambda x, y: x.union(y),
                          map(get_hypernym_stems, sent2))
    s2_hyponyms = reduce(lambda x, y: x.union(y),
                         map(get_hyponym_stems, sent2))
    s2_hyper_up_to_1 = reduce(lambda x, y: x | y,
                              map(lambda w: get_hyper_up_to_lv(w, 1), sent2))
    s2_hyper_up_to_2 = reduce(lambda x, y: x | y,
                              map(lambda w: get_hyper_up_to_lv(w, 2), sent2)) - \
                       s2_hyper_up_to_1
    s2_hypo_up_to_1 = reduce(lambda x, y: x | y,
                             map(lambda w: get_hypo_up_to_lv(w, 1), sent2))
    s2_hypo_up_to_2 = reduce(lambda x, y: x | y,
                             map(lambda w: get_hypo_up_to_lv(w, 2), sent2)) - \
                      s2_hypo_up_to_1
    s2_hypernyms -= (s2_hyper_up_to_1 | s2_hyper_up_to_2)
    s2_hyponyms -= (s2_hypo_up_to_1 | s2_hypo_up_to_2)

    s1_em = [[0, 0, 0] for _ in range(len(sent1))]
    s1_ant = [[0, 0, 0] for _ in range(len(sent1))]
    s1_anc = [[0, 0, 0] for _ in range(len(sent1))]
    s1_desc = [[0, 0, 0] for _ in range(len(sent1))]
    s1_hypo_1 = [[0, 0, 0] for _ in range(len(sent1))]
    s1_hyper_1 = [[0, 0, 0] for _ in range(len(sent1))]
    s1_hypo_2 = [[0, 0, 0] for _ in range(len(sent1))]
    s1_hyper_2 = [[0, 0, 0] for _ in range(len(sent1))]
    s1_hypo_3plus = [[0, 0, 0] for _ in range(len(sent1))]
    s1_hyper_3plus = [[0, 0, 0] for _ in range(len(sent1))]
    # s1_hypo_dist = [[0, 0, 0] for _ in range(len(sent1))]
    # s1_hyper_dist = [[0, 0, 0] for _ in range(len(sent1))]

    s2_em = [[0, 0, 0] for _ in range(len(sent2))]
    s2_ant = [[0, 0, 0] for _ in range(len(sent2))]
    s2_anc = [[0, 0, 0] for _ in range(len(sent2))]
    s2_desc = [[0, 0, 0] for _ in range(len(sent2))]
    s2_hypo_1 = [[0, 0, 0] for _ in range(len(sent2))]
    s2_hyper_1 = [[0, 0, 0] for _ in range(len(sent2))]
    s2_hypo_2 = [[0, 0, 0] for _ in range(len(sent2))]
    s2_hyper_2 = [[0, 0, 0] for _ in range(len(sent2))]
    s2_hypo_3plus = [[0, 0, 0] for _ in range(len(sent2))]
    s2_hyper_3plus = [[0, 0, 0] for _ in range(len(sent2))]
    # s2_hypo_dist = [[0, 0, 0] for _ in range(len(sent2))]
    # s2_hyper_dist = [[0, 0, 0] for _ in range(len(sent2))]

    features = {}
    features['s1_em'] = s1_em
    features['s1_ant'] = s1_ant
    features['s1_anc'] = s1_anc
    features['s1_desc'] = s1_desc
    features['s1_hypo_1'] = s1_hypo_1
    features['s1_hyper_1'] = s1_hyper_1
    features['s1_hypo_2'] = s1_hypo_2
    features['s1_hyper_2'] = s1_hyper_2
    features['s1_hypo_3plus'] = s1_hypo_3plus
    features['s1_hyper_3plus'] = s1_hyper_3plus
    # features['s1_hypo_dist'] = s1_hypo_dist
    # features['s1_hyper_dist'] = s1_hyper_dist
    features['s2_em'] = s2_em
    features['s2_ant'] = s2_ant
    features['s2_anc'] = s2_anc
    features['s2_desc'] = s2_desc
    features['s2_hypo_1'] = s2_hypo_1
    features['s2_hyper_1'] = s2_hyper_1
    features['s2_hypo_2'] = s2_hypo_2
    features['s2_hyper_2'] = s2_hyper_2
    features['s2_hypo_3plus'] = s2_hypo_3plus
    features['s2_hyper_3plus'] = s2_hyper_3plus
    # features['s2_hypo_dist'] = s2_hypo_dist
    # features['s2_hyper_dist'] = s2_hyper_dist

    for i, (w, pos) in enumerate(sent1):
        stems = get_stem_set(w)

        # if len(stems & s2_lemmas) != 0:
        #     s1_em[i][0] = s1_em[i][2] = 1

        e_match = False
        for w2 in set(map(lambda x: x[0].lower(), sent2)):
            w_lower = w.lower()
            if is_exact_match(w_lower, w2):
                e_match = True
                break

        if e_match:
            s1_em[i][0] = s1_em[i][2] = 1

        if len(stems & s2_antonym_lemmas) != 0:
            s1_ant[i][0] = s1_ant[i][2] = 1

        if len(stems & s2_hypo_up_to_1) != 0:
            s1_anc[i][0] = s1_anc[i][2] = 1
            s1_hypo_1[i][0] = s1_hypo_1[i][2] = 1
            # s1_hypo_dist[i][0] = s1_hypo_dist[i][1] = 1
        elif len(stems & s2_hypo_up_to_2) != 0:
            s1_anc[i][0] = s1_anc[i][2] = 1
            s1_hypo_2[i][0] = s1_hypo_2[i][2] = 1
            # s1_hypo_dist[i][0] = s1_hypo_dist[i][1] = 2
        elif len(stems & s2_hyponyms) != 0:
            s1_anc[i][0] = s1_anc[i][2] = 1
            s1_hypo_3plus[i][0] = s1_hypo_3plus[i][2] = 1
            # s1_hypo_dist[i][0] = s1_hypo_dist[i][1] = 3

        if len(stems & s2_hyper_up_to_1) != 0:
            s1_desc[i][0] = s1_desc[i][2] = 1
            s1_hyper_1[i][0] = s1_hyper_1[i][2] = 1
            # s1_hyper_dist[i][0] = s1_hyper_dist[i][1] = 1
        elif len(stems & s2_hyper_up_to_2) != 0:
            s1_desc[i][0] = s1_desc[i][2] = 1
            s1_hyper_2[i][0] = s1_hyper_2[i][2] = 1
            # s1_hyper_dist[i][0] = s1_hyper_dist[i][1] = 2
        elif len(stems & s2_hypernyms) != 0:
            s1_desc[i][0] = s1_desc[i][2] = 1
            s1_hyper_3plus[i][0] = s1_hyper_3plus[i][2] = 1
            # s1_hyper_dist[i][0] = s1_hyper_dist[i][1] = 3

    for i, (w, pos) in enumerate(sent2):
        stems = get_stem_set(w)

        # if len(stems & s1_lemmas) != 0:
        #     s2_em[i][1] = s2_em[i][2] = 1

        e_match = False
        for (w1, pos) in sent1:
            if is_exact_match(w, w1):
                e_match = True
                break
        if e_match:
            s2_em[i][1] = s2_em[i][2] = 1

        if len(stems & s1_antonym_lemmas) != 0:
            s2_ant[i][1] = s2_ant[i][2] = 1

        if len(stems & s1_hypo_up_to_1) != 0:
            s2_anc[i][1] = s2_anc[i][2] = 1
            s2_hypo_1[i][1] = s2_hypo_1[i][2] = 1
            # s2_hypo_dist[i][1] = s2_hypo_dist[i][1] = 1
        elif len(stems & s1_hypo_up_to_2) != 0:
            s2_anc[i][1] = s2_anc[i][2] = 1
            s2_hypo_2[i][1] = s2_hypo_2[i][2] = 1
            # s2_hypo_dist[i][1] = s2_hypo_dist[i][1] = 2
        elif len(stems & s1_hyponyms) != 0:
            s2_anc[i][1] = s2_anc[i][2] = 1
            s2_hypo_3plus[i][1] = s2_hypo_3plus[i][2] = 1
            # s2_hypo_dist[i][1] = s2_hypo_dist[i][1] = 3

        if len(stems & s1_hyper_up_to_1) != 0:
            s2_desc[i][1] = s2_desc[i][2] = 1
            s2_hyper_1[i][1] = s2_hyper_1[i][2] = 1
            # s2_hyper_dist[i][1] = s2_hyper_dist[i][1] = 1
        elif len(stems & s1_hyper_up_to_2) != 0:
            s2_desc[i][1] = s2_desc[i][2] = 1
            s2_hyper_2[i][1] = s2_hyper_2[i][2] = 1
            # s2_hyper_dist[i][1] = s2_hyper_dist[i][1] = 2
        elif len(stems & s1_hypernyms) != 0:
            s2_desc[i][1] = s2_desc[i][2] = 1
            s2_hyper_3plus[i][1] = s2_hyper_3plus[i][2] = 1
            # s2_hyper_dist[i][1] = s2_hyper_dist[i][1] = 3

    return features


def compute_wn_features_p(s1, s2, p_dict):
    sent1 = wn_pos_tag(s1)
    sent2 = wn_pos_tag(s2)

    hypernym_stems_dict = p_dict['hypernym_stems_dict']
    get_hypernym_stems = partial(get_hypernym_stems_p, hypernym_stems_dict=hypernym_stems_dict)

    hyponym_stems_dict = p_dict['hyponym_stems_dict']
    get_hyponym_stems = partial(get_hyponym_stems_p, hyponym_stems_dict=hyponym_stems_dict)

    hyper_lvl_dict = p_dict['hyper_lvl_dict']
    get_hyper_up_to_lv = partial(get_hyper_up_to_lv_p, hyper_lvl_dict=hyper_lvl_dict)

    hypo_lvl_dict = p_dict['hypo_lvl_dict']
    get_hypo_up_to_lv = partial(get_hypo_up_to_lv_p, hypo_lvl_dict=hypo_lvl_dict)

    ant_dict = p_dict['ant_dict']
    get_antonym_stems = partial(get_antonym_stems_p, ant_dict=ant_dict)

    em_dict = p_dict['em_dict']
    # is_exact_match = partial(is_exact_match_p, em_dict=em_dict)

    s1_lemmas = reduce(lambda x, y: x | y,
                       map(lambda item: get_stem_set(item[0]), sent1))
    s1_antonym_lemmas = reduce(lambda x, y: x | y,
                               map(get_antonym_stems, sent1))
    s1_hypernyms = reduce(lambda x, y: x.union(y),
                          map(get_hypernym_stems, sent1))
    s1_hyponyms = reduce(lambda x, y: x.union(y),
                         map(get_hyponym_stems, sent1))
    s1_hyper_up_to_1 = reduce(lambda x, y: x | y,
                              map(lambda w: get_hyper_up_to_lv(item=w, level=1), sent1))
    s1_hyper_up_to_2 = reduce(lambda x, y: x | y,
                              map(lambda w: get_hyper_up_to_lv(item=w, level=2), sent1)) - \
                       s1_hyper_up_to_1
    s1_hypo_up_to_1 = reduce(lambda x, y: x | y,
                             map(lambda w: get_hypo_up_to_lv(item=w, level=1), sent1))
    s1_hypo_up_to_2 = reduce(lambda x, y: x | y,
                             map(lambda w: get_hypo_up_to_lv(item=w, level=2), sent1)) - \
                      s1_hypo_up_to_1
    s1_hypernyms -= (s1_hyper_up_to_1 | s1_hyper_up_to_2)
    s1_hyponyms -= (s1_hypo_up_to_1 | s1_hypo_up_to_2)

    s2_lemmas = reduce(lambda x, y: x | y, map(lambda item: get_stem_set(item[0]), sent2))

    s2_antonym_lemmas = reduce(lambda x, y: x | y,
                               map(get_antonym_stems, sent2))
    s2_hypernyms = reduce(lambda x, y: x.union(y),
                          map(get_hypernym_stems, sent2))
    s2_hyponyms = reduce(lambda x, y: x.union(y),
                         map(get_hyponym_stems, sent2))
    s2_hyper_up_to_1 = reduce(lambda x, y: x | y,
                              map(lambda w: get_hyper_up_to_lv(item=w, level=1), sent2))
    s2_hyper_up_to_2 = reduce(lambda x, y: x | y,
                              map(lambda w: get_hyper_up_to_lv(item=w, level=2), sent2)) - \
                       s2_hyper_up_to_1
    s2_hypo_up_to_1 = reduce(lambda x, y: x | y,
                             map(lambda w: get_hypo_up_to_lv(item=w, level=1), sent2))
    s2_hypo_up_to_2 = reduce(lambda x, y: x | y,
                             map(lambda w: get_hypo_up_to_lv(item=w, level=2), sent2)) - \
                      s2_hypo_up_to_1
    s2_hypernyms -= (s2_hyper_up_to_1 | s2_hyper_up_to_2)
    s2_hyponyms -= (s2_hypo_up_to_1 | s2_hypo_up_to_2)

    s1_em = [[0, 0, 0] for _ in range(len(sent1))]
    s1_ant = [[0, 0, 0] for _ in range(len(sent1))]
    s1_anc = [[0, 0, 0] for _ in range(len(sent1))]
    s1_desc = [[0, 0, 0] for _ in range(len(sent1))]
    s1_hypo_1 = [[0, 0, 0] for _ in range(len(sent1))]
    s1_hyper_1 = [[0, 0, 0] for _ in range(len(sent1))]
    s1_hypo_2 = [[0, 0, 0] for _ in range(len(sent1))]
    s1_hyper_2 = [[0, 0, 0] for _ in range(len(sent1))]
    s1_hypo_3plus = [[0, 0, 0] for _ in range(len(sent1))]
    s1_hyper_3plus = [[0, 0, 0] for _ in range(len(sent1))]
    # s1_hypo_dist = [[0, 0, 0] for _ in range(len(sent1))]
    # s1_hyper_dist = [[0, 0, 0] for _ in range(len(sent1))]

    s2_em = [[0, 0, 0] for _ in range(len(sent2))]
    s2_ant = [[0, 0, 0] for _ in range(len(sent2))]
    s2_anc = [[0, 0, 0] for _ in range(len(sent2))]
    s2_desc = [[0, 0, 0] for _ in range(len(sent2))]
    s2_hypo_1 = [[0, 0, 0] for _ in range(len(sent2))]
    s2_hyper_1 = [[0, 0, 0] for _ in range(len(sent2))]
    s2_hypo_2 = [[0, 0, 0] for _ in range(len(sent2))]
    s2_hyper_2 = [[0, 0, 0] for _ in range(len(sent2))]
    s2_hypo_3plus = [[0, 0, 0] for _ in range(len(sent2))]
    s2_hyper_3plus = [[0, 0, 0] for _ in range(len(sent2))]
    # s2_hypo_dist = [[0, 0, 0] for _ in range(len(sent2))]
    # s2_hyper_dist = [[0, 0, 0] for _ in range(len(sent2))]

    features = {}
    features['s1_em'] = s1_em
    features['s1_ant'] = s1_ant
    features['s1_anc'] = s1_anc
    features['s1_desc'] = s1_desc
    features['s1_hypo_1'] = s1_hypo_1
    features['s1_hyper_1'] = s1_hyper_1
    features['s1_hypo_2'] = s1_hypo_2
    features['s1_hyper_2'] = s1_hyper_2
    features['s1_hypo_3plus'] = s1_hypo_3plus
    features['s1_hyper_3plus'] = s1_hyper_3plus
    # features['s1_hypo_dist'] = s1_hypo_dist
    # features['s1_hyper_dist'] = s1_hyper_dist
    features['s2_em'] = s2_em
    features['s2_ant'] = s2_ant
    features['s2_anc'] = s2_anc
    features['s2_desc'] = s2_desc
    features['s2_hypo_1'] = s2_hypo_1
    features['s2_hyper_1'] = s2_hyper_1
    features['s2_hypo_2'] = s2_hypo_2
    features['s2_hyper_2'] = s2_hyper_2
    features['s2_hypo_3plus'] = s2_hypo_3plus
    features['s2_hyper_3plus'] = s2_hyper_3plus
    # features['s2_hypo_dist'] = s2_hypo_dist
    # features['s2_hyper_dist'] = s2_hyper_dist

    for i, (w, pos) in enumerate(sent1):
        stems = get_stem_set(w)

        # if len(stems & s2_lemmas) != 0:
        #     s1_em[i][0] = s1_em[i][2] = 1

        e_match = False
        for w2 in set(map(lambda x: x[0].lower(), sent2)):
            w_lower = w.lower()
            if is_exact_match_p(w_lower, w2, em_dict):
                e_match = True
                break
        if e_match:
            s1_em[i][0] = s1_em[i][2] = 1

        if len(stems & s2_antonym_lemmas) != 0:
            s1_ant[i][0] = s1_ant[i][2] = 1

        if len(stems & s2_hypo_up_to_1) != 0:
            s1_anc[i][0] = s1_anc[i][2] = 1
            s1_hypo_1[i][0] = s1_hypo_1[i][2] = 1
            # s1_hypo_dist[i][0] = s1_hypo_dist[i][1] = 1
        elif len(stems & s2_hypo_up_to_2) != 0:
            s1_anc[i][0] = s1_anc[i][2] = 1
            s1_hypo_2[i][0] = s1_hypo_2[i][2] = 1
            # s1_hypo_dist[i][0] = s1_hypo_dist[i][1] = 2
        elif len(stems & s2_hyponyms) != 0:
            s1_anc[i][0] = s1_anc[i][2] = 1
            s1_hypo_3plus[i][0] = s1_hypo_3plus[i][2] = 1
            # s1_hypo_dist[i][0] = s1_hypo_dist[i][1] = 3

        if len(stems & s2_hyper_up_to_1) != 0:
            s1_desc[i][0] = s1_desc[i][2] = 1
            s1_hyper_1[i][0] = s1_hyper_1[i][2] = 1
            # s1_hyper_dist[i][0] = s1_hyper_dist[i][1] = 1
        elif len(stems & s2_hyper_up_to_2) != 0:
            s1_desc[i][0] = s1_desc[i][2] = 1
            s1_hyper_2[i][0] = s1_hyper_2[i][2] = 1
            # s1_hyper_dist[i][0] = s1_hyper_dist[i][1] = 2
        elif len(stems & s2_hypernyms) != 0:
            s1_desc[i][0] = s1_desc[i][2] = 1
            s1_hyper_3plus[i][0] = s1_hyper_3plus[i][2] = 1
            # s1_hyper_dist[i][0] = s1_hyper_dist[i][1] = 3

    for i, (w, pos) in enumerate(sent2):
        stems = get_stem_set(w)

        # if len(stems & s1_lemmas) != 0:
        #     s2_em[i][1] = s2_em[i][2] = 1

        e_match = False
        for w1 in set(map(lambda x: x[0].lower(), sent1)):
            w_lower = w.lower()
            if is_exact_match_p(w_lower, w1, em_dict):
                e_match = True
                break
        if e_match:
            s2_em[i][1] = s2_em[i][2] = 1

        if len(stems & s1_antonym_lemmas) != 0:
            s2_ant[i][1] = s2_ant[i][2] = 1

        if len(stems & s1_hypo_up_to_1) != 0:
            s2_anc[i][1] = s2_anc[i][2] = 1
            s2_hypo_1[i][1] = s2_hypo_1[i][2] = 1
            # s2_hypo_dist[i][1] = s2_hypo_dist[i][1] = 1
        elif len(stems & s1_hypo_up_to_2) != 0:
            s2_anc[i][1] = s2_anc[i][2] = 1
            s2_hypo_2[i][1] = s2_hypo_2[i][2] = 1
            # s2_hypo_dist[i][1] = s2_hypo_dist[i][1] = 2
        elif len(stems & s1_hyponyms) != 0:
            s2_anc[i][1] = s2_anc[i][2] = 1
            s2_hypo_3plus[i][1] = s2_hypo_3plus[i][2] = 1
            # s2_hypo_dist[i][1] = s2_hypo_dist[i][1] = 3

        if len(stems & s1_hyper_up_to_1) != 0:
            s2_desc[i][1] = s2_desc[i][2] = 1
            s2_hyper_1[i][1] = s2_hyper_1[i][2] = 1
            # s2_hyper_dist[i][1] = s2_hyper_dist[i][1] = 1
        elif len(stems & s1_hyper_up_to_2) != 0:
            s2_desc[i][1] = s2_desc[i][2] = 1
            s2_hyper_2[i][1] = s2_hyper_2[i][2] = 1
            # s2_hyper_dist[i][1] = s2_hyper_dist[i][1] = 2
        elif len(stems & s1_hypernyms) != 0:
            s2_desc[i][1] = s2_desc[i][2] = 1
            s2_hyper_3plus[i][1] = s2_hyper_3plus[i][2] = 1
            # s2_hyper_dist[i][1] = s2_hyper_dist[i][1] = 3

    return features


def compute_wn_features_p_accerate(s1, s2, p_dict):
    sent1 = wn_pos_tag(s1)
    sent2 = wn_pos_tag(s2)

    hypernym_stems_dict = p_dict['hypernym_stems_dict']
    get_hypernym_stems = partial(get_hypernym_stems_p, hypernym_stems_dict=hypernym_stems_dict)

    hyponym_stems_dict = p_dict['hyponym_stems_dict']
    get_hyponym_stems = partial(get_hyponym_stems_p, hyponym_stems_dict=hyponym_stems_dict)

    hyper_lvl_dict = p_dict['hyper_lvl_dict']
    get_hyper_up_to_lv = partial(get_hyper_up_to_lv_p, hyper_lvl_dict=hyper_lvl_dict)

    hypo_lvl_dict = p_dict['hypo_lvl_dict']
    get_hypo_up_to_lv = partial(get_hypo_up_to_lv_p, hypo_lvl_dict=hypo_lvl_dict)

    ant_dict = p_dict['ant_dict']
    get_antonym_stems = partial(get_antonym_stems_p, ant_dict=ant_dict)

    # em_dict = p_dict['em_dict']
    em_lemmas_dict = p_dict['em_lemmas_dict']
    # is_exact_match = partial(is_exact_match_p, em_dict=em_dict)

    # s1_lemmas = reduce(lambda x, y: x | y,
    #                    map(lambda item: get_stem_set(item[0]), sent1))

    s1_em_stem_lemmas = set()
    s1_antonym_lemmas = set()
    s1_hypernyms = set()
    s1_hyponyms = set()
    s1_hyper_up_to_1 = set()
    s1_hyper_up_to_2 = set()
    s1_hypo_up_to_1 = set()
    s1_hypo_up_to_2 = set()

    for word, tag in sent1:
        s1_em_stem_lemmas |= get_lemma_stem_dict(word, em_lemmas_dict)
        s1_antonym_lemmas |= get_antonym_stems((word, tag))
        s1_hypernyms |= get_hypernym_stems((word, tag))
        s1_hyponyms |= get_hyponym_stems((word, tag))
        s1_hyper_up_to_1 |= get_hyper_up_to_lv(item=(word, tag), level=1)
        s1_hyper_up_to_2 |= get_hyper_up_to_lv(item=(word, tag), level=2)
        s1_hypo_up_to_1 |= get_hypo_up_to_lv(item=(word, tag), level=1)
        s1_hypo_up_to_2 |= get_hypo_up_to_lv(item=(word, tag), level=2)

    s1_hyper_up_to_2 -= s1_hyper_up_to_1
    s1_hypo_up_to_2 -= s1_hypo_up_to_1

    s1_hypernyms -= (s1_hyper_up_to_1 | s1_hyper_up_to_2)
    s1_hyponyms -= (s1_hypo_up_to_1 | s1_hypo_up_to_2)

    s2_em_stem_lemmas = set()
    s2_antonym_lemmas = set()
    s2_hypernyms = set()
    s2_hyponyms = set()
    s2_hyper_up_to_1 = set()
    s2_hyper_up_to_2 = set()
    s2_hypo_up_to_1 = set()
    s2_hypo_up_to_2 = set()

    for word, tag in sent2:
        s2_em_stem_lemmas |= get_lemma_stem_dict(word, em_lemmas_dict)
        s2_antonym_lemmas |= get_antonym_stems((word, tag))
        s2_hypernyms |= get_hypernym_stems((word, tag))
        s2_hyponyms |= get_hyponym_stems((word, tag))
        s2_hyper_up_to_1 |= get_hyper_up_to_lv(item=(word, tag), level=1)
        s2_hyper_up_to_2 |= get_hyper_up_to_lv(item=(word, tag), level=2)
        s2_hypo_up_to_1 |= get_hypo_up_to_lv(item=(word, tag), level=1)
        s2_hypo_up_to_2 |= get_hypo_up_to_lv(item=(word, tag), level=2)

    s2_hyper_up_to_2 -= s2_hyper_up_to_1
    s2_hypo_up_to_2 -= s2_hypo_up_to_1

    s2_hypernyms -= (s2_hyper_up_to_1 | s2_hyper_up_to_2)
    s2_hyponyms -= (s2_hypo_up_to_1 | s2_hypo_up_to_2)

    # s1_antonym_lemmas = reduce(lambda x, y: x | y,
    #                            map(get_antonym_stems, sent1))
    # s1_hypernyms = reduce(lambda x, y: x.union(y),
    #                       map(get_hypernym_stems, sent1))
    # s1_hyponyms = reduce(lambda x, y: x.union(y),
    #                      map(get_hyponym_stems, sent1))
    # s1_hyper_up_to_1 = reduce(lambda x, y: x | y,
    #                           map(lambda w: get_hyper_up_to_lv(item=w, level=1), sent1))
    # s1_hyper_up_to_2 = reduce(lambda x, y: x | y,
    #                           map(lambda w: get_hyper_up_to_lv(item=w, level=2), sent1)) - \
    #                    s1_hyper_up_to_1
    # s1_hypo_up_to_1 = reduce(lambda x, y: x | y,
    #                          map(lambda w: get_hypo_up_to_lv(item=w, level=1), sent1))
    # s1_hypo_up_to_2 = reduce(lambda x, y: x | y,
    #                          map(lambda w: get_hypo_up_to_lv(item=w, level=2), sent1)) - \
    #                   s1_hypo_up_to_1
    # s1_hypernyms -= (s1_hyper_up_to_1 | s1_hyper_up_to_2)
    # s1_hyponyms -= (s1_hypo_up_to_1 | s1_hypo_up_to_2)

    # s2_lemmas = reduce(lambda x, y: x | y, map(lambda item: get_stem_set(item[0]), sent2))

    # s2_antonym_lemmas = reduce(lambda x, y: x | y,
    #                            map(get_antonym_stems, sent2))
    # s2_hypernyms = reduce(lambda x, y: x.union(y),
    #                       map(get_hypernym_stems, sent2))
    # s2_hyponyms = reduce(lambda x, y: x.union(y),
    #                      map(get_hyponym_stems, sent2))
    # s2_hyper_up_to_1 = reduce(lambda x, y: x | y,
    #                           map(lambda w: get_hyper_up_to_lv(item=w, level=1), sent2))
    # s2_hyper_up_to_2 = reduce(lambda x, y: x | y,
    #                           map(lambda w: get_hyper_up_to_lv(item=w, level=2), sent2)) - \
    #                    s2_hyper_up_to_1
    # s2_hypo_up_to_1 = reduce(lambda x, y: x | y,
    #                          map(lambda w: get_hypo_up_to_lv(item=w, level=1), sent2))
    # s2_hypo_up_to_2 = reduce(lambda x, y: x | y,
    #                          map(lambda w: get_hypo_up_to_lv(item=w, level=2), sent2)) - \
    #                   s2_hypo_up_to_1
    # s2_hypernyms -= (s2_hyper_up_to_1 | s2_hyper_up_to_2)
    # s2_hyponyms -= (s2_hypo_up_to_1 | s2_hypo_up_to_2)

    s1_em = [[0, 0, 0] for _ in range(len(sent1))]
    s1_ant = [[0, 0, 0] for _ in range(len(sent1))]
    s1_anc = [[0, 0, 0] for _ in range(len(sent1))]
    s1_desc = [[0, 0, 0] for _ in range(len(sent1))]
    s1_hypo_1 = [[0, 0, 0] for _ in range(len(sent1))]
    s1_hyper_1 = [[0, 0, 0] for _ in range(len(sent1))]
    s1_hypo_2 = [[0, 0, 0] for _ in range(len(sent1))]
    s1_hyper_2 = [[0, 0, 0] for _ in range(len(sent1))]
    s1_hypo_3plus = [[0, 0, 0] for _ in range(len(sent1))]
    s1_hyper_3plus = [[0, 0, 0] for _ in range(len(sent1))]

    # s1_hypo_dist = [[0, 0, 0] for _ in range(len(sent1))]
    # s1_hyper_dist = [[0, 0, 0] for _ in range(len(sent1))]

    s2_em = [[0, 0, 0] for _ in range(len(sent2))]
    s2_ant = [[0, 0, 0] for _ in range(len(sent2))]
    s2_anc = [[0, 0, 0] for _ in range(len(sent2))]
    s2_desc = [[0, 0, 0] for _ in range(len(sent2))]
    s2_hypo_1 = [[0, 0, 0] for _ in range(len(sent2))]
    s2_hyper_1 = [[0, 0, 0] for _ in range(len(sent2))]
    s2_hypo_2 = [[0, 0, 0] for _ in range(len(sent2))]
    s2_hyper_2 = [[0, 0, 0] for _ in range(len(sent2))]
    s2_hypo_3plus = [[0, 0, 0] for _ in range(len(sent2))]
    s2_hyper_3plus = [[0, 0, 0] for _ in range(len(sent2))]
    # s2_hypo_dist = [[0, 0, 0] for _ in range(len(sent2))]
    # s2_hyper_dist = [[0, 0, 0] for _ in range(len(sent2))]

    features = {}
    features['s1_em'] = s1_em
    features['s1_ant'] = s1_ant
    features['s1_anc'] = s1_anc
    features['s1_desc'] = s1_desc
    features['s1_hypo_1'] = s1_hypo_1
    features['s1_hyper_1'] = s1_hyper_1
    features['s1_hypo_2'] = s1_hypo_2
    features['s1_hyper_2'] = s1_hyper_2
    features['s1_hypo_3plus'] = s1_hypo_3plus
    features['s1_hyper_3plus'] = s1_hyper_3plus
    # features['s1_hypo_dist'] = s1_hypo_dist
    # features['s1_hyper_dist'] = s1_hyper_dist
    features['s2_em'] = s2_em
    features['s2_ant'] = s2_ant
    features['s2_anc'] = s2_anc
    features['s2_desc'] = s2_desc
    features['s2_hypo_1'] = s2_hypo_1
    features['s2_hyper_1'] = s2_hyper_1
    features['s2_hypo_2'] = s2_hypo_2
    features['s2_hyper_2'] = s2_hyper_2
    features['s2_hypo_3plus'] = s2_hypo_3plus
    features['s2_hyper_3plus'] = s2_hyper_3plus
    # features['s2_hypo_dist'] = s2_hypo_dist
    # features['s2_hyper_dist'] = s2_hyper_dist

    for i, (w, pos) in enumerate(sent1):
        stems = get_stem_set(w)

        if len(get_lemma_stem_dict(w, em_lemmas_dict) & s2_em_stem_lemmas) != 0:
            s1_em[i][0] = s1_em[i][2] = 1

        # e_match = False
        # for w2 in set(map(lambda x: x[0].lower(), sent2)):
        #     w_lower = w.lower()
        #     if is_exact_match_p(w_lower, w2, em_dict):
        #         e_match = True
        #         break
        # if e_match:
        #     s1_em[i][0] = s1_em[i][2] = 1

        if len(stems & s2_antonym_lemmas) != 0:
            s1_ant[i][0] = s1_ant[i][2] = 1

        if len(stems & s2_hypo_up_to_1) != 0:
            s1_anc[i][0] = s1_anc[i][2] = 1
            s1_hypo_1[i][0] = s1_hypo_1[i][2] = 1
            # s1_hypo_dist[i][0] = s1_hypo_dist[i][1] = 1
        elif len(stems & s2_hypo_up_to_2) != 0:
            s1_anc[i][0] = s1_anc[i][2] = 1
            s1_hypo_2[i][0] = s1_hypo_2[i][2] = 1
            # s1_hypo_dist[i][0] = s1_hypo_dist[i][1] = 2
        elif len(stems & s2_hyponyms) != 0:
            s1_anc[i][0] = s1_anc[i][2] = 1
            s1_hypo_3plus[i][0] = s1_hypo_3plus[i][2] = 1
            # s1_hypo_dist[i][0] = s1_hypo_dist[i][1] = 3

        if len(stems & s2_hyper_up_to_1) != 0:
            s1_desc[i][0] = s1_desc[i][2] = 1
            s1_hyper_1[i][0] = s1_hyper_1[i][2] = 1
            # s1_hyper_dist[i][0] = s1_hyper_dist[i][1] = 1
        elif len(stems & s2_hyper_up_to_2) != 0:
            s1_desc[i][0] = s1_desc[i][2] = 1
            s1_hyper_2[i][0] = s1_hyper_2[i][2] = 1
            # s1_hyper_dist[i][0] = s1_hyper_dist[i][1] = 2
        elif len(stems & s2_hypernyms) != 0:
            s1_desc[i][0] = s1_desc[i][2] = 1
            s1_hyper_3plus[i][0] = s1_hyper_3plus[i][2] = 1
            # s1_hyper_dist[i][0] = s1_hyper_dist[i][1] = 3

    for i, (w, pos) in enumerate(sent2):
        stems = get_stem_set(w)

        # if len(stems & s1_lemmas) != 0:
        #     s2_em[i][1] = s2_em[i][2] = 1

        if len(get_lemma_stem_dict(w, em_lemmas_dict) & s1_em_stem_lemmas) != 0:
            s2_em[i][1] = s2_em[i][2] = 1

        # e_match = False
        # for w1 in set(map(lambda x: x[0].lower(), sent1)):
        #     w_lower = w.lower()
        #     if is_exact_match_p(w_lower, w1, em_dict):
        #         e_match = True
        #         break
        # if e_match:
        #     s2_em[i][1] = s2_em[i][2] = 1

        if len(stems & s1_antonym_lemmas) != 0:
            s2_ant[i][1] = s2_ant[i][2] = 1

        if len(stems & s1_hypo_up_to_1) != 0:
            s2_anc[i][1] = s2_anc[i][2] = 1
            s2_hypo_1[i][1] = s2_hypo_1[i][2] = 1
            # s2_hypo_dist[i][1] = s2_hypo_dist[i][1] = 1
        elif len(stems & s1_hypo_up_to_2) != 0:
            s2_anc[i][1] = s2_anc[i][2] = 1
            s2_hypo_2[i][1] = s2_hypo_2[i][2] = 1
            # s2_hypo_dist[i][1] = s2_hypo_dist[i][1] = 2
        elif len(stems & s1_hyponyms) != 0:
            s2_anc[i][1] = s2_anc[i][2] = 1
            s2_hypo_3plus[i][1] = s2_hypo_3plus[i][2] = 1
            # s2_hypo_dist[i][1] = s2_hypo_dist[i][1] = 3

        if len(stems & s1_hyper_up_to_1) != 0:
            s2_desc[i][1] = s2_desc[i][2] = 1
            s2_hyper_1[i][1] = s2_hyper_1[i][2] = 1
            # s2_hyper_dist[i][1] = s2_hyper_dist[i][1] = 1
        elif len(stems & s1_hyper_up_to_2) != 0:
            s2_desc[i][1] = s2_desc[i][2] = 1
            s2_hyper_2[i][1] = s2_hyper_2[i][2] = 1
            # s2_hyper_dist[i][1] = s2_hyper_dist[i][1] = 2
        elif len(stems & s1_hypernyms) != 0:
            s2_desc[i][1] = s2_desc[i][2] = 1
            s2_hyper_3plus[i][1] = s2_hyper_3plus[i][2] = 1
            # s2_hyper_dist[i][1] = s2_hyper_dist[i][1] = 3

    return features


default_fn_list = [
    'em',
    'ant',
    'anc',
    'desc',
    'hypo_1',
    'hyper_1',
    'hypo_2',
    'hyper_2',
    'hypo_3plus',
    'hyper_3plus'
]


def wn_raw_feature_to_nparray(feature, wn_feature_name_list=default_fn_list):
    p_feature_list = []
    for fname in wn_feature_name_list:
        p_cur_feature_array = np.asarray(feature[f's1_{fname}'], dtype=np.float32)
        p_feature_list.append(p_cur_feature_array)
    p_w_feature = np.concatenate(p_feature_list, axis=1)

    h_feature_list = []
    for fname in wn_feature_name_list:
        h_cur_feature_array = np.asarray(feature[f's2_{fname}'], dtype=np.float32)
        h_feature_list.append(h_cur_feature_array)
    h_w_feature = np.concatenate(h_feature_list, axis=1)

    return p_w_feature, h_w_feature


if __name__ == '__main__':

    # print(get_lemma_stem_dict('woman', dict()))
    # print(get_stem_set('women'))
    # print(get_lemma_stem_dict('women', dict()))
    #
    # exit(0)
    # sent1 = "Two woman are embracing while holding to go packages ."
    # sent2 = "The sisters are hugging goodbye while holding to go packages after just eating lunch ."
    #
    # features = compute_wn_features(sent1.split(' '), sent2.split(' '))
    # print(features)
    # print(features.keys())

    sent1 = ['Two', 'women', 'who', 'just', 'had', 'lunch', 'hugging', 'and', 'saying', 'goodbye', '.']
    sent2 = ['There', 'are', 'two', 'woman', 'in', 'this', 'picture', '.']
    # sent2 = "The sisters are hugging goodbye while holding to go packages after just eating lunch ."

    features = compute_wn_features(sent1, sent2)
    print(features['s1_em'])
    print(features['s2_em'])
    # print(features['s2_em'])
    print(features.keys())

    # for i in tqdm(range(1000)):
    #     features = compute_wn_features(sent1.split(' '), sent2.split(' '))

    p_dict = {
        'hypernym_stems_dict': dict(),
        'hyponym_stems_dict': dict(),
        'hyper_lvl_dict': dict(),
        'hypo_lvl_dict': dict(),
        'ant_dict': dict(),
        'em_dict': dict(),
        'em_lemmas_dict': dict()
    }
    #
    # for i in tqdm(range(2000)):
        # features_p = compute_wn_features_p(sent1.split(' '), sent2.split(' '), p_dict)
        # features_p = compute_wn_features_p(sent1, sent2, p_dict)

    # exit(0)
    # features = compute_wn_features(sent1, sent2)
    # print(features_p['s1_em'])
    # print(features_p['s2_em'])

    # p_feature, h_feature = wn_raw_feature_to_nparray(features_p)
    # print(p_feature)
    # print(h_feature)

    for i in tqdm(range(2000)):
        feature_p_a = compute_wn_features_p_accerate(sent1, sent2, p_dict)

    # assert features_p == feature_p_a
    # print("AAAAAAAAAA")
    print(feature_p_a['s1_em'])
    print(feature_p_a['s2_em'])

    assert feature_p_a['s1_em'] == features['s1_em']
    # print(features['s2_em'])
    # print(features.keys())
    #     if not features_p == features:
    #         print(features_p)

    # hypernym_stems_dict = p_dict['hypernym_stems_dict']
    # get_hypernym_stems = partial(get_hypernym_stems_p, hypernym_stems_dict=hypernym_stems_dict)
    #
    # hyponym_stems_dict = p_dict['hyponym_stems_dict']
    # get_hyponym_stems = partial(get_hyponym_stems_p, hyponym_stems_dict=hyponym_stems_dict)
    #
    # hyper_lvl_dict = p_dict['hyper_lvl_dict']
    # get_hyper_up_to_lv = partial(get_hyper_up_to_lv_p, hyper_lvl_dict=hyper_lvl_dict)
    #
    # hypo_lvl_dict = p_dict['hypo_lvl_dict']
    # get_hypo_up_to_lv = partial(get_hypo_up_to_lv_p, hypo_lvl_dict=hypo_lvl_dict)
    #
    # ant_dict = p_dict['ant_dict']
    # get_antonym_stems = partial(get_antonym_stems_p, ant_dict=ant_dict)
