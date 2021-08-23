"""
A Vocabulary maps strings to integers, allowing for strings to be mapped to an
out-of-vocabulary token.
"""

from collections import defaultdict
from pathlib import Path
from typing import Callable, Any, List, Dict, Union, Sequence, Set, Optional, Iterable
import codecs
import logging
import os
import gzip
import hashlib

import numpy
import torch
from allennlp.common.util import namespace_match
from allennlp.common.params import Params
from allennlp.common.checks import ConfigurationError
from allennlp.common.file_utils import cached_path
from allennlp.common.tqdm import Tqdm
from allennlp.data import instance as adi  # pylint: disable=unused-import

from enum import Enum

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

DEFAULT_NON_PADDED_NAMESPACES = ("*tags", "*labels")
DEFAULT_PADDING_TOKEN = "@@PADDING@@"
DEFAULT_OOV_TOKEN = "@@UNKNOWN@@"
NAMESPACE_PADDING_FILE = 'non_padded_namespaces.txt'
UNK_COUNT_FILE = 'unk_count_namespaces.txt'

TOKEN_INDEX_SEP = '<(-.-)>'


class MappingMode(Enum):
    token2index = 1
    index2token = 2


class _NamespaceDependentDefaultDict(defaultdict):
    """
    This is a `defaultdict
    <http://docs.python.org/2/library/collections.html#collections.defaultdict>`_ where the
    default value is dependent on the key that is passed.

    We use "namespaces" in the :class:`Vocabulary` object to keep track of several different
    mappings from strings to integers, so that we have a consistent API for mapping words, tags,
    labels, characters, or whatever else you want, into integers.  The issue is that some of those
    namespaces (words and characters) should have integers reserved for padding and
    out-of-vocabulary tokens, while others (labels and tags) shouldn't.  This class allows you to
    specify filters on the namespace (the key used in the ``defaultdict``), and use different
    default values depending on whether the namespace passes the filter.

    To do filtering, we take a sequence of ``non_padded_namespaces``.  This is a list or tuple of
    strings that are either matched exactly against the keys, or treated as suffixes, if the
    string starts with ``*``.  In other words, if ``*tags`` is in ``non_padded_namespaces`` then
    ``passage_tags``, ``question_tags``, etc. (anything that ends with ``tags``) will have the
    ``non_padded`` default value.

    Parameters
    ----------
    non_padded_namespaces : ``Sequence[str]``
        A list or tuple of strings describing which namespaces are not padded.  If a namespace
        (key) is missing from this dictionary, we will use :func:`namespace_match` to see whether
        the namespace should be padded.  If the given namespace matches any of the strings in this
        list, we will use ``non_padded_function`` to initialize the value for that namespace, and
        we will use ``padded_function`` otherwise.
    padded_function : ``Callable[[], Any]``
        A zero-argument function to call to initialize a value for a namespace that `should` be
        padded.
    non_padded_function : ``Callable[[], Any]``
        A zero-argument function to call to initialize a value for a namespace that should `not` be
        padded.
    """

    def __init__(self,
                 non_padded_namespaces: Sequence[str],
                 padded_function: Callable[[], Any],
                 non_padded_function: Callable[[], Any],
                 padding_token: str = DEFAULT_PADDING_TOKEN,
                 oov_token: str = DEFAULT_OOV_TOKEN) -> None:

        self._non_padded_namespaces = non_padded_namespaces
        self.padding_token = padding_token
        self.oov_token = oov_token.replace('@', '')

        self._padded_function = padded_function
        self._non_padded_function = non_padded_function
        super(_NamespaceDependentDefaultDict, self).__init__()

    def __missing__(self, key: str):
        if any(namespace_match(pattern, key) for pattern in self._non_padded_namespaces):
            value = self._non_padded_function()
        else:
            value = self._padded_function()
        dict.__setitem__(self, key, value)
        return value

    def initialize_dictionary(self, namespace: str, unk_num: int, mode: MappingMode):
        if mode == MappingMode.token2index:
            if any(namespace_match(pattern, namespace) for pattern in self._non_padded_namespaces):
                dict.__setitem__(self, namespace, {})
            else:
                init_namespace_dictionary = RandomHashDict(unk_num=unk_num, oov_token=self.oov_token)
                init_namespace_dictionary.update({self.padding_token: 0})
                init_namespace_dictionary.add_unk_tokens()

                dict.__setitem__(self, namespace, init_namespace_dictionary)

        elif mode == MappingMode.index2token:
            if any(namespace_match(pattern, namespace) for pattern in self._non_padded_namespaces):
                dict.__setitem__(self, namespace, {})
            else:
                init_namespace_dictionary = {0: self.padding_token}
                for i in range(unk_num):
                    init_namespace_dictionary[len(init_namespace_dictionary)] = f"@@{self.oov_token}#{str(i)}@@"

                dict.__setitem__(self, namespace, init_namespace_dictionary)


class RandomHashDict(dict):
    def __init__(self, unk_num: int = 1, oov_token: str = DEFAULT_OOV_TOKEN, **kwargs):
        super().__init__(**kwargs)
        self.unk_num = unk_num
        self.oov_token = oov_token.replace('@', '')

    def add_unk_tokens(self):
        for i in range(self.unk_num):
            self.__setitem__(f"@@{self.oov_token}#{str(i)}@@", self.__len__())

    def __getitem__(self, k):
        if self.unk_num == 0 or k in super().keys():
            return super().__getitem__(k)
        else:
            k = self.hash_string(k, self.unk_num)
            return super().__getitem__(k)

    def hash_string(self, input_str, unk_num):
        hcode = int(hashlib.sha1(input_str.encode('utf-8')).hexdigest(), 16)
        hcode %= unk_num
        return f"@@{self.oov_token}#{str(hcode)}@@"

    def __str__(self) -> str:
        return super().__str__()


def _read_pretrained_words(embeddings_filename: str) -> Set[str]:
    words = set()
    with gzip.open(cached_path(embeddings_filename), 'rb') as embeddings_file:
        for line in embeddings_file:
            fields = line.decode('utf-8').strip().split(' ')
            word = fields[0]
            words.add(word)
    return words


class ExVocabulary:
    def __init__(self,
                 counter: Dict[str, Dict[str, int]] = None,
                 min_count: Dict[str, int] = None,
                 max_vocab_size: Union[int, Dict[str, int]] = None,
                 non_padded_namespaces: Sequence[str] = DEFAULT_NON_PADDED_NAMESPACES,
                 pretrained_files: Optional[Dict[str, str]] = None,
                 only_include_pretrained_words: bool = False,
                 unk_token_num: Dict[str, int] = None) -> None:

        self._padding_token = DEFAULT_PADDING_TOKEN
        self._oov_token = DEFAULT_OOV_TOKEN
        if not isinstance(max_vocab_size, dict):
            int_max_vocab_size = max_vocab_size
            max_vocab_size = defaultdict(lambda: int_max_vocab_size)  # type: ignore
        self._non_padded_namespaces = non_padded_namespaces
        self.unk_token_num = unk_token_num

        self._initialize_dictionary(list(self.unk_token_num.keys()), non_padded_namespaces,
                                    self._padding_token, self._oov_token)

        min_count = min_count or {}
        pretrained_files = pretrained_files or {}
        if counter is not None:
            for namespace in counter:
                if namespace in pretrained_files:
                    pretrained_list = _read_pretrained_words(pretrained_files[namespace])
                else:
                    pretrained_list = None
                token_counts = list(counter[namespace].items())
                token_counts.sort(key=lambda x: x[1], reverse=True)
                max_vocab = max_vocab_size[namespace]
                if max_vocab:
                    token_counts = token_counts[:max_vocab]
                for token, count in token_counts:
                    if pretrained_list is not None:
                        if only_include_pretrained_words:
                            if token in pretrained_list and count >= min_count.get(namespace, 1):
                                self.add_token_to_namespace(token, namespace)
                        elif token in pretrained_list or count >= min_count.get(namespace, 1):
                            self.add_token_to_namespace(token, namespace)
                    elif count >= min_count.get(namespace, 1):
                        self.add_token_to_namespace(token, namespace)

    def _initialize_dictionary(self, namespaces: List[str], non_padded_namespaces, padding_token, oov_token):

        self._token_to_index = _NamespaceDependentDefaultDict(
            non_padded_namespaces,
            lambda: {padding_token: 0, oov_token: 1},
            lambda: {},
            padding_token,
            oov_token
        )

        self._index_to_token = _NamespaceDependentDefaultDict(
            non_padded_namespaces,
            lambda: {0: padding_token, 1: oov_token},
            lambda: {},
            padding_token,
            oov_token
        )

        if self.unk_token_num is None:
            self.unk_token_num = defaultdict(lambda: 1)
        for namespace in namespaces:
            cur_unk_num = self.unk_token_num.get(namespace, 1)
            self._token_to_index.initialize_dictionary(namespace, cur_unk_num, MappingMode.token2index)
            self._index_to_token.initialize_dictionary(namespace, cur_unk_num, MappingMode.index2token)

    def save_to_files(self, directory: str) -> None:
        """
        Persist this Vocabulary to files so it can be reloaded later.
        Each namespace corresponds to one file.

        Parameters
        ----------
        directory : ``str``
            The directory where we save the serialized vocabulary.
        """
        os.makedirs(directory, exist_ok=True)
        if os.listdir(directory):
            logging.warning("vocabulary serialization directory %s is not empty", directory)

        with codecs.open(os.path.join(directory, NAMESPACE_PADDING_FILE), 'w', 'utf-8') as namespace_file:
            for namespace_str in self._non_padded_namespaces:
                print(namespace_str, file=namespace_file)

        with codecs.open(os.path.join(directory, UNK_COUNT_FILE), 'w', 'utf-8') as unk_count_file:
            for namespace_str, unk_count_value in self.unk_token_num.items():
                print(namespace_str + '###' + str(unk_count_value), file=unk_count_file)

        # assert len(self._token_to_index) == len(self._index_to_token)

        for namespace, mapping in self._index_to_token.items():
            if namespace in self._token_to_index:
                assert len(self._token_to_index[namespace]) == len(self._index_to_token[namespace])
            # Each namespace gets written to its own file, in index order.
            with codecs.open(os.path.join(directory, namespace + '.txt'), 'w', 'utf-8') as token_file:
                num_tokens = len(mapping)
                for i in range(num_tokens):
                    if namespace in self._token_to_index:
                        assert mapping[i] == self._index_to_token[namespace][self._token_to_index[namespace][mapping[i]]]

                    print(mapping[i].replace('\n', '@@NEWLINE@@') + TOKEN_INDEX_SEP + str(i), file=token_file)

    @classmethod
    def from_files(cls, directory: str) -> 'ExVocabulary':
        """
        Loads a ``Vocabulary`` that was serialized using ``save_to_files``.

        Parameters
        ----------
        directory : ``str``
            The directory containing the serialized vocabulary.
        """
        logger.info("Loading token dictionary from %s.", directory)
        with codecs.open(os.path.join(directory, NAMESPACE_PADDING_FILE), 'r', 'utf-8') as namespace_file:
            non_padded_namespaces = [namespace_str.strip() for namespace_str in namespace_file]

        unk_token_num = dict()
        with codecs.open(os.path.join(directory, UNK_COUNT_FILE), 'r', 'utf-8') as unk_count_file:
            for unk_count in unk_count_file:
                namespace_str, unk_count_value = unk_count.split('###')[0], int(unk_count.split('###')[1])
                unk_token_num[namespace_str] = unk_count_value

        vocab = ExVocabulary(non_padded_namespaces=non_padded_namespaces, unk_token_num=unk_token_num)

        # Check every file in the directory.
        for namespace_filename in os.listdir(directory):
            if namespace_filename == NAMESPACE_PADDING_FILE:
                continue
            elif namespace_filename == UNK_COUNT_FILE:
                continue
            elif namespace_filename == 'weights':
                continue # This is used for save weights

            namespace = namespace_filename.replace('.txt', '')

            filename = os.path.join(directory, namespace_filename)
            vocab.set_from_file(filename, namespace=namespace)

        return vocab

    def set_from_file(self,
                      filename: str,
                      namespace: str = "tokens"):
        """
        If you already have a vocabulary file for a trained model somewhere, and you really want to
        use that vocabulary file instead of just setting the vocabulary from a dataset, for
        whatever reason, you can do that with this method.  You must specify the namespace to use,
        and we assume that you want to use padding and OOV tokens for this.

        Parameters
        ----------
        filename : ``str``
            The file containing the vocabulary to load.  It should be formatted as one token per
            line, with nothing else in the line.  The index we assign to the token is the line
            number in the file (1-indexed if ``is_padded``, 0-indexed otherwise).  Note that this
            file should contain the OOV token string!
        is_padded : ``bool``, optional (default=True)
            Is this vocabulary padded?  For token / word / character vocabularies, this should be
            ``True``; while for tag or label vocabularies, this should typically be ``False``.  If
            ``True``, we add a padding token with index 0, and we enforce that the ``oov_token`` is
            present in the file.
        oov_token : ``str``, optional (default=DEFAULT_OOV_TOKEN)
            What token does this vocabulary use to represent out-of-vocabulary characters?  This
            must show up as a line in the vocabulary file.  When we find it, we replace
            ``oov_token`` with ``self._oov_token``, because we only use one OOV token across
            namespaces.
        namespace : ``str``, optional (default="tokens")
            What namespace should we overwrite with this vocab file?
        """
        # self._token_to_index[namespace] = {}
        # self._index_to_token[namespace] = {}

        with codecs.open(filename, 'r', 'utf-8') as input_file:
            lines = input_file.read().split('\n')
            # Be flexible about having final newline or not
            if lines and lines[-1] == '':
                lines = lines[:-1]
            for i, line in enumerate(lines):
                row = line.split(TOKEN_INDEX_SEP)
                token, index = row[0], int(row[1])
                token = token.replace('@@NEWLINE@@', '\n')

                if token not in self._token_to_index[namespace]:
                    self._token_to_index[namespace][token] = index
                    self._index_to_token[namespace][index] = token
                else:
                    assert self._token_to_index[namespace][token] == index
                    assert self._index_to_token[namespace][index] == token

                assert len(self._token_to_index) == len(self._index_to_token)

    @classmethod
    def from_instances(cls,
                       instances: Iterable['adi.Instance'],
                       min_count: Dict[str, int] = None,
                       max_vocab_size: Union[int, Dict[str, int]] = None,
                       non_padded_namespaces: Sequence[str] = DEFAULT_NON_PADDED_NAMESPACES,
                       pretrained_files: Optional[Dict[str, str]] = None,
                       only_include_pretrained_words: bool = False,
                       unk_token_num: Dict[str, int] = None,
                       exclude_namespaces=None,
                       include_namespaces=None) -> 'ExVocabulary':
        """
        Constructs a vocabulary given a collection of `Instances` and some parameters.
        We count all of the vocabulary items in the instances, then pass those counts
        and the other parameters, to :func:`__init__`.  See that method for a description
        of what the other parameters do.
        """
        logger.info("Fitting token dictionary from dataset.")
        namespace_token_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        for instance in Tqdm.tqdm(instances):
            instance.count_vocab_items(namespace_token_counts)

        if exclude_namespaces is not None:
            for namespace in namespace_token_counts:
                if namespace in exclude_namespaces:
                    namespace_token_counts[namespace] = dict()

                if include_namespaces is not None:
                    # If include namespaces is not None, we only include those namespaces.
                    if namespace not in include_namespaces:
                        namespace_token_counts[namespace] = dict()

        print("Start counting for namespaces:")
        for namespace, counter in namespace_token_counts.items():
            if len(counter) != 0:
                print(namespace)

        return ExVocabulary(counter=namespace_token_counts,
                            min_count=min_count,
                            max_vocab_size=max_vocab_size,
                            non_padded_namespaces=non_padded_namespaces,
                            pretrained_files=pretrained_files,
                            only_include_pretrained_words=only_include_pretrained_words,
                            unk_token_num=unk_token_num)

    @classmethod
    def from_params(cls, params: Params, instances: Iterable['adi.Instance'] = None):
        """
        There are two possible ways to build a vocabulary; from a
        collection of instances, using :func:`Vocabulary.from_instances`, or
        from a pre-saved vocabulary, using :func:`Vocabulary.from_files`.
        This method wraps both of these options, allowing their specification
        from a ``Params`` object, generated from a JSON configuration file.

        Parameters
        ----------
        params: Params, required.
        dataset: Dataset, optional.
            If ``params`` doesn't contain a ``vocabulary_directory`` key,
            the ``Vocabulary`` can be built directly from a ``Dataset``.

        Returns
        -------
        A ``Vocabulary``.
        """
        vocabulary_directory = params.pop("directory_path", None)
        if not vocabulary_directory and not instances:
            raise ConfigurationError("You must provide either a Params object containing a "
                                     "vocab_directory key or a Dataset to build a vocabulary from.")
        if vocabulary_directory and instances:
            logger.info("Loading Vocab from files instead of dataset.")

        if vocabulary_directory:
            params.assert_empty("Vocabulary - from files")
            return Vocabulary.from_files(vocabulary_directory)

        min_count = params.pop("min_count", None)
        max_vocab_size = params.pop_int("max_vocab_size", None)
        non_padded_namespaces = params.pop("non_padded_namespaces", DEFAULT_NON_PADDED_NAMESPACES)
        pretrained_files = params.pop("pretrained_files", {})
        only_include_pretrained_words = params.pop_bool("only_include_pretrained_words", False)
        params.assert_empty("Vocabulary - from dataset")
        return ExVocabulary.from_instances(instances=instances,
                                           min_count=min_count,
                                           max_vocab_size=max_vocab_size,
                                           non_padded_namespaces=non_padded_namespaces,
                                           pretrained_files=pretrained_files,
                                           only_include_pretrained_words=only_include_pretrained_words)

    def is_padded(self, namespace: str) -> bool:
        """
        Returns whether or not there are padding and OOV tokens added to the given namepsace.
        """
        return self._index_to_token[namespace][0] == self._padding_token

    def add_token_to_namespace(self, token: str, namespace: str = 'tokens') -> int:
        """
        Adds ``token`` to the index, if it is not already present.  Either way, we return the index of
        the token.
        """
        if not isinstance(token, str):
            raise ValueError("Vocabulary tokens must be strings, or saving and loading will break."
                             "  Got %s (with type %s)" % (repr(token), type(token)))
        if token not in self._token_to_index[namespace]:
            index = len(self._token_to_index[namespace])
            self._token_to_index[namespace][token] = index
            self._index_to_token[namespace][index] = token
            return index
        else:
            return self._token_to_index[namespace][token]

    def change_token_with_index_to_namespace(self, token: str, index: int, namespace: str = 'tokens') -> int:
        """
        Adds ``token`` to the index, if it is not already present.  Either way, we return the index of
        the token.
        """
        if not isinstance(token, str):
            raise ValueError("Vocabulary tokens must be strings, or saving and loading will break."
                             "  Got %s (with type %s)" % (repr(token), type(token)))

        if index in self._index_to_token[namespace] or index >= 0:
            if self._token_to_index[namespace][token] == index and \
                    self._index_to_token[namespace][index] == token:
                    return 0  # Already changed

            raise ValueError(f"Index: {index} already exists or is invalid in the {namespace}, "
                             f"can not set special Token: {token}")

        if token in self._token_to_index[namespace]:
            print(f"Change index to for an existing token: {token}")
            self._index_to_token[namespace].pop(self._token_to_index[namespace][token])

        self._token_to_index[namespace][token] = index
        self._index_to_token[namespace][index] = token

    def get_index_to_token_vocabulary(self, namespace: str = 'tokens') -> Dict[int, str]:
        return self._index_to_token[namespace]

    def get_token_to_index_vocabulary(self, namespace: str = 'tokens') -> Dict[str, int]:
        return self._token_to_index[namespace]

    def get_token_index(self, token: str, namespace: str = 'tokens') -> int:
        if token in self._token_to_index[namespace]:
            return self._token_to_index[namespace][token]
        elif namespace in self.unk_token_num.keys():  # If we specify the unk token.
            return self._token_to_index[namespace][token]
        else:
            try:
                return self._token_to_index[namespace][self._oov_token]
            except KeyError:
                logger.error('Namespace: %s', namespace)
                logger.error('Token: %s', token)
                raise

    def get_token_from_index(self, index: int, namespace: str = 'tokens') -> str:
        return self._index_to_token[namespace][index]

    def get_vocab_size(self, namespace: str = 'tokens') -> int:
        return len(self._token_to_index[namespace])

    def __eq__(self, other):
        if isinstance(self, other.__class__):
            return self.__dict__ == other.__dict__
        return False

    def __str__(self) -> str:
        base_string = f"Vocabulary with namespaces:\n"
        non_padded_namespaces = f"\tNon Padded Namespaces: {self._non_padded_namespaces}\n"
        namespaces = [f"\tNamespace: {name}, Size: {self.get_vocab_size(name)} \n"
                      for name in self._index_to_token]
        return " ".join([base_string, non_padded_namespaces] + namespaces)


def read_normal_embedding_file(embeddings_filename: str,  # pylint: disable=invalid-name
                               embedding_dim: int,
                               vocab: ExVocabulary,
                               namespace: str = "tokens") -> torch.FloatTensor:
    words_to_keep = set(vocab.get_index_to_token_vocabulary(namespace).values())
    vocab_size = vocab.get_vocab_size(namespace)
    embeddings = {}

    # First we read the embeddings from the file, only keeping vectors for the words we need.
    logger.info("Reading embeddings from file")
    with open(embeddings_filename, 'rb') as embeddings_file:
        for line in embeddings_file:
            fields = line.decode('utf-8').rstrip().split(' ')
            if len(fields) - 1 != embedding_dim:
                # Sometimes there are funny unicode parsing problems that lead to different
                # fields lengths (e.g., a word with a unicode space character that splits
                # into more than one column).  We skip those lines.  Note that if you have
                # some kind of long header, this could result in all of your lines getting
                # skipped.  It's hard to check for that here; you just have to look in the
                # embedding_misses_file and at the model summary to make sure things look
                # like they are supposed to.
                logger.warning("Found line with wrong number of dimensions (expected %d, was %d): %s",
                               embedding_dim, len(fields) - 1, line)
                continue
            word = fields[0]
            if word in words_to_keep:
                vector = numpy.asarray(fields[1:], dtype='float32')
                embeddings[word] = vector

    if not embeddings:
        raise ValueError("No embeddings of correct dimension found; you probably "
                         "misspecified your embedding_dim parameter, or didn't "
                         "pre-populate your Vocabulary")

    all_embeddings = numpy.asarray(list(embeddings.values()))
    embeddings_mean = float(numpy.mean(all_embeddings))
    embeddings_std = float(numpy.std(all_embeddings))
    # Now we initialize the weight matrix for an embedding layer, starting with random vectors,
    # then filling in the word vectors we just read.
    logger.info("Initializing pre-trained embedding layer")
    embedding_matrix = torch.FloatTensor(vocab_size, embedding_dim).normal_(embeddings_mean,
                                                                            embeddings_std)

    for i in Tqdm.tqdm(range(0, vocab_size)):
        word = vocab.get_token_from_index(i, namespace)

        # If we don't have a pre-trained vector for this word, we'll just leave this row alone,
        # so the word has a random initialization.
        if word in embeddings:
            embedding_matrix[i] = torch.FloatTensor(embeddings[word])
        else:
            logger.debug("Word %s was not found in the embedding file. Initialising randomly.", word)

    # The weight matrix is initialized, so we construct and return the actual Embedding.
    return embedding_matrix


def build_vocab_embeddings(vocab: ExVocabulary, external_embd_path, embd_dim, namespaces=None, saved_path=None):
    if not isinstance(external_embd_path, list):
        external_embd_path = [external_embd_path]

    weight_list = []
    if namespaces is None:
        namespaces = ['tokens'] * len(external_embd_path)
    elif not isinstance(namespaces, list):
        namespaces = [namespaces] * len(external_embd_path)
    else:
        assert len(namespaces) == len(external_embd_path)

    for external_embd_file, namespace in zip(external_embd_path, namespaces):
        weight = read_normal_embedding_file(external_embd_file,
                                            embd_dim, vocab, namespace=namespace)
        weight_list.append(weight)

    if saved_path is not None:
        vocab.save_to_files(saved_path)
        Path(saved_path / "weights").mkdir(parents=True)

        assert len(weight_list) == len(external_embd_path)

        for weight, external_embd_file in zip(weight_list, external_embd_path):
            embd_name = Path(external_embd_file).stem
            torch.save(weight, saved_path / "weights" / embd_name)


def load_vocab_embeddings(saved_path):
    vocab = ExVocabulary.from_files(saved_path)
    weight_dict = {}
    for file in Path(saved_path / "weights").iterdir():
        weight_dict[str(file.name)] = torch.load(file)
        # assert vocab.get_vocab_size('tokens') == int(weight_dict[str(file.name)].size(0))

    return vocab, weight_dict


if __name__ == '__main__':
    d = RandomHashDict(unk_num=10, oov_token=DEFAULT_OOV_TOKEN)
    d.update({'@@PADDING@@': 0})
    d.add_unk_tokens()

    print(d)
    print(d['hello'])
    print(d['yes'])
    print(d['ho'])
    print(d['no'])
    print(d['hello1'])
    print(d['yes2'])
    print(d['ho3'])
    print(d['no4'])
