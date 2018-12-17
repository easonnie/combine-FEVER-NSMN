import itertools
from typing import Dict, List
import json

from overrides import overrides
import logging
import torch.tensor

from allennlp.data.dataset_readers.dataset_reader import DatasetReader

from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer, ELMoTokenCharactersIndexer
from allennlp.data.fields import Field, TextField, LabelField
from allennlp.data.instance import Instance
from allennlp.data import Token

from data_util.customized_field import IdField
from data_util.exvocab import ExVocabulary, read_normal_embedding_file, load_vocab_embeddings, build_vocab_embeddings

# from pathlib import Path
from pathlib import Path
import config
from sample_for_nli.tf_idf_sample_v1_0 import select_sent_for_eval, sample_v1_0

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class BasicReader(DatasetReader):
    """
    This is a customized SNLI Reader for original NLI format dataset reading.
    """

    def __init__(self,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 lazy: bool = False,
                 example_filter=None,
                 max_l=None) -> None:

        super().__init__(lazy=lazy)
        self._token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer(namespace='tokens')}
        self._example_filter = example_filter
        self.max_l = max_l

    @overrides
    def _read(self, data_list):
        logger.info("Reading Fever instances from upstream sampler")
        for example in data_list:
            label = example["label"]

            if self._example_filter is None:
                pass
            elif self._example_filter(example):
                continue

            # We use binary parse here
            premise = example["evid"]
            hypothesis = example["claim"]

            if premise == "":
                premise = "@@@EMPTY@@@"

            pid = str(example['id'])

            yield self.text_to_instance(premise, hypothesis, pid, label)

    @overrides
    def text_to_instance(self,  # type: ignore
                         premise: str,
                         hypothesis: str,
                         pid: str = None,
                         label: str = None) -> Instance:

        fields: Dict[str, Field] = {}

        premise_tokens = [Token(t) for t in premise.split(' ')]  # Removing code for parentheses in NLI
        hypothesis_tokens = [Token(t) for t in hypothesis.split(' ')]

        if self.max_l is not None:
            premise_tokens = premise_tokens[:self.max_l]
            hypothesis_tokens = hypothesis_tokens[:self.max_l]

        fields['premise'] = TextField(premise_tokens, self._token_indexers)
        fields['hypothesis'] = TextField(hypothesis_tokens, self._token_indexers)

        if label:
            fields['label'] = LabelField(label, label_namespace='labels')

        if pid:
            fields['pid'] = IdField(pid)

        return Instance(fields)


def fever_build_vocab(d_list, unk_token_num=None) -> ExVocabulary:
    if unk_token_num is None:
        unk_token_num = {'tokens': 2600}

    token_indexers = {
        'tokens': SingleIdTokenIndexer(namespace='tokens'), # This is the raw tokens
        'elmo_chars': ELMoTokenCharactersIndexer(namespace='elmo_characters')   # This is the elmo_characters
    }

    nli_dataset_reader = BasicReader(token_indexers=token_indexers)

    # for in_file in d_list:
    instances = nli_dataset_reader.read(d_list)

    whole_vocabulary = ExVocabulary.from_instances(instances, unk_token_num=unk_token_num)

    print(whole_vocabulary.get_vocab_size('tokens'))  # 122827
    print(type(whole_vocabulary.get_token_to_index_vocabulary('tokens')))

    return whole_vocabulary


def build_fever_vocab_with_embeddings_and_save():
    # The first version of build vocab, which is build the basic vocabulary.
    resample_time = 1
    d_list = []

    # Sampled Training set
    input_file = config.T_FEVER_TRAIN_JSONL
    additional_file = config.RESULT_PATH / "sent_retri/2018_07_05_17:17:50_r/train.jsonl"
    for _ in range(resample_time):
        sampled_d_list = sample_v1_0(input_file, additional_file, tokenized=True)
        d_list.extend(sampled_d_list)

    # Dev set
    input_file = config.T_FEVER_DEV_JSONL
    additional_file = config.RESULT_PATH / "sent_retri/2018_07_05_17:17:50_r/dev.jsonl"
    dev_d_list = select_sent_for_eval(input_file, additional_file, tokenized=True)

    d_list.extend(dev_d_list)

    vocab = fever_build_vocab(d_list)
    print(vocab)

    build_vocab_embeddings(vocab, config.DATA_ROOT / "embeddings/glove.840B.300d.txt",
                           embd_dim=300, saved_path=config.DATA_ROOT / "vocab_cache" / "nli_basic")


if __name__ == '__main__':
    # nli_path_list = []
    # for in_file in (config.DATA_ROOT / "mnli").iterdir():
    #     nli_path_list.append(in_file)
    #
    # for in_file in (config.DATA_ROOT / "snli").iterdir():
    #     nli_path_list.append(in_file)
    #
    # build_fever_vocab_with_embeddings_and_save()

    # build_vocab_embeddings(vocab, config.DATA_ROOT / "embeddings/glove.840B.300d.txt",
    #                        embd_dim=300, saved_path=config.DATA_ROOT / "vocab_cache" / "nli")

    vocab, weight_dict = load_vocab_embeddings(config.DATA_ROOT / "vocab_cache" / "nli_basic")

    print(weight_dict)
    print(vocab)

    # print(vocab.get_vocab_size())
    # vocab.change_token_with_index_to_namespace('hidden', -2, namespace='labels')
    #
    # print(vocab.get_token_to_index_vocabulary('labels'))
    # print(vocab.get_index_to_token_vocabulary('labels'))
    # print()
    # print(weight_dict['glove.840B.300d'].size())
#