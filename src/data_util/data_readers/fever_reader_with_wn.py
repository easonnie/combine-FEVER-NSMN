import itertools
from typing import Dict, List
import json

from overrides import overrides
import logging
import torch.tensor
import numpy as np

from allennlp.data.dataset_readers.dataset_reader import DatasetReader

from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer, ELMoTokenCharactersIndexer
from allennlp.data.fields import Field, TextField, LabelField, ArrayField
from allennlp.data.instance import Instance
from allennlp.data import Token

from data_util.customized_field import IdField
from data_util.exvocab import ExVocabulary, read_normal_embedding_file, load_vocab_embeddings, build_vocab_embeddings

# from pathlib import Path
from pathlib import Path
import config
from sample_for_nli.tf_idf_sample_v1_0 import select_sent_for_eval, sample_v1_0

from wn_featurizer import wn_persistent_api

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class WNReader(DatasetReader):
    """
    WordNet augmented Data Reader.
    """

    def __init__(self,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 lazy: bool = False,
                 example_filter=None,
                 wn_p_dict=None, wn_feature_list=wn_persistent_api.default_fn_list,
                 max_l=None) -> None:

        super().__init__(lazy=lazy)
        self._token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer(namespace='tokens')}
        self._example_filter = example_filter
        self.wn_p_dict = wn_p_dict
        if wn_p_dict is None:
            raise ValueError("Need to specify WN feature dict for FEVER Reader.")
        self.wn_feature_list = wn_feature_list
        self.wn_feature_size = len(self.wn_feature_list) * 3
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

        # WN feature dict:
        premise_s = premise.split(' ')
        hypothesis_s = hypothesis.split(' ')

        if self.max_l is not None:
            premise_s = premise_s[:self.max_l]
            hypothesis_s = hypothesis_s[:self.max_l]

        example_feature = wn_persistent_api.compute_wn_features_p_accerate(premise_s,
                                                                           hypothesis_s,
                                                                           self.wn_p_dict)

        p_wn_nparray, h_wn_nparray = wn_persistent_api.wn_raw_feature_to_nparray(
            example_feature,
            self.wn_feature_list)

        assert len(premise_tokens) == p_wn_nparray.shape[0]
        assert len(hypothesis_tokens) == h_wn_nparray.shape[0]

        fields['p_wn_feature'] = ArrayField(p_wn_nparray)
        fields['h_wn_feature'] = ArrayField(h_wn_nparray)

        if label:
            fields['label'] = LabelField(label, label_namespace='labels')

        if pid:
            fields['pid'] = IdField(pid)

        return Instance(fields)
