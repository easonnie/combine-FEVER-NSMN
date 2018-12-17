import itertools
from typing import Dict, List, Tuple
import json

from overrides import overrides
import logging
import torch.tensor
import numpy as np
import random

from wn_featurizer.additional_feature import encode_num_in_ltokens
from allennlp.data.fields import MetadataField

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

from data_util.paragraph_span import ParagraphSpan

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class VCSS_Reader(DatasetReader):
    """
    VCSS mixed Data Reader.
    Remember ths label space is: SUPPORTS(0), REFUTES(1), NOT ENOUGH INFO(2), true(3), false(4), hidden(-2)
    """

    def __init__(self,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 lazy: bool = False,
                 example_filter=None,
                 max_l=None, shuffle_sentences=False,
                 include_relatedness_score=True) -> None:

        super().__init__(lazy=lazy)
        self._token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer(namespace='tokens')}
        self._example_filter = example_filter

        self.max_l = max_l
        self.shuffle_sentences = shuffle_sentences
        self.include_relatedness_score = include_relatedness_score

    @overrides
    def _read(self, data_list):
        logger.info("Reading Fever instances from upstream sampler")
        for example in data_list:
            task = example["task"]

            if task == 'vc':
                label = example["label"]

                if self._example_filter is None:
                    pass
                elif self._example_filter(example):
                    continue

                # We use binary parse here
                premise: List[Tuple[str, float]] = example["evid"]
                hypothesis = example["claim"]

                if len(premise) == 0:
                    premise = [("@@@EMPTY@@@", 0.0)]

                pid = str(example['id'])

                yield self.text_to_instance(premise, hypothesis, pid, label)

            elif task == 'ss':
                label = example["selection_label"]

                if self._example_filter is None:
                    pass
                elif self._example_filter(example):
                    continue

                # We use binary parse here
                premise: List[Tuple[str, float]] = [(example["text"], 1.0)]
                hypothesis = example["query"]

                if premise == "":
                    premise = "@@@EMPTY@@@"

                pid = str(example['selection_id'])

                yield self.text_to_instance(premise, hypothesis, pid, label)

            else:
                raise ValueError("Unkown task!", task)

    @overrides
    def text_to_instance(self,  # type: ignore
                         premise: List[Tuple[str, float]],  # Important type information
                         hypothesis: str,
                         pid: str = None,
                         label: str = None) -> Instance:

        fields: Dict[str, Field] = {}

        premise_tokens_list = []
        premise_prob_values = []

        premise_span_list: List[Tuple[int, int]] = []
        premise_span_prob: List[float] = []

        # sentence_count = len(premise)

        if self.shuffle_sentences:
            # Potential improvement. Shuffle the input sentences. Maybe close this at last several epoch.
            random.shuffle(premise)

        span_start = 0
        for premise_sent, prob in premise:
            cur_premise_tokens = [Token(t) for t in premise_sent.split(' ')]  # Removing code for parentheses in NLI
            span_end = span_start + len(cur_premise_tokens)  #
            premise_span_list.append((span_start, span_end))    # Calculate the span.
            span_start = span_end
            premise_span_prob.append(prob)
            prob_value = np.ones((len(cur_premise_tokens), 1), dtype=np.float32) * prob
            premise_tokens_list.extend(cur_premise_tokens)
            premise_prob_values.append(prob_value)

        premise_prob = np.concatenate(premise_prob_values, axis=0)

        hypothesis_tokens = [Token(t) for t in hypothesis.split(' ')]
        hypothesis_prob = np.ones((len(hypothesis_tokens), 1), dtype=np.float32)

        if self.max_l is not None:
            premise_tokens_list = premise_tokens_list[:self.max_l]
            hypothesis_tokens = hypothesis_tokens[:self.max_l]
            premise_prob = premise_prob[:self.max_l, :]
            hypothesis_prob = hypothesis_prob[:self.max_l, :]

        fields['premise_spans'] = MetadataField(ParagraphSpan(premise_span_list))
        fields['premise_probs'] = MetadataField(premise_span_prob)
        fields['premise'] = TextField(premise_tokens_list, self._token_indexers)    # (t_len, 1)
        fields['hypothesis'] = TextField(hypothesis_tokens, self._token_indexers)

        # We don't give score here because we want the sentence selection and verification to be consistent with each other
        if label in ['true', 'false', 'hidden']:
            # p_t_indicator = np.ones((len(premise_tokens_list), 1), dtype=np.float32)
            # h_t_indicator = np.ones((len(hypothesis_tokens), 0), dtype=np.float32)
            pass
        elif label in ['SUPPORTS', 'REFUTES', 'NOT ENOUGH INFO']:
            # p_t_indicator = np.ones((len(premise_tokens_list), 1), dtype=np.float32)
            # h_t_indicator = np.ones((len(hypothesis_tokens), 0), dtype=np.float32)
            pass
        else:
            raise ValueError("Unknow label", label)

        p_feature_array = np.concatenate([premise_prob], axis=1)
        h_feature_array = np.concatenate([hypothesis_prob], axis=1)

        fields['p_wn_feature'] = ArrayField(p_feature_array)
        fields['h_wn_feature'] = ArrayField(h_feature_array)

        if label:
            fields['label'] = LabelField(label, label_namespace='labels')

        if pid:
            fields['pid'] = IdField(pid)

        return Instance(fields)
