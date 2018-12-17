# This is created for building data from bert tokenizer.

import itertools
from typing import Dict, List, Tuple
import json

from overrides import overrides
import logging
import torch.tensor
import numpy as np
import random

from neural_modules.bert_servant import BertServant
from wn_featurizer.additional_feature import encode_num_in_ltokens
from allennlp.data.fields import MetadataField

from allennlp.data.dataset_readers.dataset_reader import DatasetReader

from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer, ELMoTokenCharactersIndexer
from allennlp.data.fields import Field, TextField, LabelField, ArrayField
from allennlp.data.instance import Instance
from allennlp.data import Token

from data_util.customized_field import IdField, BertIndexField
from data_util.exvocab import ExVocabulary, read_normal_embedding_file, load_vocab_embeddings, build_vocab_embeddings

# from pathlib import Path
from pathlib import Path
import config
from sample_for_nli.tf_idf_sample_v1_0 import select_sent_for_eval, sample_v1_0


from wn_featurizer import wn_persistent_api

from data_util.paragraph_span import ParagraphSpan

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

MAX_EVIDENCE_SENT_NUM = 5   # We might change this for ground truth evidence


class BertReader(DatasetReader):
    """
    WordNet augmented Data Reader.
    """

    def __init__(self,
                 bert_servant: BertServant,
                 lazy: bool = False,
                 example_filter=None,
                 max_l=60, shuffle_sentences=False) -> None:

        # max_l indicate the max length of each individual sentence.
        # the final concatenation of sentences is 60 * 6 = 5(evid) * 60 + 1(claim) * 60

        super().__init__(lazy=lazy)
        self._example_filter = example_filter
        self.max_l = max_l
        self.shuffle_sentences = shuffle_sentences
        self.bert_servant: BertServant = bert_servant

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
            # first element is the sentence and the second is the upstream semantic relatedness score.
            premise: List[Tuple[str, float]] = example["evid"]
            # truncate premise
            premise = premise[:MAX_EVIDENCE_SENT_NUM]

            hypothesis = example["claim"]

            if len(premise) == 0:
                premise = [("EEMMPPTTYY", 0.0)]

            pid = str(example['id'])

            yield self.text_to_instance(premise, hypothesis, pid, label)

    @overrides
    def text_to_instance(self,  # type: ignore
                         premise: List[Tuple[str, float]],  # Important type information
                         hypothesis: str,
                         pid: str = None,
                         label: str = None) -> Instance:

        fields: Dict[str, Field] = {}

        if self.shuffle_sentences:
            # Potential improvement. Shuffle the input sentences. Maybe close this at last several epoch.
            random.shuffle(premise)

        premise_prob_list = []
        premise_tokens_list = []

        for premise_sent, prob in premise:
            tokenized_cur_sent = self.bert_servant.tokenize(premise_sent, modify_from_corenlp=True)
            # cur_sent_ids = self.bert_servant.tokens_to_ids(tokenized_cur_sent)

            if self.max_l is not None:
                tokenized_cur_sent = tokenized_cur_sent[:self.max_l]    # truncate max length (default 60)

            premise_tokens_list.extend(tokenized_cur_sent)
            prob_value = np.ones((len(tokenized_cur_sent), 1), dtype=np.float32) * prob
            premise_prob_list.append(prob_value)

        premise_prob = np.concatenate(premise_prob_list, axis=0)
        # premise_tokens_id_list = self.bert_servant.tokens_to_ids(premise_tokens_list)

        hypothesis_tokens_list = self.bert_servant.tokenize(hypothesis, modify_from_corenlp=True)

        # print("WTF!!!, p", len(premise_tokens_list))
        # print("WTF!!!, h", len(hypothesis_tokens_list))

        if self.max_l is not None:
            hypothesis_tokens_list = hypothesis_tokens_list[:self.max_l]

        hypothesis_prob = np.ones((len(hypothesis_tokens_list), 1), dtype=np.float32)

        assert len(premise_tokens_list) == len(premise_prob)
        assert len(hypothesis_tokens_list) == len(hypothesis_prob)

        paired_tokens_sequence = ['[CLS]'] + premise_tokens_list + ['[SEP]'] + hypothesis_tokens_list + ['[SEP]']
        token_type_ids = [0] * (2 + len(premise_tokens_list)) + [1] * (1 + len(hypothesis_tokens_list))

        paired_ids_seq = self.bert_servant.tokens_to_ids(paired_tokens_sequence)
        assert len(paired_ids_seq) == len(token_type_ids)
        fields['paired_sequence'] = BertIndexField(np.asarray(paired_ids_seq, dtype=np.int64))
        fields['paired_token_type_ids'] = BertIndexField(np.asarray(token_type_ids, dtype=np.int64))

        premise_span = (1, 1 + len(premise_tokens_list)) # End is exclusive (important for later use)
        hypothesis_span = (premise_span[1] + 1, premise_span[1] + 1 + len(hypothesis_tokens_list))

        assert len(paired_ids_seq) == 1 + (premise_span[1] - premise_span[0]) + 1 + \
               (hypothesis_span[1] - hypothesis_span[0]) + 1

        fields['bert_premise_span'] = MetadataField(premise_span)
        fields['bert_hypothesis_span'] = MetadataField(hypothesis_span)

        fields['premise_probs'] = MetadataField(premise_prob)
        fields['hypothesis_probs'] = MetadataField(hypothesis_prob)

        if label:
            fields['label'] = LabelField(label, label_namespace='labels')

        if pid:
            fields['pid'] = IdField(pid)

        return Instance(fields)
