from typing import Dict

from overrides import overrides
import logging

from allennlp.data.dataset_readers.dataset_reader import DatasetReader

from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.fields import Field, TextField, LabelField
from allennlp.data.instance import Instance
from allennlp.data import Token

from data_util.customized_field import IdField

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class SSelectorReader(DatasetReader):
    """
    WordNet augmented Data Reader.
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
            label = example["selection_label"]

            if self._example_filter is None:
                pass
            elif self._example_filter(example):
                continue

            # We use binary parse here
            premise = example["text"]
            hypothesis = example["query"]

            if premise == "":
                premise = "@@@EMPTY@@@"

            pid = str(example['selection_id'])

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
            fields['selection_label'] = LabelField(label, label_namespace='labels')

        if pid:
            fields['pid'] = IdField(pid)

        return Instance(fields)
