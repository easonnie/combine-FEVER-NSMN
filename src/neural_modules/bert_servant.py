import torch
from pytorch_pretrained_bert import BertTokenizer, BertModel
from typing import List
import flint.torch_util as torch_util
from allennlp.modules import ScalarMix


def symbol_convert(token):
    if token == '-LRB-':
        return '('
    if token == '-RRB-':
        return ')'
    if token == '-LSB-':
        return '['
    if token == '-RSB-':
        return ']'
    if token == '-LCB-':
        return '{'
    if token == '-RCB-':
        return '}'
    return token


class BertServant(object):
    # The convention in BERT is:
    # (a) For sequence pairs:
    #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
    #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
    # (b) For single sequences:
    #  tokens:   [CLS] the dog is hairy . [SEP]
    #  type_ids: 0   0   0   0  0     0 0

    MASK = '[MASK]'
    SEP = '[SEP]'
    CLS = '[CLS]'

    def __init__(self, bert_type_name='') -> None:
        super().__init__()
        self.bert_type_name = bert_type_name

        self.bert_tokenizer = BertTokenizer.from_pretrained(self.bert_type_name)

        self.bert_model: BertModel = BertModel.from_pretrained(self.bert_type_name)
        self.bert_model.eval()

    def tokenize(self, text: str, modify_from_corenlp=False) -> List[str]:
        if modify_from_corenlp:
            new_text = []
            for t_text in text.split(' '):
                new_text.append(symbol_convert(t_text))
            text = ' '.join(new_text)

        return self.bert_tokenizer.tokenize(text)

    def tokens_to_ids(self, tokenized_text: List[str]) -> List[int]:
        return self.bert_tokenizer.convert_tokens_to_ids(tokenized_text)

    def run_paired_seq(self, batch_size, batch, scalar_mix: ScalarMix=None):
        actual_batch_size = int(batch['paired_sequence'].size(0))
        # num_batch = actual_batch_size // batch_size

        input_seq = batch['paired_sequence']
        input_seq_type_ids = batch['paired_token_type_ids']

        r_out_list = []
        input_chucks = zip(torch.split(input_seq, batch_size, dim=0), torch.split(input_seq_type_ids, batch_size, dim=0))

        for input_seq_c, input_seq_type_ids_c in input_chucks:
            seq_mask, seq_len = torch_util.get_length_and_mask(input_seq_c)

            # Put onto gpus
            with torch.no_grad():
                input_seq_c = input_seq_c.to(next(self.bert_model.parameters()).device)
                input_seq_type_ids_c = input_seq_type_ids_c.to(next(self.bert_model.parameters()).device)
                seq_mask = seq_mask.to(next(self.bert_model.parameters()).device)

                # print(input_seq_c, input_seq_type_ids_c, seq_mask)
                # print(batch['paired_sequence'].size())
                # print(batch['paired_token_type_ids'].size())
                # print(batch)
                bert_layer_out, _ = self.bert_model(input_seq_c, input_seq_type_ids_c, attention_mask=seq_mask)
                bert_layer_outs = bert_layer_out[-4:]
                del bert_layer_out[:-4]

            if ScalarMix is not None:
                r_out_c = sum(bert_layer_outs)
            else:
                r_out_c = scalar_mix(bert_layer_outs)
            r_out_list.append(r_out_c)

        return torch.cat(r_out_list, dim=0)

    def paired_seq_split(self, input_seq, a_span_list, b_span_list):
        assert len(a_span_list) == len(b_span_list)
        a_seq_list = []
        a_l_list = []
        b_seq_list = []
        b_l_list = []
        for b_i, ((a_start, a_end), (b_start, b_end)) in enumerate(zip(a_span_list, b_span_list)):
            a_seq_list.append(input_seq[b_i, a_start:a_end])
            a_l_list.append(a_end - a_start)
            b_seq_list.append(input_seq[b_i, b_start:b_end])
            b_l_list.append(b_end - b_start)

        a_seq = torch_util.pack_list_sequence(a_seq_list, a_l_list)
        b_seq = torch_util.pack_list_sequence(b_seq_list, b_l_list)
        a_l = torch.tensor(a_l_list).to(input_seq.device)
        b_l = torch.tensor(b_l_list).to(input_seq.device)
        return a_seq, a_l, b_seq, b_l


if __name__ == '__main__':
    print(BertServant.MASK)