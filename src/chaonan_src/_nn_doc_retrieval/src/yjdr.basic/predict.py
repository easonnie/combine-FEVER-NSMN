#!/usr/bin/env python
# -*- coding: utf-8 -*-


from tqdm import tqdm

import torch
import torch.nn.functional as F


__author__ = ['chaonan99', 'yixin1']


def hidden_eval(model, data_iter, dev_data_list):
    # select < (-.-) > 0
    # non-select < (-.-) > 1
    # hidden < (-.-) > -2

    with torch.no_grad():
        id2label = {
            0: "true",
            1: "false",
            -2: "hidden"
        }

        print("Evaluating ...")
        model.eval()
        totoal_size = 0
        y_pred_logits_list, y_pred_prob_list, y_id_list = [], [], []

        for batch_idx, batch in enumerate(tqdm(data_iter)):
            out = model(batch)
            prob = F.softmax(out, dim=1)

            y = batch['selection_label']
            y_id_list.extend(list(batch['pid']))

            y_pred_logits_list.extend(out[:, 0].tolist())
            y_pred_prob_list.extend(prob[:, 0].tolist())

            totoal_size += y.size(0)

        assert len(y_id_list) == len(dev_data_list)
        assert len(y_pred_logits_list) == len(dev_data_list)

        for i in range(len(dev_data_list)):
            assert str(y_id_list[i]) == str(dev_data_list[i]['selection_id'])
            # Matching id

            dev_data_list[i]['score'] = y_pred_logits_list[i]
            dev_data_list[i]['prob'] = y_pred_prob_list[i]
            # Reset neural set

        print('total_size:', totoal_size)

    return dev_data_list


def toy_test():
    import config
    import nn_doc_retrieval.disabuigation_training as disamb
    from utils import c_scorer
    from data import DocIDCorpus
    from model import Model


    dev_upstream_file = config.RESULT_PATH / \
                        "doc_retri_bls/docretri.basic.nopageview/dev_toy.jsonl"
    corpus = DocIDCorpus(dev_upstream_file, train=False)
    corpus.initialize()

    model = Model(weight=corpus.weight_dict['glove.840B.300d'],
                  vocab_size=corpus.vocab.get_vocab_size('tokens'),
                  embedding_dim=300, max_l=160, num_of_class=2)

    eval_iter = corpus.get_batch(10)
    complete_upstream_data = hidden_eval(model,
                                         eval_iter,
                                         corpus.complete_upstream_data)
    disamb.enforce_disabuigation_into_retrieval_result_v0( \
        complete_upstream_data, corpus.d_list)
    oracle_score, pr, rec, f1 = c_scorer.fever_doc_only(corpus.d_list,
                                                        corpus.d_list,
                                                        max_evidence=5)

    from IPython import embed; embed(); import os; os._exit(1)


def main():
    toy_test()


if __name__ == '__main__':
    main()