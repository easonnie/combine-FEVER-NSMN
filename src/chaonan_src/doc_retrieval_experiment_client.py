# -*- coding: utf-8 -*-
"""
If the client cannot connect to the server, mostly it is because the firewall
on the server. Try to close that and you are good to go!
"""

import json

import websocket

from chaonan_src._utils.spcl import spcl
from chaonan_src.doc_retrieval_experiment import DocRetrievalExperiment


class DocRetrievalClient(DocRetrievalExperiment):
    """docstring for DocRetrievalClient"""
    first_round_path = "ws://bvisionserver4.cs.unc.edu:9199/first"
    feed_path = "ws://bvisionserver4.cs.unc.edu:9199/feed"
    second_round_path = "ws://bvisionserver4.cs.unc.edu:9199/second"

    def __init__(self):
        pass

    @classmethod
    def sample_answer_with_priority(cls, d_list, top_k=5):
        ws_first = websocket.create_connection(cls.first_round_path)

        for i in spcl(range(len(d_list))):
        # for i in spcl(range(10)):
            ws_first.send(json.dumps(d_list[i]))
            item = json.loads(ws_first.recv())
            item['predicted_docids'] = \
                list(set([k for k, v \
                            in sorted(item['prioritized_docids'],
                                      key=lambda x: (-x[1], x[0]))][:top_k]))
            d_list[i] = item
        ws_first.close()

    @classmethod
    def feed_sent_file(cls, path):
        """Use absolute path, and it should appear in my repo"""
        ws_feed = websocket.create_connection(cls.feed_path)
        ws_feed.send(path)
        return_msg = ws_feed.recv()
        print(return_msg)
        ws_feed.close()

    @classmethod
    def find_sent_link_with_priority(cls, d_list, top_k=5, predict=False):
        ws_second = websocket.create_connection(cls.second_round_path)
        for i in spcl(range(len(d_list))):
            ws_second.send(json.dumps(d_list[i]))
            item = json.loads(ws_second.recv())
            pids = [it[0] for it in item['prioritized_docids']]
            item['prioritized_docids_aside'] = \
                [it for it in item['prioritized_docids_aside']\
                    if it[0] not in pids]
            if predict:
                porg = \
                    set([k for k, v \
                           in sorted(item['prioritized_docids'],
                                     key=lambda x: (-x[1], x[0]))][:top_k])
                paside = \
                    set([k for k, v \
                            in sorted(item['prioritized_docids_aside'],
                                      key=lambda x: (-x[1], x[0]))][:top_k])
                item['predicted_docids'] = list(porg | paside)
                item['predicted_docids_origin'] = list(porg)
                item['predicted_docids_aside'] = list(paside)
            d_list[i] = item
        ws_second.close()


if __name__ == "__main__":
    from chaonan_src._utils.doc_utils import read_jsonl
    import config

    d_list = read_jsonl(config.FEVER_DEV_JSONL)
    d_list_test = d_list[:20]
    DocRetrievalClient.sample_answer_with_priority(d_list_test)
    DocRetrievalClient.print_eval(d_list_test)

    # DocRetrievalClient.feed_sent_file(
    #     "../../results"
    #     "/sent_retri_nn/2018_07_17_16-34-19_r/dev_scale(0.1).jsonl")
    # DocRetrievalClient.find_sent_link_with_priority(d_list_test, predict=True)
    DocRetrievalExperiment.print_eval(d_list_test)
