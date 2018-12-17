from chaonan_src.doc_retrieval_experiment import DocRetrievalExperimentTwoStep
from chaonan_src._doc_retrieval.item_rules_spiral import ItemRuleBuilderSpiral, ItemRuleBuilderNoPageview
from chaonan_src._doc_retrieval.item_rules_test import ItemRuleBuilderTest
from utils import common, c_scorer
import nn_doc_retrieval.disabuigation_training as disamb
import config


class HAONAN_DOCRETRI_OBJECT:
    def __init__(self):
        self.instance = None


def init_haonan_docretri_object(object, method='pageview'):
    item_rb_selector = {
        'word_freq': ItemRuleBuilderTest,
        'pageview': ItemRuleBuilderSpiral,
        'nopageview': ItemRuleBuilderNoPageview,
    }[method]
    if object.instance is None:
        object.instance = DocRetrievalExperimentTwoStep(item_rb_selector())


def first_doc_retrieval(retri_object, in_file, method='pageview', top_k=100):
    # doc_exp = DocRetrievalExperiment()
    init_haonan_docretri_object(retri_object, method=method)
    d_list = common.load_jsonl(in_file)
    retri_object.instance.sample_answer_with_priority(d_list, top_k=top_k)
    return d_list


def prepare_data_only_page_view(tokenized_file, eval_file, doc_retrieval_output_file):
    """
    This method prepare document retrieval data using only page view.
    :return:
    """
    doc_retrieval_method = 'pageview'
    print("Method:", doc_retrieval_method)

    haonan_docretri_object = HAONAN_DOCRETRI_OBJECT()

    doc_retrieval_result_list = first_doc_retrieval(haonan_docretri_object, tokenized_file,
                                                    method=doc_retrieval_method, top_k=100)
    eval_list = common.load_jsonl(eval_file)

    disamb.item_resorting(doc_retrieval_result_list)

    print("Evaluating 1st Doc Retrieval")
    eval_mode = {'check_doc_id_correct': True, 'standard': False}
    print(c_scorer.fever_score(doc_retrieval_result_list, eval_list, mode=eval_mode, verbose=False))
    print("Max_doc_num_5:", c_scorer.fever_doc_only(doc_retrieval_result_list, eval_list, max_evidence=5))
    print("Max_doc_num_10:", c_scorer.fever_doc_only(doc_retrieval_result_list, eval_list, max_evidence=10))
    print("Max_doc_num_15:", c_scorer.fever_doc_only(doc_retrieval_result_list, eval_list, max_evidence=15))
    print("Max_doc_num_20:", c_scorer.fever_doc_only(doc_retrieval_result_list, eval_list, max_evidence=20))
    # First Document retrieval End.
    common.save_jsonl(doc_retrieval_result_list, doc_retrieval_output_file)


if __name__ == '__main__':
    prepare_data_only_page_view(config.T_FEVER_DEV_JSONL,
                                config.T_FEVER_DEV_JSONL,
                                config.PRO_ROOT / "results/doc_retri/std_upstream_data_using_pageview/dev_doc.jsonl")

    # prepare_data_only_page_view(config.T_FEVER_TRAIN_JSONL,
    #                             config.T_FEVER_TRAIN_JSONL,
    #                             config.PRO_ROOT / "results/doc_retri/std_upstream_data_using_pageview/train_doc.jsonl")