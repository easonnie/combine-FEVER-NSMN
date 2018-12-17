from sample_for_nli.tf_idf_sample_v1_0 import convert_evidence2scoring_format
from utils import common, c_scorer
import config
from tqdm import tqdm
import six
import utils
from utils.c_scorer import is_correct_label, is_strictly_correct, evidence_macro_recall, evidence_macro_precision, \
    check_doc_id_correct, check_sent_correct
from utils.confusion_matrix_analysis import report

label_to_id = {
    'SUPPORTS': 0,
    'REFUTES': 1,
    'NOT ENOUGH INFO': 2
}

label_list = ['SUPPORTS', 'REFUTES', 'NOT ENOUGH INFO']

def fever_score_ana(predictions, actual=None, max_evidence=5, mode=None, error_analysis_file=None,
                    verbose=False):
    '''
    This is a important function for different scoring.
    Pass in different parameter in mode for specific score.

    :param verbose:
    :param predictions:
    :param actual:
    :param max_evidence:
    :param mode:
    :return:
    '''

    log_print = utils.get_adv_print_func(error_analysis_file, verbose=verbose)

    correct = 0
    strict = 0
    error_count = 0

    macro_precision = 0
    macro_precision_hits = 0

    macro_recall = 0
    macro_recall_hits = 0

    error_list = []

    stat_dict = {
        'pred_list': [],
        'true_list': [],
    }

    # ana_f = None
    # if error_analysis_file is not None:
    #     ana_f = open(error_analysis_file, mode='w')

    if mode is not None:
        key_list = []
        for key in mode.keys():
            key_list.append(key)

        for key in key_list:
            mode[key + '_hits'] = 0

    for idx, instance in enumerate(predictions):
        if mode['standard']:
            assert 'predicted_evidence' in instance.keys(), 'evidence must be provided for the prediction'

            # If it's a blind test set, we need to copy in the values from the actual data
            if 'evidence' not in instance or 'label' not in instance:
                assert actual is not None, 'in blind evaluation mode, actual data must be provided'
                assert len(actual) == len(predictions), 'actual data and predicted data length must match'
                assert 'evidence' in actual[idx].keys(), 'evidence must be provided for the actual evidence'
                instance['evidence'] = actual[idx]['evidence']
                instance['label'] = actual[idx]['label']

            assert 'evidence' in instance.keys(), 'gold evidence must be provided'

            if is_correct_label(instance):
                correct += 1.0

                if is_strictly_correct(instance, max_evidence):
                    strict += 1.0

                # if not is_strictly_correct(instance, max_evidence):
                #     error_list.append(instance)
            # else:
            #     error_list.append(instance)

            macro_prec = evidence_macro_precision(instance, max_evidence)
            macro_precision += macro_prec[0]
            macro_precision_hits += macro_prec[1]

            macro_rec = evidence_macro_recall(instance, max_evidence)
            macro_recall += macro_rec[0]
            macro_recall_hits += macro_rec[1]

            # if check_sent_correct(instance) and not is_correct_label(instance):
            if check_sent_correct(instance):
                error_list.append(instance)
                append_instance(instance, stat_dict)

        if mode is not None:
            if 'check_doc_id_correct' in mode and mode['check_doc_id_correct']:
                if check_doc_id_correct(instance):
                    mode['check_doc_id_correct_hits'] += 1
                else:
                    # error_count += 1
                    log_print(instance)

            if 'check_sent_id_correct' in mode and mode['check_sent_id_correct']:
                if check_sent_correct(instance):
                    mode['check_sent_id_correct_hits'] += 1
                else:
                    # error_count += 1
                    log_print(instance)

    log_print("Error count:", error_count)
    total = len(predictions)

    log_print("Total:", total)
    print("Total:", total)
    log_print("Strict:", strict)
    print("Strict:", strict)

    for k, v in mode.items():
        if k.endswith('_hits'):
            log_print(k, v, v / total)
            print(k, v, v / total)

    strict_score = strict / total
    acc_score = correct / total

    pr = (macro_precision / macro_precision_hits) if macro_precision_hits > 0 else 1.0
    rec = (macro_recall / macro_recall_hits) if macro_recall_hits > 0 else 0.0

    f1 = 2.0 * pr * rec / (pr + rec)

    report(stat_dict['true_list'], stat_dict['pred_list'], [0, 1, 2], label_list)

    return strict_score, acc_score, pr, rec, f1, error_list


def format_printing(item):
    print("-" * 50)
    print("Claim:", item['claim'])
    print("Evidence:", item['evid'])
    print("Pred Label:", item['predicted_label'])
    print("Pred Evid:", item['predicted_evidence'])
    # print("Pred Evid F:", item['predicted_sentids'])
    print("Label:", item['label'])
    print("Evid:", item['evidence'])
    print("-" * 50)


def append_instance(instance, stat_dict):
    stat_dict['pred_list'].append(label_to_id[instance['predicted_label']])
    stat_dict['true_list'].append(label_to_id[instance['label']])


if __name__ == '__main__':
    dev_results_list = common.load_jsonl(config.RESULT_PATH / "nli_results/pipeline_results_1.jsonl")
    # print(len(dev_results_list))

    eval_mode = {'check_sent_id_correct': True, 'standard': True}
    c_scorer.delete_label(dev_results_list)
    strict_score, acc_score, pr, rec, f1, error_list = fever_score_ana(dev_results_list,
                                                                       common.load_jsonl(config.FEVER_DEV_JSONL),
                                                                       mode=eval_mode, verbose=False)
    print(strict_score, acc_score, pr, rec, f1)

    print(len(error_list))

    for item in error_list[:100]:
        format_printing(item)

