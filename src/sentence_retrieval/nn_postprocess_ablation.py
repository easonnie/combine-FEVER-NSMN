from sample_for_nli.tf_idf_sample_v1_0 import convert_evidence2scoring_format
from utils import common, c_scorer
import config
from tqdm import tqdm


def score_converter_scaled(org_data_file, full_sent_list, scale_prob=0.5, delete_prob=True):
    """
    :param org_data_file:
    :param full_sent_list: append full_sent_score list to evidence of original data file
    :param delete_prob: delete the probability for sanity check
    :param scale_prob:  0.5
    :return:
    """
    d_list = common.load_jsonl(org_data_file)
    augmented_dict = dict()
    print("Build selected sentences file:", len(full_sent_list))
    for sent_item in tqdm(full_sent_list):
        selection_id = sent_item['selection_id']  # The id for the current one selection.
        org_id = int(selection_id.split('<##>')[0])
        if org_id in augmented_dict:
            augmented_dict[org_id].append(sent_item)
        else:
            augmented_dict[org_id] = [sent_item]

    for item in d_list:
        if int(item['id']) not in augmented_dict:
            # cur_predicted_sentids = []
            cur_adv_predicted_sentids = []
        else:
            # cur_predicted_sentids = []  # formating doc_id + c_score.SENTLINT + line_number
            cur_adv_predicted_sentids = []
            sents = augmented_dict[int(item['id'])]
            # Modify some mechaism here to selection sentence whether by some score or label
            for sent_i in sents:
                if sent_i['prob'] >= scale_prob:
                    cur_adv_predicted_sentids.append((sent_i['sid'], sent_i['score'], sent_i['prob']))
                # del sent_i['prob']

            cur_adv_predicted_sentids = sorted(cur_adv_predicted_sentids, key=lambda x: -x[1])

        item['scored_sentids'] = cur_adv_predicted_sentids[:5]  # Important sentences for scaling training. Jul 21.
        item['predicted_sentids'] = [sid for sid, _, _ in item['scored_sentids']][:5]
        item['predicted_evidence'] = convert_evidence2scoring_format(item['predicted_sentids'])
        item['predicted_label'] = item['label']  # give ground truth label

    # Removing all score and prob
    if delete_prob:
        for sent_item in full_sent_list:
            if 'score' in sent_item.keys():
                del sent_item['score']
                del sent_item['prob']

    return d_list


def delete_gold_label(d_list):
    for item in d_list:
        if 'label' in item:
            del item['label']
        if 'evidence' in item:
            del item['evidence']


if __name__ == '__main__':
    # IN_FILE = config.RESULT_PATH / "sent_retri_nn/2018_07_17_15-11-11_r/dev_sent.jsonl"
    # IN_FILE = config.RESULT_PATH / "sent_retri_nn/2018_07_17_15-52-19_r/dev_sent.jsonl"
    IN_FILE = config.RESULT_PATH / "sent_retri_nn/2018_07_20_15-17-59_r/dev_sent.jsonl"
    # IN_FILE = config.RESULT_PATH / "sent_retri_nn/2018_07_20_15-17-59_r/train_sent.jsonl"
    dev_sent_result_lsit = common.load_jsonl(IN_FILE)
    dev_results_list = score_converter_scaled(config.T_FEVER_DEV_JSONL, dev_sent_result_lsit, scale_prob=0.1)
    # dev_results_list = score_converter_scaled(config.T_FEVER_TRAIN_JSONL, dev_sent_result_lsit, scale_prob=0.1)

    common.save_jsonl(dev_results_list, config.RESULT_PATH / "sent_retri_nn/2018_07_20_15-17-59_r/dev_scale(0.1).jsonl")

    # for item in dev_results_list:
    #     print(item['scored_sentids'])

    # common.save_jsonl(dev_results_list, "/Users/Eason/RA/FunEver/results/sent_retri_nn/2018_07_17_16-34-19_r/dev_scale(0.1).jsonl")
    # common.save_jsonl(dev_results_list, "/Users/Eason/RA/FunEver/results/sent_retri_nn/2018_07_17_16-34-19_r/dev_scale(0.1).jsonl")

    # eval_mode = {'check_doc_id_correct': True, 'check_sent_id_correct': True, 'standard': True}
    eval_mode = {'check_sent_id_correct': True, 'standard': True}
    # c_scorer.delete_label(dev_results_list)
    strict_score, acc_score, pr, rec, f1 = c_scorer.fever_score(dev_results_list,
                                                                common.load_jsonl(config.FEVER_DEV_UNLABELED_JSONL),
                                                                mode=eval_mode, verbose=False)
    print(strict_score, acc_score, pr, rec, f1)


    # total = len(dev_results_list)
    # hit = eval_mode['check_sent_id_correct_hits']
    # tracking_score = hit / total
    #
    # print(f"Dev(fever_score/pr/rec/f1):{strict_score}/{pr}/{rec}/{f1}")
    # print(f"Tracking score:", f"{tracking_score}")
    # eval_mode = {'check_sent_id_correct': True, 'standard': True}
    # delete_gold_label(dev_results_list)
    # strict_score, acc_score, pr, rec, f1, error_list = c_scorer.fever_score_analysis(dev_results_list,
    #                                                                                  common.load_jsonl(config.T_FEVER_DEV_JSONL),
    #                                                                                  mode=eval_mode, verbose=False)
    # print(strict_score, acc_score, pr, rec, f1)
    #
    # empty, total = c_scorer.nei_stats(dev_results_list, common.load_jsonl(config.T_FEVER_DEV_JSONL))
    # print(empty, total, empty / total)
    # print(len(error_list))
    #
    # for item in error_list[:100]:
    #     print("ID:", item['id'])
    #     print("Claim:", item['claim'])
    #     print("Label", item['label'])
    #     print("Evidence:", item['evidence'])
    #     print("Scored_sentids:", item['scored_sentids'])
    #     print("-" * 50)

    # Total: 19998
    # Strict: 18134.0
    # check_sent_id_correct_hits
    # 18134
    # 0.9067906790679068
    # standard_hits
    # 0
    # 0.0
    # 0.9067906790679068
    # 1.0
    # 0.37073957395735513
    # 0.8601860186018602
    # 0.5181547934144295