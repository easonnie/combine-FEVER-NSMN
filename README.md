# Combine-FEVER-NSMN
This repository provides the implementation for the paper [Combining Fact Extraction and Verification with Neural Semantic Matching Networks](https://arxiv.org/abs/1811.07039) (AAAI 2019 and EMNLP-FEVER Shared Task Rank-1 System).

## Requirement
* Python 3.6
* pytorch 0.4.1
* allennlp 0.7.1
* sqlitedict
* wget
* flashtext
* pexpect
* fire
* inflection

Try to install the package as the order above.
Previous version of pytorch can be find at [legacy pytorch](https://pytorch.org/get-started/previous-versions/).

## Preparation
1. Setup the python environment and download the required package listed above.
2. Run the preparation script.
```bash
source setup.sh
bash ./scripts/prepare.sh
```
The script will download all the required data, the auxiliary packages and files.

3. Tokenize the dataset and build wiki document database for easy and fast access and query.
```bash
python src/pipeline/prepare_data.py tokenization        # Tokenization
python src/pipeline/prepare_data.py build_database      # Build document database. (This might take a while)
```

After preparation, the following folder should contain similar files as listed below.
```bash
data
├── fever
│   ├── license.html
│   ├── shared_task_dev.jsonl
│   ├── shared_task_test.jsonl
│   └── train.jsonl
├── fever.db
├── id_dict.jsonl
├── license.html
├── sentence_tokens.json
├── tokenized_doc_id.json
├── tokenized_fever
│   ├── shared_task_dev.jsonl
│   └── train.jsonl
├── vocab_cache
│   └── nli_basic
│       ├── labels.txt
│       ├── non_padded_namespaces.txt
│       ├── tokens.txt
│       ├── unk_count_namespaces.txt
│       └── weights
│           └── glove.840B.300d
├── wiki-pages
│   ├── wiki-001.jsonl
│   ├── ... ...
│   └── wiki-109.jsonl
└── wn_feature_p
    ├── ant_dict
    ├── em_dict
    ├── em_lemmas_dict
    ├── hyper_lvl_dict
    ├── hypernym_stems_dict
    ├── hypo_lvl_dict
    └── hyponym_stems_dict
```
```bash
dep_packages
├── DrQA
└── stanford-corenlp-full-2017-06-09
```
```bash
results
└── chaonan99
```
```bash
saved_models
├── saved_nli_m
├── nn_doc_selector
└── saved_sselector
```

## Automatic pipeline procedure.
Running the pipeline system on the dev set with the code below:
```bash
python src/pipeline/auto_pipeline.py
```
Note that this pipeline is the (SotA) model in the AAAI paper. 
For EMNLP-FEVER Shared Task version, please refer to `src/nli/mesim_wn_simi_v1_3.py` and `src/pipeline/pipeline_process.py`.

## Other Resources
To facilitate the NLI-focused research in making use of the original FEVER dataset, the repo also comes with **a NLI style FEVER dataset**, 
More information can be found in [**nli_fever**](https://github.com/easonnie/combine-FEVER-NSMN/blob/master/other_resources/nli_fever.md) and [**Adversarial NLI**](https://adversarialnli.com/).

## Citation
If you find this implementation helpful, please consider citing:
```
@inproceedings{nie2019combining,
  title={Combining Fact Extraction and Verification with Neural Semantic Matching Networks},
  author={Yixin Nie and Haonan Chen and Mohit Bansal},
  booktitle={Association for the Advancement of Artificial Intelligence ({AAAI})},
  year={2019}
}
```