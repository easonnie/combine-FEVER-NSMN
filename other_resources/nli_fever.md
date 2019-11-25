## NLI style FEVER Download Link
Link: [NLI style FEVER dataset](https://www.dropbox.com/s/hylbuaovqwo2zav/nli_fever.zip?dl=0).

## What is in the file?
This file contains the NLI style FEVER dataset introduced in the [**Adversarial NLI paper**](https://arxiv.org/abs/1910.14599).
The dataset is used together with [**SNLI**](https://nlp.stanford.edu/projects/snli/) and [**MultiNLI**](https://www.nyu.edu/projects/bowman/multinli/) to train the backend NLI model in the [**Adversarial NLI**](https://adversarialnli.com/).

## What is the Original FEVER dataset?
Each data point in the original FEVER dataset is a textual claim paired with a label (support / refute / not enough information) depending on whether the claim can be verified by the Wikipedia.
For examples with support and refute labels in the training set and dev set, ground truth location of the evidence in the Wikipedia is also provided. (Please refer to [the original paper](https://arxiv.org/abs/1803.05355) for details)

## What is the difference between the original FEVER and this file?
In the original FEVER setting, the input is a claim and the Wikipedia and the expected output is a label. 
However, this is different from the standard NLI formalization which is basically a *pair-of-sequence to label* problem.
To facilitate NLI-related research take advantage of the FEVER dataset, we pair the claims in the FEVER dataset with **textual evidence** and make it a *pair-of-sequence to label* formatted dataset.

## How is the pairing implemented?
We first applied evidence selection using the method in previous [SOTA fact-checking system](https://arxiv.org/abs/1811.07039) such that each claim will have a collection of potential evidential sentences.
Then, for claims in FEVER dev set, test set and the claims with not-enough-info label in training set, we directly paired them with the concatenation of all selected evidential sentences. 
(Note that for not-enough-info claims in FEVER training set, no ground truth evidence locations are provided in the original dataset.)
For claims in FEVER training set with support and refute label where ground truth evidence locations are provided, we paired them with ground truth textual evidence plus some other randomly sampled evidence from the sentence collection selected by [SOTA fact-checking system](https://arxiv.org/abs/1811.07039).
Therefore, the same claim might got paired with multiple different contexts.
This can help the final NLI model be adaptive to the noisy upstream evidence.

## What is the format?
The train/dev/test data are contained in the three jsonl files. 
The `query` and `context` field correspond to `premise` and `hypothesis` and the `SUPPORT`, `REFUTE`, and `NOT ENOUGH INFO` labels correspond to `ENTAILMENT', `CONTRADICT', and `NEUTRAL` label, respectively, in typical NLI settings.
The `cid` can be mapped back the original FEVER `id` field. (The labels for both dev and test are hidden but you can retrieve the label for dev using the cid and the original FEVER data.)
Finally, you can train your NLI model using this data and get FEVER verification label results. The label accuracy on dev and test will be comparable to the previous fact-checking works and you can submit your entries to [FEVER CodaLab Leaderboard](https://competitions.codalab.org/competitions/18814#results) to report test results.

## Citation
If you used the data in this file, please cite the following paper:
```
@inproceedings{nie2019combining,
  title={Combining Fact Extraction and Verification with Neural Semantic Matching Networks},
  author={Yixin Nie and Haonan Chen and Mohit Bansal},
  booktitle={Association for the Advancement of Artificial Intelligence ({AAAI})},
  year={2019}
}
```