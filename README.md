# Combine-FEVER-NSMN
This repository provides the implementation for the paper [Combining Fact Extraction and Verification with Neural Semantic Matching Networks](https://arxiv.org/abs/1811.07039).

## Requirement
* Python 3.6
* pytorch 0.4.1
* allennlp 0.7.1
* sqlitedict
* wget
* tqdm

## Preparation
1. Setup the python environment and download the required package listed above.
2. Run the preparation script.
```bash
source setup.sh
bash ./scripts/prepare.sh
```
The script will download all the required data, the auxiliary packages and files.
After preparation, your project folder should be similar to the one below.
```bash
# Coming.
```

(More coming...)
## More coming...


## Citation
If you find this implementation helpful, please consider citing:
```
@inproceedings{nie2018combining,
  title={Combining Fact Extraction and Verification with Neural Semantic Matching Networks},
  author={Nie, Yixin and Chen, Haonan and Bansal, Mohit},
  booktitle={Association for the Advancement of Artificial Intelligence ({AAAI})},
  year={2019}
}
```