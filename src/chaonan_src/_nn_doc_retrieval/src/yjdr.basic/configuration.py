"""Config file
"""


import os

from chaonan_src._utils.nn_config import ConfigBase


__author__ = 'chaonan99'
__copyright__ = 'Copyright 2018, Haonan Chen'


class Config(ConfigBase):
    num_epoch = 10
    seed = 12
    batch_size = 128
    dev_batch_size = 128
    # experiment_name = 'simple_nn_doc_first_sent'
    lazy = True
    contain_first_sentence = False
    pn_ratio = 1.0
    start_lr = 0.0002
    log_interval = 5

    run_name = 'basic'


config = Config()


def main():
    from IPython import embed; embed(); import os; os._exit(1)
    repr(config)


if __name__ == '__main__':
    main()
