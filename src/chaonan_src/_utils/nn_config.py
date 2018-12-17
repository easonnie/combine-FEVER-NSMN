"""Config file
"""


import logging
import os
from logging.handlers import RotatingFileHandler


__author__ = 'chaonan99'
__copyright__ = "Copyright 2018, Haonan Chen"


class ConfigBase(object):
  """Create experiment directory"""

  def initialize_exp(self):
    ## Check directory layout
    exp_par_dir, exp_name = os.path.split(self.exp_dir)
    # from IPython import embed; embed(); os._exit(1)
    proj_dir, src_name = os.path.split(exp_par_dir)
    assert src_name == 'src', "Layout sould be src/exp_name"
    dump_dir = os.path.join(proj_dir, 'dump')
    assert os.path.exists(dump_dir), "Should have a dump dir under project dir"
    dump_exp_dir = os.path.join(dump_dir, exp_name)
    if not os.path.exists(dump_exp_dir):
      os.makedirs(dump_exp_dir)
    rel_dump_exp_dir = os.path.relpath(dump_exp_dir)
    if os.path.exists('./dump'):
      os.remove('./dump')
    os.symlink(rel_dump_exp_dir, './dump')

  @property
  def exp_dir(self):
    """Does this work if inherited somewhere else?"""
    # return os.path.dirname(os.path.abspath(__file__))
    # print(os.getcwd())
    return os.getcwd()

  @property
  def save_path(self):
    return os.path.join('dump', self.run_name) if 'dump' not in self.run_name \
                                               else self.run_name

  @property
  def model_save_path(self):
    return os.path.join(self.save_path, 'model.pt')

  @property
  def experiment_name(self):
    return os.path.split(self.exp_dir)[1]

  def get_logger(self):
    log_file_dir = self.save_path
    log_file_name = os.path.join(log_file_dir, 'train.log')
    if not os.path.exists(log_file_dir):
      os.makedirs(log_file_dir)
    logger = logging.getLogger()
    # Debug = write everything
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter('%(asctime)s :: %(levelname)s :: %(message)s')
    file_handler = RotatingFileHandler(log_file_name, 'a', 1000000, 1)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    steam_handler = logging.StreamHandler()
    steam_handler.setLevel(logging.INFO)
    logger.addHandler(steam_handler)

    return logger

  @classmethod
  def all_members(cls):
    return vars(cls)

  def __repr__(self):
    members = self.all_members()
    return repr({k:v for k, v in members.items() if not k.startswith('__')})


def main():
  config = Config()
  config.initialize_exp()


if __name__ == '__main__':
  main()