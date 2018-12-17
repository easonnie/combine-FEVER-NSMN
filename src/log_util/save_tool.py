import os
import config
from datetime import datetime


def gen_file_prefix(model_name, date=None):
    date_now = datetime.now().strftime("%m-%d-%H:%M:%S") if not date else date
    file_path = os.path.join(config.PRO_ROOT / 'saved_models' / '_'.join((date_now, model_name)))
    if not os.path.exists(file_path):
        os.makedirs(file_path)
    return file_path, date_now


def get_cur_time_str():
    date_now = datetime.now().strftime("%m-%d[%H:%M:%S]")
    return date_now


if __name__ == "__main__":
    # print(gen_file_prefix("this_is_my_model."))
    print(get_cur_time_str())