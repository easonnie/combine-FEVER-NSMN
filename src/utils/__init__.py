from datetime import datetime


def get_current_time_str():
    return str(datetime.now().strftime('%Y_%m_%d_%H:%M:%S'))


def get_adv_print_func(filename=None, verbose=False):
    file = None
    if filename is not None:
        file = open(filename, encoding='utf-8', mode='w')

    def the_function(*args, **kwargs):
        if verbose:
            print(*args, **kwargs)
        if file is not None:
            print(file=file, *args, **kwargs)

    return the_function


if __name__ == '__main__':
    log_print = get_adv_print_func('/Users/Eason/RA/FunEver/src/extremelyHard/test.log')
    log_print("hi")