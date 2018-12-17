#!/usr/bin/env python
"""spcl is the father of tqdm. It is simpler but works in both command line
and notebook
"""

import time


__all__ = ['spcl']
__author__ = ['chaonan99']


class __SPCLManager(object):
  """docstring for __SPCLManager"""
  number_tracker = 0

  @classmethod
  def register_one(cls):
    cls.number_tracker += 1

  @classmethod
  def unregister_one(cls):
    cls.number_tracker -= 1


def spcl(iter, every=5, timeit=True):
  curr_time = time.time()
  start_time = curr_time
  for i, it in enumerate(iter):
    if hasattr(iter, '__len__'):
      if (i+1) % every == 0 or (i+1) == len(iter):
        prev_time = curr_time
        curr_time = time.time()
        info_str = f"\rProcessed {i+1}/{len(iter)}; "
        if timeit:
          info_str += f"Time {(curr_time - prev_time):.6f}" \
                      f" / {(curr_time - start_time):.6f} s"
        if (i+1) % every == 0:
          print(info_str,
                end='',
                flush=True)
        else:
          print(info_str)
    else:
      if (i+1) % every == 0:
        prev_time = curr_time
        curr_time = time.time()
        info_str = f"\rProcessed {i+1}; "
        if timeit:
          info_str += f"Time {(curr_time - prev_time):.6f}" \
                      f" / {(curr_time - start_time):.6f} s"
        if (i+1) % every == 0:
          print(info_str,
                end='',
                flush=True)
        else:
          print(info_str)
    yield it

  print('\nFinished')


def main():
  for i in spcl(range(100)):
    time.sleep(0.2)

if __name__ == '__main__':
  main()