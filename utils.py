import random


def new_batch(data1, data2, batch_size):
  idx_arr = random.sample(range(0, len(data1)), batch_size)
  ret_arr_1 = []
  ret_arr_2 = []
  for i in range(len(idx_arr)):
    ret_arr_1.append(data1[idx_arr[i]])
    ret_arr_2.append(data2[idx_arr[i]])
  return ret_arr_1, ret_arr_2


def next_batch(data1, data2, idx):
  ret_arr_1 = []
  ret_arr_2 = []
  ret_arr_1.append(data1[idx])
  ret_arr_2.append(data2[idx])
  return ret_arr_1, ret_arr_2
