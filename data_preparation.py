"""
Generate residuals data file so that we can read it lately instead of computing each time
"""

import residual_calculator as res_calc
import numpy as np


ori_dir = 'input\procedure_208p_10min_wave\in'
trs_dir = 'input\procedure_208p_10min_wave\out'
data_dir = 'input\procedure_208p_10min_wave\data'

# 3597 frames, index will be 0-3596, 3596 residuals
ori_res_train = res_calc.get_residuals(ori_dir, 1, 15001)  # frame0 to frame3000, 15000 residuals
trs_res_train = res_calc.get_residuals(trs_dir, 1, 15001)

np.save(data_dir + '\ori_res_train.npy', ori_res_train)
np.save(data_dir + '\\trs_res_train.npy', trs_res_train)

ori_res_test = res_calc.get_residuals(ori_dir, 15001, 17982)  # frame15000 to frame17982, 2981 residuals
trs_res_test = res_calc.get_residuals(trs_dir, 15001, 17982)

np.save(data_dir + '\ori_res_test.npy', ori_res_test)
np.save(data_dir + '\\trs_res_test.npy', trs_res_test)




