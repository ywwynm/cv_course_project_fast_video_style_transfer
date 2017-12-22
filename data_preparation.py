"""
Generate residuals data file so that we can read it lately instead of computing each time
"""

import residual_calculator as res_calc
import numpy as np


ori_dir = 'input\procedure_480p_1min_wave\in'
trs_dir = 'input\procedure_480p_1min_wave\out'
data_dir = 'input\procedure_480p_1min_wave\data'

ori_res_train = res_calc.get_residuals(ori_dir, 0, 1200)
trs_res_train = res_calc.get_residuals(trs_dir, 0, 1200)

np.save(data_dir + '\ori_res_train.npy', ori_res_train)
np.save(data_dir + '\\trs_res_train.npy', trs_res_train)

ori_res_test = res_calc.get_residuals(ori_dir, 1200, 1798)
trs_res_test = res_calc.get_residuals(trs_dir, 1200, 1798)

np.save(data_dir + '\ori_res_test.npy', ori_res_test)
np.save(data_dir + '\\trs_res_test.npy', trs_res_test)

