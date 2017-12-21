import residual_calculator as res_calc

ori_res = res_calc.get_residuals('input\procedure_480p_1min_wave\in')
trs_res = res_calc.get_residuals('input\procedure_480p_1min_wave\out')

# Now we want to learn the reflection from ori_res to trs_res
# And then we can apply the learned knowledge to new ori_res and get new trs_res



