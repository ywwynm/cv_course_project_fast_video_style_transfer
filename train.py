import optimize
import residual_calculator
import os
import transform_video as trsv

# mkdirs
if not os.path.exists('output/video/'): os.makedirs('output/video/')

train_name = 'wave_208p_2min_wave'

trsv_model_path_name = 'fast_style_transfer/models/wave.ckpt'
trsv_ori_video_path_name = 'input/output2_t_312.mp4'
trsv_output_video_path = 'output/video/' + train_name + '/'
if not os.path.exists(trsv_output_video_path): os.makedirs(trsv_output_video_path)
trsv_output_video_path_name = trsv_output_video_path + 'output.mp4'
trsv_video_frames_path = 'frames/' + train_name + '/'  # will be created by fast_style_transfer

res_ori_frames_dir = trsv_video_frames_path + '/in'
res_trs_frames_dir = trsv_video_frames_path + '/out'
res_npy_store_dir = 'data/' + train_name # store in this project's directory

frame_from_train = 1
frame_to_train = 1501
frame_from_test = 1501
frame_to_test = 1797

num_train_examples = 20
model_save_path = 'models/' + train_name + '/'
if not os.path.exists(model_save_path): os.makedirs(model_save_path)
model_save_path_name = model_save_path + 'model.ckpt'



class TrsvOpts:
  checkpoint: str
  in_path: str
  out: str
  tmp_dir: str
  device = '/gpu:0'
  batch_size = 4
  no_disk = False

trsv_opts = TrsvOpts()
trsv_opts.checkpoint = trsv_model_path_name
trsv_opts.in_path = trsv_ori_video_path_name
trsv_opts.out = trsv_output_video_path_name
trsv_opts.tmp_dir = trsv_video_frames_path


def calculate_and_save_residuals():
  # frame_from_train = 1
  # frame_to_train = 15001
  residual_calculator.save_residuals(
    res_ori_frames_dir, res_npy_store_dir, 'res_ori_train.npy', frame_from_train, frame_to_train)
  residual_calculator.save_residuals(
    res_trs_frames_dir, res_npy_store_dir, 'res_trs_train.npy', frame_from_train, frame_to_train)

  # frame_from_test = 15001
  # frame_to_test = 17982
  residual_calculator.save_residuals(
    res_ori_frames_dir, res_npy_store_dir, 'res_ori_test.npy', frame_from_test, frame_to_test)
  residual_calculator.save_residuals(
    res_trs_frames_dir, res_npy_store_dir, 'res_trs_test.npy', frame_from_test, frame_to_test)


trsv.transform_video_and_generate_frames(trsv_opts)
calculate_and_save_residuals()
optimize.optimize(
  res_npy_store_dir + '/res_ori_train.npy',
  res_npy_store_dir + '/res_trs_train.npy',
  model_save_path_name, num_train_examples)

