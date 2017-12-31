import our_optimize
import residual_calculator
import os
import fast_style_transfer.transform_video as trsv

# mkdirs
if not os.path.exists('output/video/'): os.makedirs('output/video/')

train_name = 'wave_208p_10min_wave' # 训练名称

trsv_model_path_name = 'fast_style_transfer/models/wave.ckpt' # 风格模型
trsv_ori_video_path_name = 'input/output10_t_312.mp4' # 原始视频文件
trsv_output_video_path = 'output/video/' + train_name + '/'
if not os.path.exists(trsv_output_video_path): os.makedirs(trsv_output_video_path)
trsv_output_video_path_name = trsv_output_video_path + 'output.mp4' # 输出视频文件路径
trsv_video_frames_path = 'frames/' + train_name + '/'  # 原始视频、风格迁移后视频的每一帧图像

res_ori_frames_dir = trsv_video_frames_path + '/in' # 原始视频的每一帧所在目录
res_trs_frames_dir = trsv_video_frames_path + '/out' # 风格迁移后视频的每一帧所在目录
res_npy_store_dir = 'data/' + train_name # 残差存放位置

frame_from_train = 1
frame_to_train = 15001
frame_from_test = 15001
frame_to_test = 17982

num_train_examples = 128000 # 32000 iterations
model_save_path = 'models/' + train_name + '/' # 训练好的模型存放位置
if not os.path.exists(model_save_path): os.makedirs(model_save_path)
model_save_path_name = model_save_path + 'model.ckpt'

# “风格化”残差叠加到的第一帧的完整路径
first_frame_path_name = res_trs_frames_dir + '/frame_' + str(frame_from_test) + '.png'

# 根据风格化残差叠加生成的帧所在的位置
gen_frames_path = 'output/frames/' + train_name + '/'

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
our_optimize.optimize(
  res_npy_store_dir + '/res_ori_train.npy',
  res_npy_store_dir + '/res_trs_train.npy',
  model_save_path_name, num_train_examples
)

# -------------- 注意，下面的代码不要直接运行，在得到模型（确认models文件夹里有内容）后，再把上面71行-79行注释掉，取消下面的注释并执行

# our_optimize.generate_frames(
#   first_frame_path_name,
#   res_npy_store_dir + '/ori_res_test.npy',
#   res_npy_store_dir + '/trs_res_test.npy',
#   model_save_path_name, gen_frames_path
# )


