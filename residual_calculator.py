"""
Calculate residuals of neighbour frames for a video.

Input: a folder containing frames to calculate residuals. Frames' names should be 'frame_%d.png'.

Output: an array represents the residuals:
R[[ r0 ], [ r1 ], [ r2 ], ..., [ rn ]], where r0 is residual between frame1 and frame2, etc.

API: get_residuals(in_dir)
"""

from PIL import Image
import numpy as np
from argparse import ArgumentParser
import os
from global_variable import logging as log


def build_parser():
  parser = ArgumentParser()
  parser.add_argument('--in_dir', type=str, dest="in_dir", required=True)

  return parser


def img_to_tensor(img_path):
  return np.asarray(Image.open(img_path), dtype=np.int16)


def calculate_residual(frame_tensor_1, frame_tensor_2):
  return frame_tensor_2 - frame_tensor_1


def get_residuals(frames_dir, frame_from, frame_to):
  log.info('calculating residuals for ' + frames_dir + ', from: ' + str(frame_from) + ', to: ' + str(frame_to))
  in_files = os.listdir(frames_dir)
  log.info('total files count: ' + str(len(in_files)))
  residuals = []
  for x in range(frame_from, frame_to):
    tensor_x = img_to_tensor(frames_dir + '/' + in_files[x - 1])
    tensor_x_next = img_to_tensor(frames_dir + '/' + in_files[x])
    residuals.append(calculate_residual(tensor_x, tensor_x_next))
  log.info('residuals calculated, size: ' + str(len(residuals)))
  return residuals


def save_residuals(frames_dir, data_dir, npy_name, frame_from, frame_to):
  residuals = get_residuals(frames_dir, frame_from, frame_to)
  if not os.path.exists(data_dir):
    os.makedirs(data_dir)
  npy_path = data_dir + '/' + npy_name
  log.info('saving residuals to ' + npy_path)
  np.save(npy_path, residuals)
  log.info('residuals saved, size: ' + str(os.stat(npy_path).st_size) + ' bytes')
  return True


def main():
  parser = build_parser()
  args = parser.parse_args()
  in_dir = args.in_dir


if __name__ == '__main__':
  main()
