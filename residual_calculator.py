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


def build_parser():
  parser = ArgumentParser()
  parser.add_argument('--in_dir', type=str, dest="in_dir", required=True)

  return parser


def img_to_tensor(img_path):
  return np.asarray(Image.open(img_path), dtype=np.int16)


def calculate_residual(frame_tensor_1, frame_tensor_2):
  return frame_tensor_2 - frame_tensor_1


def get_residuals(in_dir, from_frame_idx, to_frame_idx):
  print('calculating residuals for ' + in_dir + ', from: ' + str(from_frame_idx) + ', to: ' + str(to_frame_idx))
  in_files = os.listdir(in_dir)
  print('total files count: ' + str(len(in_files)))
  residuals = []
  for x in range(from_frame_idx, to_frame_idx - 1):
    tensor_x = img_to_tensor(in_dir + '\\' + in_files[x])
    tensor_x_next = img_to_tensor(in_dir + '\\' + in_files[x + 1])
    residuals.append(calculate_residual(tensor_x, tensor_x_next))
  print('residuals calculated, size: ' + str(len(residuals)))
  return residuals


def main():
  parser = build_parser()
  args = parser.parse_args()
  in_dir = args.in_dir


if __name__ == '__main__':
  main()
