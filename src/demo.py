from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths

import os
import sys
import cv2
import json
import copy
import numpy as np
from opts import opts
from detector import Detector

image_ext = ['jpg', 'jpeg', 'png', 'webp']
video_ext = ['mp4', 'mov', 'avi', 'mkv']
time_stats = ['tot', 'load', 'pre', 'net', 'dec', 'post', 'merge', 'display']

def demo(opt):
  os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
  opt.debug = max(opt.debug, 1)
  detector = Detector(opt)

  if opt.demo == 'webcam' or \
    opt.demo[opt.demo.rfind('.') + 1:].lower() in video_ext:
    is_video = True
    cam = cv2.VideoCapture(0 if opt.demo == 'webcam' else opt.demo)
  else:
    is_video = False
    if os.path.isdir(opt.demo):
      image_names = []
      ls = os.listdir(opt.demo)
      for file_name in sorted(ls):
          ext = file_name[file_name.rfind('.') + 1:].lower()
          if ext in image_ext:
              image_names.append(os.path.join(opt.demo, file_name))
    else:
      image_names = [opt.demo]

  out = None
  out_name = opt.demo[opt.demo.rfind('/') + 1:]
  if opt.save_video:
    if not os.path.exists('../results'):
        os.mkdir('../results')
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('../results/{}'.format(
      opt.exp_id + '_' + out_name), fourcc, opt.save_framerate, (
        opt.input_w, opt.input_h))

  if opt.debug < 5:
    detector.pause = False
  cnt = 0
  results = {}
  results_test = {}
  temporary = []

  while True:
      if is_video:
        _, img = cam.read()
        if img is None:
          save_and_exit(opt, out, results, results_test, out_name)
      else:
        if cnt < len(image_names):
          img = cv2.imread(image_names[cnt])
        else:
          save_and_exit(opt, out, results, results_test, out_name)
      cnt += 1

      if opt.resize_video:
        img = cv2.resize(img, (opt.input_w, opt.input_h))

      if cnt < opt.skip_first:
        continue

      if not opt.save_video:
        cv2.imshow('input', img)

      ret = detector.run(img)

      time_str = 'frame {} |'.format(cnt)
      for stat in time_stats:
        time_str = time_str + '{} {:.3f}s |'.format(stat, ret[stat])
      print(time_str)

      results[cnt] = ret['results']

      temporary = results[cnt]
      temporary = [(item['tracking_id'], item['bbox'], item['class']) for item in temporary]
      temporary = [(t[0],) + tuple(t[1]) + (t[2],) for t in temporary]
      results_test[cnt] = temporary

      if opt.save_video:
        out.write(ret['generic'])
        if not is_video:
          cv2.imwrite('../results/demo{}.jpg'.format(cnt), ret['generic'])

      if cv2.waitKey(1) == 27:
        save_and_exit(opt, out, results, results_test, out_name)
        return 

  save_and_exit(opt, out, results, results_test)


def save_and_exit(opt, out=None, results=None, results_test=None, out_name=''):
  if opt.save_results and (results is not None):
    save_dir =  '../results/{}_results.json'.format(opt.exp_id + '_' + out_name)
    print('saving results to', save_dir)
    json.dump(_to_list(copy.deepcopy(results)), 
              open(save_dir, 'w'))

  if opt.save_results and (results_test is not None):
    save_txt_dir = '../results/{}_results.txt'.format(opt.exp_id + '_' + out_name)
    with open(save_txt_dir, 'w') as file:
        for k, vs in results_test.items():
            for v in vs:
                x_top_left = format((v[1] / opt.input_w) * 4096, '.2f')
                y_top_left = format((v[2] / opt.input_h) * 3000, '.2f')
                x_bottom_right = format((v[3] / opt.input_w) * 4096, '.2f')
                y_bottom_right = format((v[4] / opt.input_h) * 3000, '.2f')
                x_min = x_top_left
                y_min = y_top_left
                w = format(float(x_bottom_right) - float(x_top_left), '.2f')
                h = format(float(y_bottom_right) - float(y_top_left), '.2f')

                content = [str(k)] + [str(int(v[0])-1)] + [str(x_min)] + [str(y_min)] + [str(w)] + [str(h)]\
                          + [str(1)] + [str(v[5])] + [str(1)]
                line = ','.join(content)
                file.write(line + '\n')

  if opt.save_video and out is not None:
    out.release()

  sys.exit(0)


def _to_list(results):
  for img_id in results:
    for t in range(len(results[img_id])):
      for k in results[img_id][t]:
        if isinstance(results[img_id][t][k], (np.ndarray, np.float32)):
          results[img_id][t][k] = results[img_id][t][k].tolist()
  return results


if __name__ == '__main__':
  opt = opts().init()
  demo(opt)
