import motmetrics as mm
import numpy as np
import os


def cal_metrics(gt_file, ts_file, path):
    metrics = list(mm.metrics.motchallenge_metrics)
    gt = mm.io.loadtxt(gt_file, fmt="mot16", min_confidence=1)
    ts = mm.io.loadtxt(ts_file, fmt="mot16")
    name = os.path.splitext(os.path.basename(ts_file))[0]
    acc = mm.utils.compare_to_groundtruth(gt, ts, 'iou', distth=0.5)
    mh = mm.metrics.create()
    summary = mh.compute(acc, metrics=metrics, name=name)
    print(mm.io.render_summary(summary, formatters=mh.formatters, namemap=mm.io.motchallenge_metric_names))
    file = open(path, 'w')
    file.write('++++++++')
    file.write(mm.io.render_summary(summary, formatters=mh.formatters, namemap=mm.io.motchallenge_metric_names))


if __name__ == '__main__':
    ts_file = '/results.txt'
    gt_file = '/gt.txt'
    path = './mota.txt'

    cal_metrics(gt_file, ts_file, path)
