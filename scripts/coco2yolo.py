#!/usr/bin/env python
# -*- coding: utf-8 -*-
# pylint: disable=missing-docstring
import os
import shutil
import tqdm

import numpy as np
from pycocotools.coco import COCO
from coco.label import coco91_labels


def coco91_to_yolo_id(yolo_names_file, coco_labels=coco91_labels()):
  coco_names = np.array(coco_labels)
  yolo_names = np.loadtxt(yolo_names_file, dtype="str", delimiter="\n", ndmin=1)
  return yolo_names.tolist(), [list(coco_names[i] == yolo_names).index(True) if any(coco_names[i] == yolo_names) else None for i in range(coco_names.size)]


def coco2yolo(coco_img_dir, coco_ann_file, yolo_names_file, output_dir,
    output_name, output_img_prefix=None):
  path = os.path

  output_txt = path.join(output_dir, output_name + ".txt")
  output_img_dir = path.join(output_dir, output_name)
  os.makedirs(output_img_dir, exist_ok=True)

  output_txt_file = open(output_txt, "w")

  cat_names, coco_to_yolo_id = coco91_to_yolo_id(yolo_names_file)

  coco = COCO(coco_ann_file)

  cat_ids = coco.getCatIds(catNms=cat_names)
  for cat_id in cat_ids:
    yolo_id = coco_to_yolo_id[cat_id-1]

    img_ids = coco.getImgIds(catIds=cat_id)

    print(f"\nCategory: {yolo_id}={cat_names[yolo_id]}, imgs: {len(img_ids)}")

    cat_anns_n = 0
    for img_id in tqdm.tqdm(img_ids):
      img = coco.loadImgs(ids=img_id)
      img_name = img[0]["file_name"]
      image_width = img[0]["width"]
      image_height = img[0]["height"]

      output_txt_file.write(img_name if output_img_prefix is None \
          else output_img_prefix + img_name)
      output_txt_file.write(os.linesep)

      img_src = path.join(coco_img_dir, img_name)
      img_dst = path.join(output_img_dir, img_name)
      img_dst_txt = path.splitext(img_dst)[0] + ".txt"
      # print(f"{img_src}")
      # print(f"  {img_dst}")
      # print(f"  {img_dst_txt}")
      shutil.copy(img_src, img_dst)

      with open(img_dst_txt, "w") as txt:
        ann_ids = coco.getAnnIds(imgIds=img_id, catIds=cat_id)
        anns = coco.loadAnns(ann_ids)
        cat_anns_n += len(anns)

        for ann in anns:
          x_top_left = ann["bbox"][0]
          y_top_left = ann["bbox"][1]
          bbox_width = ann["bbox"][2]
          bbox_height = ann["bbox"][3]

          x_center = x_top_left + bbox_width / 2
          y_center = y_top_left + bbox_height / 2

          # darknet annotation format
          #  <object-class> <x_center> <y_center> <width> <height>
          a = x_center / image_width
          b = y_center / image_height
          c = bbox_width / image_width
          d = bbox_height / image_height
          print(f"{yolo_id} {a:.6f} {b:.6f} {c:.6f} {d:.6f}", file=txt)

    print(f"Category: {yolo_id}={cat_names[yolo_id]}, anns: {cat_anns_n}, COMPLETED")

  output_txt_file.close()


def _parse_args():
  import argparse
  parser = argparse.ArgumentParser(usage="python scripts/coco2yolo.py <options>")

  parser.add_argument("--coco_img_dir", type=str,
      default=f"{os.environ['HOME']}/Codes/devel/datasets/coco2017/train2017/",
      help="coco image dir, default: %(default)s")
  parser.add_argument("--coco_ann_file", type=str,
      default=f"{os.environ['HOME']}/Codes/devel/datasets/coco2017/annotations/instances_train2017.json",
      help="coco annotation file, default: %(default)s")
  parser.add_argument("--yolo_names_file", type=str,
      default="./cfg/coco/coco.names",
      help="coco desired objects, default: %(default)s")

  parser.add_argument("--output_dir", type=str,
      default=f"{os.environ['HOME']}/yolov4/datasets/",
      help="output dir for yolo datasets, default: %(default)s")
  parser.add_argument("--output_name", type=str,
      default="train",
      help="output name for img txt and dir, default: %(default)s")
  parser.add_argument("--output_img_prefix", type=str,
      help="output img prefix before img name, default: %(default)s")

  args = parser.parse_args()

  print("Args")
  print(f"  coco_img_dir: {args.coco_img_dir}")
  print(f"  coco_ann_file: {args.coco_ann_file}")
  print(f"  yolo_names_file: {args.yolo_names_file}")
  print(f"  output_dir: {args.output_dir}")
  print(f"  output_name: {args.output_name}")
  print(f"  output_img_prefix: {args.output_img_prefix}")

  return args


if __name__ == "__main__":
  args = _parse_args()
  coco2yolo(args.coco_img_dir, args.coco_ann_file, args.yolo_names_file,
      args.output_dir, args.output_name, args.output_img_prefix)


# https://gist.github.com/kaancolak/c66ba49540bbf075fbd46bd98100b544
# https://github.com/ultralytics/JSON2YOLO
