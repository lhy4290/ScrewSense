"""count.py 與 classify.py 共用的工具：參數、相機、部位分群、繪圖。"""

import argparse
import math
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = ROOT / "models" / "screw_best.pt"

# 兩個部位框的距離小於這個值（像素）就視為同一顆螺絲
MERGE_DISTANCE_PX = 90

# 各類別框線顏色（BGR）：0=Space, 1=Head, 2=Thread
CLASS_COLORS = {
    0: (255, 0, 0),
    1: (255, 255, 0),
    2: (255, 255, 255),
}


def parse_args(description):
  parser = argparse.ArgumentParser(description=description)
  parser.add_argument("--model", default=str(DEFAULT_MODEL), help="模型權重路徑")
  parser.add_argument("--camera", type=int, default=1, help="相機編號")
  parser.add_argument("--conf", type=float, default=0.5, help="信心門檻")
  parser.add_argument("--device", default="0", help="推論裝置，例如 0（GPU）或 cpu")
  return parser.parse_args()


def open_camera(index):
  cap = cv2.VideoCapture(index)
  if not cap.isOpened():
    print(f"錯誤：無法開啟編號 {index} 的相機，請確認相機是否正確連接。")
    sys.exit(1)
  print("相機已啟動，按 'q' 鍵即可關閉視窗結束程式。")
  return cap


def get_box_distance(box1, box2):
  """兩個 xyxy 框之間的最短距離，重疊時為 0。"""
  dx = max(0, box1[0] - box2[2], box2[0] - box1[2])
  dy = max(0, box1[1] - box2[3], box2[1] - box1[3])
  return math.hypot(dx, dy)


def cluster_parts(boxes, classes, max_distance=MERGE_DISTANCE_PX):
  """把 Head / Thread / Space 等部位框合併成一顆顆螺絲。

  每次合併距離最近的兩群，同一群裡每個類別最多出現一次，
  直到沒有距離小於 max_distance 的兩群為止。
  回傳 list[list[int]]，每個子 list 是一顆螺絲包含的框索引。
  """
  clusters = [[i] for i in range(len(boxes))]

  while True:
    best_pair = None
    min_d = float("inf")

    for i in range(len(clusters)):
      for j in range(i + 1, len(clusters)):
        merged_classes = [classes[idx] for idx in clusters[i] + clusters[j]]
        if len(merged_classes) != len(set(merged_classes)):
          continue

        d = min(
            get_box_distance(boxes[a], boxes[b])
            for a in clusters[i]
            for b in clusters[j]
        )
        if d < max_distance and d < min_d:
          min_d = d
          best_pair = (i, j)

    if best_pair is None:
      return clusters

    i, j = best_pair
    clusters[i].extend(clusters[j])
    clusters.pop(j)


def draw_box(frame, box, color, label):
  x1, y1, x2, y2 = map(int, box)
  cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
  (w, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
  cv2.rectangle(frame, (x1, y1 - 20), (x1 + w, y1), color, -1)
  cv2.putText(
      frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX,
      0.5, (0, 0, 0), 1, cv2.LINE_AA,
  )
