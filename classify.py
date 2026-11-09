"""即時分類：依照螺絲在畫面中的長度，自動分成不同種類（Type_A、Type_B…）。

長度取同一顆螺絲各部位中心點的最大距離；只偵測到一個部位時改用外框對角線。
種類中心會動態建立：與現有中心差距超過 NEW_TYPE_GAP_PX 就新增一類，
否則歸入最接近的一類，並用移動平均更新該類中心。
"""

import math
import time

import cv2
from ultralytics import YOLO

from screw_utils import CLASS_COLORS, cluster_parts, draw_box, open_camera, parse_args

STABLE_SECONDS = 1.0
NEW_TYPE_GAP_PX = 30
CENTER_UPDATE_RATE = 0.10


def screw_length(cluster, boxes):
  centers = [((boxes[i][0] + boxes[i][2]) / 2, (boxes[i][1] + boxes[i][3]) / 2) for i in cluster]
  if len(centers) >= 2:
    span = max(
        math.dist(centers[a], centers[b])
        for a in range(len(centers))
        for b in range(a + 1, len(centers))
    )
    if span > 0:
      return span

  x1 = min(boxes[i][0] for i in cluster)
  y1 = min(boxes[i][1] for i in cluster)
  x2 = max(boxes[i][2] for i in cluster)
  y2 = max(boxes[i][3] for i in cluster)
  return math.hypot(x2 - x1, y2 - y1)


def assign_type(length, type_centers):
  """回傳 length 所屬種類的索引，必要時新增種類或更新中心（會修改 type_centers）。"""
  if not type_centers:
    type_centers.append(length)
    return 0

  distances = [abs(length - c) for c in type_centers]
  closest = distances.index(min(distances))

  if distances[closest] > NEW_TYPE_GAP_PX:
    type_centers.append(length)
    type_centers.sort()
    return type_centers.index(length)

  type_centers[closest] = (1 - CENTER_UPDATE_RATE) * type_centers[closest] + CENTER_UPDATE_RATE * length
  return closest


def type_name(idx):
  return f"Type_{chr(65 + idx)}" if idx < 26 else f"Type_{idx}"


def main():
  args = parse_args("即時螺絲長度分類")
  model = YOLO(args.model)
  print("模型類別：", model.names)

  cap = open_camera(args.camera)
  window_name = "Screw Classification"
  cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

  screw_start_times = {}
  recorded_screws = set()
  type_centers = []
  last_print_time = time.time()

  while cap.isOpened():
    success, frame = cap.read()
    if not success:
      print("無法讀取相機畫面。")
      break

    results = model.track(
        frame, persist=True, device=args.device, verbose=False, conf=args.conf
    )
    current_time = time.time()
    annotated_frame = frame.copy()
    frame_screw_info = []

    if results[0].boxes and results[0].boxes.id is not None:
      boxes = results[0].boxes.xyxy.cpu().numpy()
      track_ids = results[0].boxes.id.cpu().numpy().astype(int)
      classes = results[0].boxes.cls.cpu().numpy().astype(int)
      confidences = results[0].boxes.conf.cpu().numpy()

      for cluster in cluster_parts(boxes, classes):
        unified_id = min(track_ids[i] for i in cluster)
        length = screw_length(cluster, boxes)
        screw_type = type_name(assign_type(length, type_centers))
        frame_screw_info.append({"id": unified_id, "type": screw_type})

        screw_start_times.setdefault(unified_id, current_time)
        duration = current_time - screw_start_times[unified_id]
        if duration >= STABLE_SECONDS and unified_id not in recorded_screws:
          print(f"[記錄] 螺絲 ID {unified_id} 歸類為 {screw_type}（長度 {length:.1f}px）")
          recorded_screws.add(unified_id)

        for i in cluster:
          cls_id = classes[i]
          label = f"id:{unified_id} {screw_type} {confidences[i]:.2f}"
          draw_box(annotated_frame, boxes[i], CLASS_COLORS.get(cls_id, (0, 255, 0)), label)

    if current_time - last_print_time >= 1.0:
      print(f"\n--- 目前畫面共有 {len(frame_screw_info)} 顆螺絲 ---")
      type_groups = {}
      for screw in frame_screw_info:
        type_groups.setdefault(screw["type"], []).append(screw["id"])
      for s_type, ids in sorted(type_groups.items()):
        print(f"  > {s_type} | ID: {ids}")
      last_print_time = current_time

    cv2.imshow(window_name, annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
      break

  cap.release()
  cv2.destroyAllWindows()


if __name__ == "__main__":
  main()
