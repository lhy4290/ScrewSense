"""即時計數：追蹤畫面中的螺絲部位，合併成完整螺絲後計算數量。

一顆螺絲連續出現 1 秒以上才會被記錄，避免誤判。
"""

import time

import cv2
from ultralytics import YOLO

from screw_utils import CLASS_COLORS, cluster_parts, draw_box, open_camera, parse_args

STABLE_SECONDS = 1.0


def main():
  args = parse_args("即時螺絲計數")
  model = YOLO(args.model)
  print("模型類別：", model.names)

  cap = open_camera(args.camera)
  window_name = "Screw Counting"
  cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

  screw_start_times = {}
  recorded_screws = set()
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
        part_names = [model.names[classes[i]] for i in cluster]
        frame_screw_info.append({"id": unified_id, "parts": part_names})

        screw_start_times.setdefault(unified_id, current_time)
        duration = current_time - screw_start_times[unified_id]
        if duration >= STABLE_SECONDS and unified_id not in recorded_screws:
          print(f"[記錄] 螺絲 ID {unified_id} 已連續出現 {duration:.1f} 秒")
          recorded_screws.add(unified_id)

        for i in cluster:
          cls_id = classes[i]
          label = f"id:{unified_id} {model.names[cls_id]} {confidences[i]:.2f}"
          draw_box(annotated_frame, boxes[i], CLASS_COLORS.get(cls_id, (0, 255, 0)), label)

    if current_time - last_print_time >= 1.0:
      print(f"\n--- 目前畫面共有 {len(frame_screw_info)} 顆螺絲，累計記錄 {len(recorded_screws)} 顆 ---")
      for screw in frame_screw_info:
        print(f"  > ID {screw['id']} | 部位: {screw['parts']}")
      last_print_time = current_time

    cv2.imshow(window_name, annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
      break

  cap.release()
  cv2.destroyAllWindows()


if __name__ == "__main__":
  main()
