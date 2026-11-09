"""即時偵測：從相機讀取畫面，畫出 Head / Thread / Space 偵測框。"""

import cv2
from ultralytics import YOLO

from screw_utils import open_camera, parse_args


def main():
  args = parse_args("即時螺絲部位偵測")
  model = YOLO(args.model)
  print("模型類別：", model.names)

  cap = open_camera(args.camera)
  window_name = "Screw Detection"
  cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

  while cap.isOpened():
    success, frame = cap.read()
    if not success:
      print("無法讀取相機畫面。")
      break

    results = model(frame, device=args.device, verbose=False, conf=args.conf)
    cv2.imshow(window_name, results[0].plot())

    if cv2.waitKey(1) & 0xFF == ord("q"):
      break

  cap.release()
  cv2.destroyAllWindows()


if __name__ == "__main__":
  main()
