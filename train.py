"""以 YOLOv8n 為預訓練模型，訓練螺絲部位偵測（Head / Thread / Space）。"""

from ultralytics import YOLO

if __name__ == "__main__":
  model = YOLO("yolov8n.pt")  # 不存在時 ultralytics 會自動下載

  model.train(
      data="screw_dataset/data.yaml",
      epochs=100,
      imgsz=640,
      batch=32,
      device=0,
      project="screw_project",
      name="screw_run",
      workers=2,
  )

  print("模型訓練完成！最佳權重在 runs/detect/screw_project/screw_run/weights/best.pt")
