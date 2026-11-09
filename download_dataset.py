"""從 Roboflow 下載螺絲資料集（YOLOv8 格式）到 screw_dataset/。

API key 請放在環境變數 ROBOFLOW_API_KEY，不要寫進程式碼：
  PowerShell: $env:ROBOFLOW_API_KEY = "你的key"
  bash:       export ROBOFLOW_API_KEY="你的key"
"""

import os
import sys

from roboflow import Roboflow

api_key = os.environ.get("ROBOFLOW_API_KEY")
if not api_key:
  sys.exit("請先設定環境變數 ROBOFLOW_API_KEY")

rf = Roboflow(api_key=api_key)
project = rf.workspace("mec-r9w1v").project("screw-lwjvf")
dataset = project.version(1).download("yolov8", location="screw_dataset")

print("下載完成：", os.listdir("screw_dataset"))
