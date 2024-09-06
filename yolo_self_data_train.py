from ultralytics import YOLO
import os
from IPython.display import display, Image
from IPython import display
display.clear_output()
# ! yolo mode=checks



# 自己制造的数据集 直接复制
# !pip install roboflow

from roboflow import Roboflow
rf = Roboflow(api_key="3F816bMIjNxXqNHwey3r")
project = rf.workspace("x-ghdui").project("x-oxwud")
version = project.version(1)
dataset = version.download("yolov8")

# 查看模型的性能
Image(filename=f'/content/runs/detect/train2/results.png', width=600)

#开始训练
!yolo task=detect mode=train model=yolov8m.pt data={dataset.location}/data.yaml epochs=20 imgsz=640

#模型评估
!yolo task=detect mode=val model=/content/runs/detect/train2/weights/best.pt data={dataset.location}/data.yaml

#展示模型进行预测
import glob
from IPython.display import Image, display
for image_path in glob.glob(f'/content/runs/detect/val/*.jpg'):
    display(Image(filename=image_path,height=600))