import os, sys
import cv2, glob, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')
from ultralytics import YOLO

# train_anno = json.load(open('训练集(有标注第一批)/标注/45.json', encoding='utf-8'))
# train_anno[0], len(train_anno)
#
# pd.read_json('训练集(有标注第一批)/标注/45.json')
#
# video_path = '训练集(有标注第一批)/视频/45.mp4'
# cap = cv2.VideoCapture(video_path)
# while True:
#     # 读取下一帧
#     ret, frame = cap.read()
#     if not ret:
#         break
#     break
#
# int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#
# bbox = [746, 494, 988, 786]
#
# pt1 = (bbox[0], bbox[1])
# pt2 = (bbox[2], bbox[3])
#
# color = (0, 255, 0)
# thickness = 2  # 线条粗细
#
# cv2.rectangle(frame, pt1, pt2, color, thickness)
#
# frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
# plt.imshow(frame)

if not os.path.exists('yolo-dataset/'):
    os.mkdir('yolo-dataset/')
if not os.path.exists('yolo-dataset/train'):
    os.mkdir('yolo-dataset/train')
if not os.path.exists('yolo-dataset/val'):
    os.mkdir('yolo-dataset/val')

dir_path = os.path.abspath('./') + '/'

# with open('yolo-dataset/yolo.yaml', 'w', encoding='utf-8') as up:
#     up.write(f'''
# path: {dir_path}/yolo-dataset/
# train: train/
# val: val/
#
# names:
#     0: 非机动车违停
#     1: 机动车违停
#     2: 垃圾桶满溢
#     3: 违法经营
# ''')

train_annos = glob.glob('train/target/*.json')
train_videos = glob.glob('train/vedio/*.mp4')
train_annos.sort()
train_videos.sort()

category_labels = ["非机动车违停", "机动车违停", "垃圾桶满溢", "违法经营"]

for anno_path, video_path in zip(train_annos, train_videos):
    print(video_path)
    anno_df = pd.read_json(anno_path)
    cap = cv2.VideoCapture(video_path)
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        img_height, img_width = frame.shape[:2]

        frame_anno = anno_df[anno_df['frame_id'] == frame_idx]
        cv2.imwrite('./yolo-dataset/train/' + anno_path.split('/')[-1][:-5] + '_' + str(frame_idx) + '.jpg', frame)

        if len(frame_anno) != 0:
            with open('./yolo-dataset/train/' + anno_path.split('/')[-1][:-5] + '_' + str(frame_idx) + '.txt',
                      'w') as up:
                for category, bbox in zip(frame_anno['category'].values, frame_anno['bbox'].values):
                    category_idx = category_labels.index(category)

                    x_min, y_min, x_max, y_max = bbox
                    x_center = (x_min + x_max) / 2 / img_width
                    y_center = (y_min + y_max) / 2 / img_height
                    width = (x_max - x_min) / img_width
                    height = (y_max - y_min) / img_height

                    if x_center > 1:
                        print(bbox)
                    up.write(f'{category_idx} {x_center} {y_center} {width} {height}\n')

        frame_idx += 1

for anno_path, video_path in zip(train_annos[-8:], train_videos[-8:]):
    print(video_path)
    anno_df = pd.read_json(anno_path)
    cap = cv2.VideoCapture(video_path)
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        img_height, img_width = frame.shape[:2]
        
        frame_anno = anno_df[anno_df['frame_id'] == frame_idx]
        cv2.imwrite('./yolo-dataset/val/' + anno_path.split('/')[-1][:-5] + '_' + str(frame_idx) + '.jpg', frame)

        if len(frame_anno) != 0:
            with open('./yolo-dataset/val/' + anno_path.split('/')[-1][:-5] + '_' + str(frame_idx) + '.txt', 'w') as up:
                for category, bbox in zip(frame_anno['category'].values, frame_anno['bbox'].values):
                    category_idx = category_labels.index(category)
                    
                    x_min, y_min, x_max, y_max = bbox
                    x_center = (x_min + x_max) / 2 / img_width
                    y_center = (y_min + y_max) / 2 / img_height
                    width = (x_max - x_min) / img_width
                    height = (y_max - y_min) / img_height

                    up.write(f'{category_idx} {x_center} {y_center} {width} {height}\n')
        frame_idx += 1

model = YOLO("yolov8x.pt")
results = model.train(data="yolo-dataset/yolo.yaml", epochs=30, imgsz=1080, batch=-1)

category_labels = ["非机动车违停", "机动车违停", "垃圾桶满溢", "违法经营"]

if not os.path.exists('result/'):
    os.mkdir('result')

model = YOLO("runs/detect/train/weights/last.pt")

for path in glob.glob('test/*.mp4'):
    submit_json = []
    results = model(path, conf=0.05, imgsz=1080, verbose=False)
    for idx, result in enumerate(results):
        boxes = result.boxes  # Boxes object for bounding box outputs
        masks = result.masks  # Masks object for segmentation masks outputs
        keypoints = result.keypoints  # Keypoints object for pose outputs
        probs = result.probs  # Probs object for classification outputs
        obb = result.obb  # Oriented boxes object for OBB outputs

        if len(boxes.cls) == 0:
            continue

        xywh = boxes.xyxy.data.cpu().numpy().round()
        cls = boxes.cls.data.cpu().numpy().round()
        conf = boxes.conf.data.cpu().numpy()
        for i, (ci, xy, confi) in enumerate(zip(cls, xywh, conf)):
            submit_json.append(
                {
                    'frame_id': idx,
                    'event_id': i + 1,
                    'category': category_labels[int(ci)],
                    'bbox': list([int(x) for x in xy]),
                    "confidence": float(confi)
                }
            )

    with open('./result_last/' + path.split('/')[-1][:-4] + '.json', 'w', encoding='utf-8') as up:
        json.dump(submit_json, up, indent=4, ensure_ascii=False)
