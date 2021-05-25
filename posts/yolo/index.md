# YOLOv5训练自己的数据集

# YOLOv5炼丹

>yolo目标检测算法是`you only look once`,图片只需要被检测一次就可以出结果   

## VOC数据集转换为Yolo数据集格式
> windows系统
- images 里面存放的是训练图片, labels_voc 里面存放的是 .xml (标签)，labels里面存放的是*.txt(转换后的标签),  classes.names 存放所有分类名字， 每行一个类别

![](https://cdn.jsdelivr.net/gh/clearyup/picgo/img/20210517220422.png)
- 转换python代码
```python
#coding:utf-8
from __future__ import print_function

import os
import random
import glob
import xml.etree.ElementTree as ET

def xml_reader(filename):
    """ Parse a PASCAL VOC xml file """
    tree = ET.parse(filename)
    size = tree.find('size')
    width = int(size.find('width').text)
    height = int(size.find('height').text)
    objects = []
    for obj in tree.findall('object'):
        obj_struct = {}
        obj_struct['name'] = obj.find('name').text
        bbox = obj.find('bndbox')
        obj_struct['bbox'] = [int(bbox.find('xmin').text),
                              int(bbox.find('ymin').text),
                              int(bbox.find('xmax').text),
                              int(bbox.find('ymax').text)]
        objects.append(obj_struct)
    return width, height, objects


def voc2yolo(filename):
    classes_dict = {}
    with open("classes.names") as f:
        for idx, line in enumerate(f.readlines()):
            class_name = line.strip()
            classes_dict[class_name] = idx
    
    width, height, objects = xml_reader(filename)

    lines = []
    for obj in objects:
        x, y, x2, y2 = obj['bbox']
        class_name = obj['name']
        label = classes_dict[class_name]
        cx = (x2+x)*0.5 / width
        cy = (y2+y)*0.5 / height
        w = (x2-x)*1. / width
        h = (y2-y)*1. / height
        line = "%s %.6f %.6f %.6f %.6f\n" % (label, cx, cy, w, h)
        lines.append(line)

    txt_name = filename.replace(".xml", ".txt").replace("labels_voc", "labels")
    with open(txt_name, "w") as f:
        f.writelines(lines)


def get_image_list(image_dir, suffix=['jpg', 'jpeg', 'JPG', 'JPEG','png']):
    '''get all image path ends with suffix'''
    if not os.path.exists(image_dir):
        print("PATH:%s not exists" % image_dir)
        return []
    imglist = []
    for root, sdirs, files in os.walk(image_dir):
        if not files:
            continue
        for filename in files:
            filepath = "data/custom/" + os.path.join(root, filename) + "\n"
            if filename.split('.')[-1] in suffix:
                imglist.append(filepath)
    return imglist


def imglist2file(imglist):
    random.shuffle(imglist)
    train_list = imglist[:-100]
    valid_list = imglist[-100:]
    with open("train.txt", "w") as f:
        f.writelines(train_list)
    with open("valid.txt", "w") as f:
        f.writelines(valid_list)


if __name__ == "__main__":
    xml_path_list = glob.glob("labels_voc/*.xml")
    for xml_path in xml_path_list:
        voc2yolo(xml_path)


    imglist = get_image_list("images")
    imglist2file(imglist)
```





## colab
> 整个训练流程大致是训练你自己的数据集获得权重文件，然后使用权重文件来对你的图片进行预测，输出结果

### 下载yolov5源码
```bash
!git init
!git clone https://github.com/ultralytics/yolov5.git
```
### 装载谷歌云盘
- 将你的数据集`images.zip`和标注集`Annotation.zip`和`yolov5`要用到的`权重文件`上传到谷歌云盘,训练的时候从谷歌云盘加载  

- 将这两个压缩包解压到`yolov5`目录下

```bash
!unzip /content/drive/MyDrive/Annotations.zip -d /content/yolov5/data/
```
- 解压`images.zip`之前先删除`yolov5/data/images`这个文件夹原有的图片

```bash
!rm -rf /content/yolov5/data/images/*
```

```bash
!unzip /content/drive/MyDrive/images.zip -d /content/yolov5/data/images
```
- 拷贝权重文件到`yolov5`的指定目录

```bash
!cp /content/drive/MyDrive/yolov5s.pt  /content/yolov5/weights
```
### 创建文件夹
- 在 yolov5/data 下创建以下几个文件夹 `ImageSets`, `labels`, `JPEGImages`
- 解压完成所需文件和创建文件夹后的目录结构如下

![](https://cdn.jsdelivr.net/gh/clearyup/picgo/img/20210519200848.png)

### 生成训练集测试集验证集目录
- 这里我们使用`python`脚本生成`train.txt`, `test.txt`, `train.txt`, `val.txt`

```bash
import os
import random
trainval_percent = 0.1
train_percent = 0.9
xmlfilepath = '/content/yolov5/data/Annotations'
txtsavepath = '/content/yolov5/data/ImageSets'
total_xml = os.listdir(xmlfilepath)
num = len(total_xml)
list = range(num)
tv = int(num * trainval_percent)
tr = int(tv * train_percent)
trainval = random.sample(list, tv)
train = random.sample(trainval, tr)
ftrainval = open('/content/yolov5/data/ImageSets/trainval.txt', 'w')
ftest = open('/content/yolov5/data/ImageSets/test.txt', 'w')
ftrain = open('/content/yolov5/data/ImageSets/train.txt', 'w')
fval = open('/content/yolov5/data/ImageSets/val.txt', 'w')
for i in list:
    name = total_xml[i][:-4] + '\n'
    if i in trainval:
        ftrainval.write(name)
        if i in train:
            ftest.write(name)
        else:
            fval.write(name)
    else:
        ftrain.write(name)
ftrainval.close()
ftrain.close()
fval.close()
ftest.close()
```


### 转换标注格式
- `yolov5`使用的是`.txt`格式的标注文件,如果你的标注格式是 类似这种`voc 2007`的`.xml`格式

```xml
<?xml version='1.0' encoding='utf-8'?>
<annotation>
	<folder>Annotation</folder>
	<filename>IMG_000001.jpg</filename>
	<path>D:\Annotation\IMG_000001.jpg</path>
	<source>
		<database>Unknown</database>
	</source>
	<size>
		<width>4608</width>
		<height>2592</height>
		<depth>3</depth>
	</size>
	<segmented>0</segmented>
	<object>
		<name>巴黎翠凤蝶</name>
		<pose>Unspecified</pose>
		<truncated>0</truncated>
		<difficult>0</difficult>
		<bndbox>
			<xmin>1992</xmin>
			<ymin>774</ymin>
			<xmax>2826</xmax>
			<ymax>1559</ymax>
		</bndbox>
	</object>
</annotation>

 


```

- 你可以使用以下`python`脚本进行`.xml`到`.txt`格式的转换
- `classes` 是数据集的类别
```python
import xml.etree.ElementTree as ET
import pickle
import os
from os import listdir, getcwd
from os.path import join
sets = ['train', 'test','val']
classes = [
"巴黎翠凤蝶",

"柑橘凤蝶",

"玉带凤蝶",

"碧凤蝶",

"红基美凤蝶",

"蓝凤蝶",

"金裳凤蝶",

"青凤蝶",

"朴喙蝶",

"密纹飒弄蝶",

"小黄斑弄蝶",

"无斑珂弄蝶",

"直纹稻弄蝶",

"花弄蝶",

"隐纹谷弄蝶",

"绢斑蝶",

"虎斑蝶",

"亮灰蝶",

"咖灰蝶",

"大紫琉璃灰蝶",

"婀灰蝶",

"曲纹紫灰蝶",

"波太玄灰蝶",

"玄灰蝶",

"红灰蝶",

"线灰蝶",

"维纳斯眼灰蝶",

"艳灰蝶",

"蓝灰蝶",

"青海红珠灰蝶",

"古北拟酒眼蝶",

"阿芬眼蝶",

"拟稻眉眼蝶",

"牧女珍眼蝶",

"白眼蝶",

"菩萨酒眼蝶",

"西门珍眼蝶",

"边纹黛眼蝶",

"云粉蝶",

"侏粉蝶",

"大卫粉蝶",

"大翅绢粉蝶",

"山豆粉蝶",

"橙黄豆粉蝶",

"突角小粉蝶",

"箭纹云粉蝶",

"箭纹绢粉蝶",

"红襟粉蝶",

"绢粉蝶",

"菜粉蝶",

"镉黄迁粉蝶",

"黎明豆粉蝶",

"依帕绢蝶",

"四川绢蝶",

"珍珠绢蝶",

"中环蛱蝶",

"云豹蛱蝶",

"伊诺小豹蛱蝶",

"小红蛱蝶",

"扬眉线蛱蝶",

"斐豹蛱蝶",

"曲斑珠蛱蝶",

"柱菲蛱蝶",

"柳紫闪蛱蝶",

"灿福蛱蝶",

"玄珠带蛱蝶",

"珍蛱蝶",

"琉璃蛱蝶",

"白钩蛱蝶",

"秀蛱蝶",

"绢蛱蝶",

"绿豹蛱蝶",

"网蛱蝶",

"美眼蛱蝶",

"翠蓝眼蛱蝶",

"老豹蛱蝶",

"荨麻蛱蝶",

"虬眉带蛱蝶",

"蟾福蛱蝶",

"钩翅眼蛱蝶",

"银斑豹蛱蝶",

"银豹蛱蝶",

"链环蛱蝶",

"锦瑟蛱蝶",

"黄环蛱蝶",

"黄钩蛱蝶",

"黑网蛱蝶",

"宽边黄粉蝶",

"尖翅翠蛱蝶",

"素弄蝶",

"翠袖锯眼蝶",

"蓝点紫斑蝶",

"蛇目褐蚬蝶",

"雅弄蝶",

]
def convert(size, box):
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)
def convert_annotation(image_id):
    in_file = open('/content/yolov5/data/Annotations/%s.xml' % (image_id))
    out_file = open('/content/yolov5/data/labels/%s.txt' % (image_id), 'w')
    tree = ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)
    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult) == 1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text),
             float(xmlbox.find('ymax').text))
        bb = convert((w, h), b)
        out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')
wd = getcwd()
print(wd)
for image_set in sets:
    if not os.path.exists('/content/yolov5/data/labels/'):
        os.makedirs('/content/yolov5/data/labels/')
    image_ids = open('/content/yolov5/data/ImageSets/%s.txt' % (image_set)).read().strip().split()
    list_file = open('/content/yolov5/data/%s.txt' % (image_set), 'w')
    for image_id in image_ids:
        list_file.write('/content/yolov5/data/images/%s.jpg\n' % (image_id))
        convert_annotation(image_id)
    list_file.close()
```

### 更改配置文件
- 新建`mytrain.yaml`文件,上传到`/content/yolov5/data/mytrain.yaml`
- `nc`是数据集里数据种类, `names`是类别名

```yaml
train: /content/yolov5/data/train.txt
val: /content/yolov5/data/val.txt
test: /content/yolov5/data/test.txt

nc: 94

names: [
"巴黎翠凤蝶",

"柑橘凤蝶",

"玉带凤蝶",

"碧凤蝶",

"红基美凤蝶",

"蓝凤蝶",

"金裳凤蝶",

"青凤蝶",

"朴喙蝶",

"密纹飒弄蝶",

"小黄斑弄蝶",

"无斑珂弄蝶",

"直纹稻弄蝶",

"花弄蝶",

"隐纹谷弄蝶",

"绢斑蝶",

"虎斑蝶",

"亮灰蝶",

"咖灰蝶",

"大紫琉璃灰蝶",

"婀灰蝶",

"曲纹紫灰蝶",

"波太玄灰蝶",

"玄灰蝶",

"红灰蝶",

"线灰蝶",

"维纳斯眼灰蝶",

"艳灰蝶",

"蓝灰蝶",

"青海红珠灰蝶",

"古北拟酒眼蝶",

"阿芬眼蝶",

"拟稻眉眼蝶",

"牧女珍眼蝶",

"白眼蝶",

"菩萨酒眼蝶",

"西门珍眼蝶",

"边纹黛眼蝶",

"云粉蝶",

"侏粉蝶",

"大卫粉蝶",

"大翅绢粉蝶",

"山豆粉蝶",

"橙黄豆粉蝶",

"突角小粉蝶",

"箭纹云粉蝶",

"箭纹绢粉蝶",

"红襟粉蝶",

"绢粉蝶",

"菜粉蝶",

"镉黄迁粉蝶",

"黎明豆粉蝶",

"依帕绢蝶",

"四川绢蝶",

"珍珠绢蝶",

"中环蛱蝶",

"云豹蛱蝶",

"伊诺小豹蛱蝶",

"小红蛱蝶",

"扬眉线蛱蝶",

"斐豹蛱蝶",

"曲斑珠蛱蝶",

"柱菲蛱蝶",

"柳紫闪蛱蝶",

"灿福蛱蝶",

"玄珠带蛱蝶",

"珍蛱蝶",

"琉璃蛱蝶",

"白钩蛱蝶",

"秀蛱蝶",

"绢蛱蝶",

"绿豹蛱蝶",

"网蛱蝶",

"美眼蛱蝶",

"翠蓝眼蛱蝶",

"老豹蛱蝶",

"荨麻蛱蝶",

"虬眉带蛱蝶",

"蟾福蛱蝶",

"钩翅眼蛱蝶",

"银斑豹蛱蝶",

"银豹蛱蝶",

"链环蛱蝶",

"锦瑟蛱蝶",

"黄环蛱蝶",

"黄钩蛱蝶",

"黑网蛱蝶",

"宽边黄粉蝶",

"尖翅翠蛱蝶",

"素弄蝶",

"翠袖锯眼蝶",

"蓝点紫斑蝶",

"蛇目褐蚬蝶",

"雅弄蝶",

]

```

- 删除`/content/yolov5/train.py`,上传更改过的`train.py`
- 主要只更改了 `nc`的值

```yaml
# parameters
nc: 94  # number of classes
depth_multiple: 0.33  # model depth multiple
width_multiple: 0.50  # layer channel multiple

# anchors
anchors:
  - [10,13, 16,30, 33,23]  # P3/8
  - [30,61, 62,45, 59,119]  # P4/16
  - [116,90, 156,198, 373,326]  # P5/32

# YOLOv5 backbone
backbone:
  # [from, number, module, args]
  [[-1, 1, Focus, [64, 3]],  # 0-P1/2
   [-1, 1, Conv, [128, 3, 2]],  # 1-P2/4
   [-1, 3, C3, [128]],
   [-1, 1, Conv, [256, 3, 2]],  # 3-P3/8
   [-1, 9, C3, [256]],
   [-1, 1, Conv, [512, 3, 2]],  # 5-P4/16
   [-1, 9, C3, [512]],
   [-1, 1, Conv, [1024, 3, 2]],  # 7-P5/32
   [-1, 1, SPP, [1024, [5, 9, 13]]],
   [-1, 3, C3, [1024, False]],  # 9
  ]

# YOLOv5 head
head:
  [[-1, 1, Conv, [512, 1, 1]],
   [-1, 1, nn.Upsample, [None, 2, 'nearest']],
   [[-1, 6], 1, Concat, [1]],  # cat backbone P4
   [-1, 3, C3, [512, False]],  # 13

   [-1, 1, Conv, [256, 1, 1]],
   [-1, 1, nn.Upsample, [None, 2, 'nearest']],
   [[-1, 4], 1, Concat, [1]],  # cat backbone P3
   [-1, 3, C3, [256, False]],  # 17 (P3/8-small)

   [-1, 1, Conv, [256, 3, 2]],
   [[-1, 14], 1, Concat, [1]],  # cat head P4
   [-1, 3, C3, [512, False]],  # 20 (P4/16-medium)

   [-1, 1, Conv, [512, 3, 2]],
   [[-1, 10], 1, Concat, [1]],  # cat head P5
   [-1, 3, C3, [1024, False]],  # 23 (P5/32-large)

   [[17, 20, 23], 1, Detect, [nc, anchors]],  # Detect(P3, P4, P5)
  ]

```

- 删除`/content/yolov5/train.py`, 上传你更改过的`train.py`
- 主要更改了以下内容
![](https://cdn.jsdelivr.net/gh/clearyup/picgo/img/20210519212553.png)
- 将生成的结果文件夹路径设置为`谷歌云盘的`, 避免断开连接拿不到训练结果
![](https://cdn.jsdelivr.net/gh/clearyup/picgo/img/20210520162946.png)

### 开始训练
- `batch-size`是批处理大小, 每批处理的数目, 比如有`3200`张图片, `batch-size`设置为`32`, 那数据会被分成`100`批, 每批 `32`张图片进行处理
- `epochs`是将所有的批次图片训练完就是`1`次`epochs`, 也叫`迭代次数`

```bash
!python /content/yolov5/train.py --data mytrain.yaml --cfg yolov5s.yaml --weights weights/yolov5s.pt --epochs 100 --batch-size 16

```

### 训练结果分析
- 训练过程的参数如下
- `P`是查准率, `R`是查全率,  `mAp@.5`和`mAp@.5:.95`是用来评估模型的, 这些值越接近`1`越好, 类似`0.994`这种就比较好
  
![](https://cdn.jsdelivr.net/gh/clearyup/picgo/img/20210519213749.png)
- 训练完成会生成一个`run`文件,里面的`weight`文件夹有生成的`权重文件`
### 测试模型
- 需要图片指定的是`yolov5/data/images`目录下, 不然检测不到
```
!python /content/yolov5/test.py   --augment --save-json

```

### 开始预测结果
- 使用训练好的权重文件`best.pt`对数据进行预测   

```bash
!python /content/yolov5/detect.py --save-json --save-txt --save-conf --nosave --augment
``` 

- detect.py代码

```python

import argparse
import time
from pathlib import Path

import json
import cv2
import torch
import torch.backends.cudnn as cudnn

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path, save_one_box
from utils.plots import colors, plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized


def detect(opt):
    source, weights, view_img, save_txt, imgsz = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size
    save_img = not opt.nosave and not source.endswith('.txt')  # save inference images
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))

    save_json = opt.save_json
    mdict = []
    # Directories
    save_dir = increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size
    names = model.module.names if hasattr(model, 'module') else model.names  # get class names
    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    t0 = time.time()
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=opt.augment)[0]

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, opt.classes, opt.agnostic_nms,
                                   max_det=opt.max_det)
        t2 = time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], f'{i}: ', im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if opt.save_crop else im0  # for opt.save_crop
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xyxy, conf) if opt.save_conf else (cls, *xywh)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')
 
                    
                    if save_json:
                        myxyxy = torch.tensor(xyxy).view(1, 4).view(-1).tolist()
                        img_id = p.stem
                        result_class = int(torch.tensor(cls))
                        result_conf  = float(torch.tensor(conf))
                        content_json = {
                            result_class:[[
                                 img_id,
                                 result_conf,
                                 myxyxy[0],
                                 myxyxy[1],
                                 myxyxy[2],
                                 myxyxy[3],
                                
                                 ]]
                            }
                        mdict.append(content_json)
                        

                    if save_img or opt.save_crop or view_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        label = None if opt.hide_labels else (names[c] if opt.hide_conf else f'{names[c]} {conf:.2f}')
                        plot_one_box(xyxy, im0, label=label, color=colors(c, True), line_thickness=opt.line_thickness)
                        if opt.save_crop:
                            save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)

            # Print time (inference + NMS)
            print(f'{s}Done. ({t2 - t1:.3f}s)')

            # Stream results
            if view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path += '.mp4'
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer.write(im0)

    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        print(f"Results saved to {save_dir}{s}")

    #找到属性名
    def findname(i,arr):
        for n in arr[i]:
            return n

    #合并
    i = 0
    while i < len(mdict):
        n = findname(i,mdict)
        j = i + 1
        while(j < len(mdict)):
            p = findname(j,mdict)
            if(p == n):
                mdict[i][n] = mdict[i][n] + mdict[j][p]
            j+=1
        i+=1

    #去重
    def noRepeat(mjson):
        i = 0
        while(i < len(mjson)):
            j = i + 1
            while(j < len(mjson)):
                if(findname(i,mjson) == findname(j,mjson)):
                    del(mjson[j])
                j+=1
            i+=1

    noRepeat(mdict)

    #归并排序
    def merge(left, right):
        result = []
        while left and right:
            #if left[0] <= right[0]:
            if int(findname(0,left)) <= int(findname(0,right)):
                result.append(left.pop(0))
            else:
                result.append(right.pop(0))
        if left:
            result += left
        if right:
            result += right
        return result

    def merge_sort(L):
        if len(L) <= 1:
            # When D&C to 1 element, just return it
            return L
        mid = len(L) // 2
        left = L[:mid]
        right = L[mid:]

        left = merge_sort(left)
        right = merge_sort(right)
        # conquer sub-problem recursively
        return merge(left, right)
        # return the answer of sub-problem

    mjson = merge_sort(mdict)

    #消除大括号(一)
    def extend(target,source):
        for obj in source:
            target[obj] = source[obj]
        return target

    newjson = extend(mjson[0],mjson[1])
    i=2
    while i < len(mjson):
        newjson = extend(newjson,mjson[i])
        i+=1
    mdict=newjson
    #单引号改双引号
    #mdict = str(newjson).replace("'","\"").replace(r"\n","")
    
    if save_json and len(mdict):
        w = Path(weights[0] if isinstance(weights, list) else weights).stem if weights is not None else ''  # weights
        
        pred_json = str(save_dir / f"{w}_predictions.json")  # predictions json
        print('\nEvaluating pycocotools mAP... saving %s...' % pred_json)
        with open(pred_json, 'w') as f:
            json.dump(mdict, f)

    print(f'Done. ({time.time() - t0:.3f}s)')
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='/content/drive/MyDrive/run/train/exp2/weights/best098.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='/content/yolov5/data/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.001, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.6, help='IOU threshold for NMS')
    parser.add_argument('--max-det', type=int, default=10, help='maximum number of detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-json', action='store_true', help='save results to json')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    opt = parser.parse_args()
    print(opt)
    check_requirements(exclude=('tensorboard', 'pycocotools', 'thop'))

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt']:
                detect(opt=opt)
                strip_optimizer(opt.weights)
        else:
            detect(opt=opt)
```
