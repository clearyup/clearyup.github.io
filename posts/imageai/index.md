# imageai

# ImageAi   
image是一个python库，用于图像预测，对象检测，视频对象检测和视频对象跟踪
## 运行ImageAi
### winsdows所需环境
-   **Python** 3.7.6 , [Download Python](https://www.python.org/downloads/release/python-376//)
    
-   **pip** , [Download pip](https://pypi.python.org/pypi/pip/) 
```bash
python -m pip install --upgrade pip
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
```


-   **Tensorflow** 2.4.0-CPU
```bash
pip install tensorflow==2.4.0
```

or **Tensorflow-GPU** if you have a `NVIDIA GPU` with `CUDA` and `cuDNN` installed

```bash
pip install tensorflow-gpu==2.4.0
```



-   **Other Dependencies**
```bash
pip install keras==2.4.3 numpy==1.19.3 pillow==7.0.0 scipy==1.4.1 h5py==2.10.0 matplotlib==3.3.2 opencv-python keras-resnet==0.2.0
```
-   **ImageAI**
```bash
pip install imageai --upgrade
```

github仓库地址
```
https://github.com/OlafenwaMoses/ImageAI
```

### colab环境
ImageAI
```bash
!pip install imageai
```

下面的训练模型和训练图片也可以手动上传或关联google云盘   
yolo-tiny.h5
```bash
!wget https://github.com/OlafenwaMoses/ImageAI/releases/download/1.0/yolo-tiny.h5
```

testImage

```bash
!wget https://github.com/OlafenwaMoses/ImageAI/raw/master/data-images/image2.jpg
```
### 运行
- vscode中运行,确保有`Python`扩展和`TabNine`扩展,并且选择了Python3.7解释器   
没选的control+shift+p然后输入Python找到Select Interpreter回车勾选Python3.7解释器后回车

![](https://cdn.jsdelivr.net/gh/clearyup/picgo/img/20210503105956.png)
![](https://cdn.jsdelivr.net/gh/clearyup/picgo/img/20210503110404.png)

- 新建一个文件夹imageai 鼠标右键`git bash here`执行
```bash
git init
git clone https://github.com/OlafenwaMoses/ImageAI.git
```

- 新建一个文件夹test,鼠标右键用vscode打开

#### 图像预测
图像预测可以预测图片中的内容，目前支持4种算法用来进行图像预测`MobileNetV2`, `ResNet50`, `InceptionV3` and `DenseNet121`,
- 下载训练模型移到test这个文件夹   
[Download ResNet50 Model](https://github.com/OlafenwaMoses/ImageAI/releases/download/essentials-v5/resnet50_imagenet_tf.2.0.h5/)   
- 从仓库文件夹imageai中选取训练图片移到test文件夹
- 新建firstPrediction.py文件,copy以下代码

```python
from imageai.Prediction import ImagePrediction
import os

# 获得当前路径
execution_path = os.getcwd()

prediction = ImagePrediction()
# 加载AsResNet50模型
prediction.setModelTypeAsResNet50()
prediction.setModelPath(os.path.join(execution_path, "resnet50_imagenet_tf.2.0.h5"))

# 加载模型
prediction.loadModel()

# 返回预测结果
predictions, probabilities = prediction.predictImage(os.path.join(execution_path, "12.jpg"), result_count=5)
for eachPrediction, eachProbability in zip(predictions, probabilities):
    print(eachPrediction , " : " , eachProbability)
    print("--------------------------------")
```

- vscode控制台 执行
```bash
python firstPrediction.py
```
- 图片

![](https://cdn.jsdelivr.net/gh/clearyup/picgo/img/20210505191935.jpg)
- 训练结果
```txt
mountain_bike  :  59.61750149726868
--------------------------------
bicycle-built-for-two  :  20.329207181930542
--------------------------------
jinrikisha  :  6.335373967885971
--------------------------------
crash_helmet  :  1.6147501766681671
--------------------------------
tricycle  :  0.8636814542114735
--------------------------------
```
可以看到识别出了最主要的山地自行车和两个骑自行车的人，但还识别出了三个概率比较小的黄包车,安全帽,三轮车
#### 目标检测
目标预测可以框出图片中的目标，并识别出来，目前支持`RetinaNet`, `YOLOv3` and `TinyYOLOv3`三种深度学习算法
- 下载模型文件移入test文件夹
[YOLOv3 Model - yolo.h5](https://github.com/OlafenwaMoses/ImageAI/releases/download/1.0/yolo.h5/)   
- 从仓库文件夹imageai选取训练图片移入test文件夹
- 新建firstDetection.py文件 copy以下代码

```python
from imageai.Detection import ObjectDetection
import os

execution_path = os.getcwd()

detector = ObjectDetection()
detector.setModelTypeAsYOLOv3()
detector.setModelPath( os.path.join(execution_path , "yolo.h5"))

detector.loadModel()
detections = detector.detectObjectsFromImage(input_image=os.path.join(execution_path , "10.jpg"), output_image_path=os.path.join(execution_path , "New11.jpg"), minimum_percentage_probability=30)

for eachObject in detections:
    print(eachObject["name"] , " : ", eachObject["percentage_probability"], " : ", eachObject["box_points"] )
    print("--------------------------------")
```
- vscode控制台 执行
```bash
python firstDetection.py
```
- 图片

![](https://cdn.jsdelivr.net/gh/clearyup/picgo/img/20210503112414.jpg)
- 训练结果   

![](https://cdn.jsdelivr.net/gh/clearyup/picgo/img/20210503215253.jpg)   

```txt
dog  :  98.87264966964722  :  [116, 103, 311, 373]
--------------------------------
dog  :  99.15174841880798  :  [338, 69, 488, 409]
--------------------------------
dog  :  98.93686175346375  :  [503, 154, 638, 386]
--------------------------------
```

在目标检测源代码的基础上,增加一个参数就可以将识别出来的对象从原图分割出来保存到新的文件夹，增加的代码在这一行 的`extract_detected_objects=True`

```python
detections, objects_path = detector.detectObjectsFromImage(input_image=os.path.join(execution_path , "10.jpg"), output_image_path=os.path.join(execution_path , "new10.jpg"), minimum_percentage_probability=30,  extract_detected_objects=True)
```

重新运行的结果:

![image-20210505194514741](https://cdn.jsdelivr.net/gh/clearyup/picgo/img/image-20210505194514741.png)

#### 视频检测
ImageAI可以提供方便的视频检测跟踪和视频分析功能,目前支持三种深度学习算法`RetinaNet`,`YOLOv3`,`TinyYOLOv3`
- 下载模型文件移入test文件夹
[TinyYOLOv3 Model - yolo-tiny.h5](https://github.com/OlafenwaMoses/ImageAI/releases/download/1.0/yolo.h5/)
- 从仓库文件夹imageai选取训练视频traffic.mp4移入test文件夹
- 新建firstVedioDetection.py文件 copy以下代码

```python
from imageai.Detection import VideoObjectDetection
import os

execution_path = os.getcwd()

detector = VideoObjectDetection()
detector.setModelTypeAsTinyYOLOv3()
detector.setModelPath( os.path.join(execution_path , "yolo-tiny.h5"))
detector.loadModel()

# frames_per_second保存的视频每秒帧数,视频检测秒数
video_path = detector.detectObjectsFromVideo(input_file_path=os.path.join(execution_path, "traffic.mp4"),
                                output_file_path=os.path.join(execution_path, "traffic_detected_all")
                                , frames_per_second=20, log_progress=True,detection_timeout=2)

```

这里的视频检测代码是输出2s的视频,每秒20帧,因为使用电脑的cpu跑的,如果你安装好了`cuda`和`cuDNN`和`GPU`版本的Tensorflow库你可以设置更多的输出时长和视频帧率   

>以下是在colab用CPU跑的每秒30帧的10s输出视频结果,一共用将近`20分钟`,设置参数时可以做个参考

![image-20210505200643955](https://cdn.jsdelivr.net/gh/clearyup/picgo/img/image-20210505200643955.png)

执行代码:
```bash
python firstVidioDetection.py
```

原视频片段:   

![](https://cdn.jsdelivr.net/gh/clearyup/picgo/img/traffic.gif)

识别后:   

![](https://cdn.jsdelivr.net/gh/clearyup/picgo/img/all.gif)

修改代码,只识别摩托车自行车和人

```python
from imageai.Detection import VideoObjectDetection
import os

execution_path = os.getcwd()

detector = VideoObjectDetection()
detector.setModelTypeAsTinyYOLOv3()
detector.setModelPath( os.path.join(execution_path , "yolo-tiny.h5"))
detector.loadModel()

custom_objects = detector.CustomObjects(person=True, bicycle=True, motorcycle=True)

video_path = detector.detectCustomObjectsFromVideo(
                custom_objects=custom_objects,
                input_file_path=os.path.join(execution_path, "traffic.mp4"),
                output_file_path=os.path.join(execution_path, "traffic_detected_cycle"),
                frames_per_second=20, log_progress=True,detection_timeout=2)
                
                

```

修改后的运行结果:   

![](https://cdn.jsdelivr.net/gh/clearyup/picgo/img/20210505204915.gif)
## 运行原理
这里的图像预测用到了`ResNet50`算法,目标检测用到了`YOLOv3`算法,视频检测用到了`TinyYOLOv3`算法,下面分别将它们的实现原理
### ResNet
#### 传统深度学习面临的问题
神经网络层次越深，学习效果不一定越好，当模型层数增加到某种程度的时候，模型的效果会不升反降，深度模型发生`退化`情况,发生这种情况有以下几点原因：
1. 过拟合   
模型层数过多,训练模型过于复杂,训练样本少,就会过分去拟合了训练集，放大了差异性，衰弱了共性,导致`过拟合`
3. 梯度消失和爆炸   

 >**梯度**的本意是一个向量（矢量），表示某一函数在该点处的方向导数沿着该方向取得最大值，即函数在该点处沿着该方向（此**梯度**的方向）变化最快，变化率最大（为该**梯度**的模）    

>**反向传播**（英语：Backpropagation，缩写为BP）该方法计算对网络中所有权重计算损失函数的梯度。这个梯度会反馈给最优化方法，用来更新权值以最小化损失函数。 在神经网络上执行梯度下降法的主要算法。该算法会先按前向传播方式计算（并缓存）每个节点的输出值，然后再按反向传播遍历图的方式计算损失函数值相对于每个参数的偏导数。

>假设我们现在需要计算一个函数$f(x,y,z)=(x+y)*z$ 在$x=-2,y=5,z=-4$的`梯度`,计算流程如下:
>- 向前传播计算: $f(x=2,y=5,z=-4)$结果为-12
>- 反向传播计算: 令$q=x+y$   
>  
>- ${\frac {df}{dz}}=q=x+y=3$   
>        
>- ${\frac {df}{dx}}={\frac {df}{dq}}\cdot{\frac {dq}{dx}}=z\cdot1=-4$    
>  
>- ${\frac {df}{dy}}={\frac {df}{dq}}\cdot{\frac {dq}{dx}}=z\cdot1=-4$

这里假设输出端初始的梯度为 1，也就是输出端对自身求导等于 1。当神经网络反向传播的时候，不难发现，在输出端梯度的模值，经过回传扩大了3或缩小4倍 
这是由于反向传播结果的**数值大小**不止取决于求导的式子，很大程度上也取决于**输入的模值**

那么当每次输入的模值都大于1时，训练10000次，每次扩大3倍，这个梯度会变成$3^{10000}$,会发生梯度爆炸 

那么当每次输入的模值都小于1时，训练10000次，每次缩小4倍，这个梯度会变成$4^{-10000}$,会发生梯度消失

#### ResNet的提出(残差网络)
如果深层网络后面的层都是`恒等映射`,那么模型就可以转换为一个浅层网络,问题是如何得到恒等映射?

恒等映射需要 $H(x)=x$,但是实现非常困难,因此残差网络是将网络设计为$H(x)=F(x)+x$,这样就把问题转化为学习一个`残差函数`$F(x)=H(x)-x$   

只要我们$F(x)=0$,就构成了恒等映射 $H(x)=x$,于是就有了`Residual block`结构   

![](https://cdn.jsdelivr.net/gh/clearyup/picgo/img/20210509111552.png)

在上图的残差网络结构图中，通过`shortcut connections（捷径连接）`的方式，直接把输入$x$ 传到输出作为初始结果，输出结果为 $H(x)=F(x)+x$ ，当$F(x)=0$ 时，那么 $H(x)=x$ ，也就是上面所提到的恒等映射。于是，ResNet相当于将学习目标改变了，不再是学习一个完整的输出，而是目标值 $H(X) 和 x$ 的差值，也就是所谓的残差 $F(x):= H(x)-x$ ，因此，后面的训练目标就是要将残差结果逼近于0，使到随着网络加深，准确率不下降。

这种残差跳跃式的结构，打破了传统的神经网络n-1层的输出只能给n层作为输入的惯例，使某一层的输出可以直接跨过几层作为后面某一层的输入，其意义在于为叠加多层网络而使得整个学习模型的错误率不降反升的难题提供了新的方向   

至此，神经网络的层数可以超越之前的约束，达到几十层、上百层甚至千层，为高级语义特征提取和分类提供了可行性

>在2015年的[ImageNet](https://baike.baidu.com/item/ImageNet/17752829)图像识别大赛中，作者何恺明和他的团队用“图像识别深度差残学习”系统，击败谷歌、英特尔、高通等业界团队，荣获第一


#### ResNet图像预测
1. 卷积(convolution),彩色图片转化为RGB三个通道的矩阵,对每个矩阵进行卷积然后合并输出,提取图像的特征
2. 池化（Pooling），有的地方也称汇聚，实际是一个下采样（Down-sample）过程。由于输入的图片尺寸可能比较大，这时候，我们需要下采样，减小图片尺寸
3. 全连接层把卷积层和池化层的输出展开成一维形式，在后面接上与普通网络结构相同的回归网络或者分类网络,一般接在池化层后面，这一层的输出即为我们神经网络运行的结果

### YOLOv3

YOLOv3目标检测分为两步：
1. 确定检测对象的位置
2. 对检测对象进行分类，识别出图像的内容，定位识别出的目标的位置，框出，框出目标需要4个参数：中心点的横纵坐标，框的宽和高

### 大致流程
图片进入YOLOv3网络,图片首先会被调整到 416*416 的大小,输入Darknet53进行图像的特征提取,为了防止失真会在图片边缘加上灰条,之后图片会分成三个网格图片（13×13，26×26，52×52）   

![](https://cdn.jsdelivr.net/gh/clearyup/picgo/img/20210509201740.png)   
网格中每个网格点都有3个先验框,通过调参来穷举,试探框内是否有目标,有目标就框选出来   

![](https://cdn.jsdelivr.net/gh/clearyup/picgo/img/20210509212109.jpg)   


13x13的特征层会进行5次卷积,会有两个方向的输出,一个输出会向上采样和26x26的特征层进行结合,另一个输出会进行3x3的卷积和1x1的卷积,之后进行通道调整,调整完的结果就是一个特征层的预测结果:   
(batch_size,13,13,75)    
13x13的网格有169个网格点   
75 可以分解为 3x25   
3表示每个网格点有3个先验框   
其中25包含了 4+1+20, 分别代表4: x_offset, y_offset, height和width, 1:置信度, 20:分类结果

13x13的特征层向上采样的输出会和 26x26 的特征层进行结合, 结合之后再进行5次卷积进行特征提取,会有两个方向的输出,一个输出会向上采样和52x52的特征层进行结合,另一个输出会进行3x3的卷积和1x1的卷积,之后进行通道调整,调整完的结果就是一个特征层的预测结果:   
(batch_size,26,26,75),类似13x13

之后是 26x26向上采样的输出会和 52x52 的特征层进行结合, 结合之后再进行5次卷积进行特征提取,最后进行进行3x3的卷积和1x1的卷积,之后进行通道调整,调整完的结果就是一个特征层的预测结果:   
(batch_size,52,52,75),类似13x13


