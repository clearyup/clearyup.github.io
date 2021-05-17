# Flask


# flask搭建web应用
## 应用简介
这是一个目标检测系统,登录成功后,菜单栏有4个功能,图像预测,图像检测,图像分割,视频中的目标检测,主要使用 python 的 flask web框架 和 imageai目标检测库   
>图像预测(Image Prediction): 向该系统上传一张图片,系统接受图片后会将图片传入卷积神经网络,最后输出图片中的对象名和概率   


>目标检测(Image Detection): 向该系统上传一个图片,系统会将图片传入卷积神经网络,最后输出图片中的对象名和概率,并将对象框选出来生成新的图片显示出来   

>图像分割(Image Extract): 向该系统上传一个图片,系统会将图片传入卷积神经网络,最后输出图片中的对象名和概率,并将对象分割出来生成新的图片,并把这些分割出来的对象图片显示出来

>视频检测(Vedio Detection): 向该系统上传一个视频,系统会将该视频传入卷积神经网络,最后输出一个新的视频,新的视频中有对检测到的目标的框选

## 运行环境
- windows10系统
- python版本： 3.7.8
- 需要的pip包已经导出到requirements.txt文件，在项目文件夹根目录下进入cmd，执行以下命令就可以安装(要在后面讲到的创建虚拟环境后再安装)
```bash
python -m pip install -r requirements.txt
```
- 导出pip包列表的命令
```bash
pip freeze > requirements.txt
```

这里的tensorflow是cpu版本,如果你电脑安装了cuda和cudnn你可以安装tensorflow的gpu版本,运行速度会更快
```bash
pip install tensorflow-gpu==2.4.0   
```

## 搭建过程
### 创建虚拟环境
虚拟环境可以为每一个项目安装独立的 Python 库，这样就可以隔离不同项目之间的 Python 库，也可以隔离项目与操作系统之间的 Python 库,避免发生冲突

新建一个文件夹project,cmd进入这个文件夹执行
```bash
py -3 -m venv venv
```

接着激活这个虚拟环境,执行
```bash
venv\Scripts\activate
```

安装项目需要的依赖环境,执行
```bash
python -m pip install -r requirements.txt
```

创建flask的static目录和templates目录,最后的目录结构如下:

![](https://cdn.jsdelivr.net/gh/clearyup/picgo/img/20210513174429.png)
### 前后端实现
> flask的hello world实现   

- 新建一个app.py,到flask官网复制hello world代码
```python
from flask import Flask
app = Flask(__name__)
@app.route('/')
def hello_world():
    return 'Hello, World!'
```

在vscode中打开cmd，设置开发模式并启动flask服务器

```bash
set FLASK_APP=app.py
set flask_env=development
flask run
```

ctrl点击本地url就可以在浏览器看到 Hello world!

![](https://cdn.jsdelivr.net/gh/clearyup/picgo/img/20210513190748.png)

>路由的实现

- 使用 route() 来把函数绑定到 url ,由于 html 引用的 css 和 js 放在 staitc 目录, html 在 templates 目录,需要配置 static_url_path
- 导入对应的包
- 定义 user 类,生成一个user对象用于登录,登录后信息写入session,如果下次登录前根据email找到了就将信息写入全局对象g
- 根目录会重定向到 login 函数处理登录,输入的邮箱和密码符合后台设置的就跳转到 index.html,否则重定向到当前页面
- index处理时如果没有全局对象g,那就重定向页面到 login 处理

```python
from flask import Flask
from flask import render_template, request, redirect,url_for,session,g
from dataclasses import dataclass
import os
app = Flask(__name__, static_url_path="/")
app.config['SECRET_KEY'] = "sldjfsdf32kdf" 

@dataclass
class User:
    email: str
    password: str
users = [
    User("2010059016@qq.com","123")
]

@app.before_request
def before_request():
    g.user = None
    if 'email' in session:
        user = [u for u in users if u.email==session['email']][0]
        g.user = user

@app.route('/')
def hello_world():
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        session.pop('email',None)
        email = request.form.get('email',None)
        password = request.form.get('password',None)
        user = [u for u in users if u.email==email]
        if len(user) > 0:
            user = user[0]
        if user and user.password == password:
            session['email'] = email
            return redirect('index')
    return render_template('login.html')

@app.route('/index')
def index():
    if not g.user:
        return redirect(url_for('login'))
    
    return render_template('index.html')

```

- 登录页面   

![](https://cdn.jsdelivr.net/gh/clearyup/picgo/img/20210513194936.png)   

- 登录成功后会跳转到 index.html ,这也是系统的首页

![](https://cdn.jsdelivr.net/gh/clearyup/picgo/img/20210513195101.png)   

- 4个功能实现过程都类似,这里选择图像检测进行说明

![](https://cdn.jsdelivr.net/gh/clearyup/picgo/img/20210513200517.png)

- 你会在 image_predection.html 中上传图片,设置 post 请求和 enctype ,不然 flask 接受不了文件,将 aciton 设置为后台路由,由路由绑定的函数处理
```html
<form method="post"  action= "/detection_upfile" enctype = multipart/form-data class="formfile">
  <input type="file" name="file" style="margin:20px 0;" accept="image/*" id="file"  />
  <input type="submit" value="上传" style="margin:20px 0;margin-left: 170px;" />
{% if filename %}
  <img src="/input/{{filename}}" width="500" />
{% endif %}
</form>
```

- 后台 flask 代码, 请求为 post 时,获得上传文件名,保存到 input 文件夹
- 加载 YOLOv3 模型,将图片输入,最后将输出结果保存到 output 字典中,将上传图片的名字和输出的字典传递到 image_detection.html,刷新页面


```python
@app.route('/detection_upfile', methods=['POST'])
def detection_upfile():
    if request.method == 'POST':
        f = request.files['file']
        filename = secure_filename(f.filename)
        output = OrderedDict()
        path = basedir+"/static/input/"
        file_path = path + filename
        f.save(file_path)
        print('上传成功！')
        print(filename)
        execution_path = os.getcwd()
        detector = ObjectDetection()
        detector.setModelTypeAsYOLOv3()
        detector.setModelPath( os.path.join(execution_path , model_path+"yolo.h5"))
        detector.loadModel()
        detections = detector.detectObjectsFromImage(input_image=os.path.join(execution_path , input_path+filename), output_image_path=os.path.join(execution_path , output_path+filename), minimum_percentage_probability=30)
        for eachObject in detections:
             output[eachObject["name"]] = eachObject["percentage_probability"]
        return render_template('image_detection.html',filename=filename,output=output)

```

- image_detection 拿到后台传过来的 filename 和 output,使用 Jinja模板引擎对图片进行输出并循环显示字典中的输出结果
- filename 为空时不显示图,不然会有默认的 img 标签显示   
- output 不为空的时候循环输出字典中的 key和value
```html
{% if filename %}
  <img src="/output/{{filename}}" width="500" class="outputimage"/>
{% endif %}
 
<table>
  {% if output %}
    {% for key, value in output.items() %}
    <tr>
        <td>{{ key }}</td>
        <td>{{ value }}</td>
    </tr>
    {% endfor %}
  {% endif %}
</table>
```

- 输出样例:    

![](https://cdn.jsdelivr.net/gh/clearyup/picgo/img/20210513204559.png)

>退出登录

- 点击 sign out就会退出系统，会重定向到login.html
