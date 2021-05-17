# FFmpeg安装使用


# FFmpeg
FFmpeg是一套可以用来记录转换数字音频,视频，并能将其转换为流的开源计算机程序
## 安装
- 从github仓库下载[FFmpeg](https://github.com/BtbN/FFmpeg-Builds/releases)
- 解压后将bin目录加载到系统path路径
- win+r输入`sysdm.cpl`,进入高级->环境变量      
    ![1](https://cdn.jsdelivr.net/gh/clearyup/picgo/img/20210426204139.png)
	
- cmd查看是否安装好了,和图中一样就是安装好了
	![2](https://cdn.jsdelivr.net/gh/clearyup/picgo/img/20210426204208.png)
	
## 使用
### 视频提取音频
这里介绍视频提取音频的使用,在存放demo.flv视频的文件夹进入cmd执行下列命令   
`视频最好更改为英文名`

```bash
ffmpeg -i demo.flv -f mp3 -vn demo.mp3
```
### 合并多个视频
- 先创建一个文本文件，包含需要拼接的文件的列表，此文档和3个视频同一目录下

filelist.txt
>file 'input1.pm4'   
>file 'input2.pm4'      
>file 'input3.pm4'     

- 在目录下进入cmd执行以下命令行
```bash
ffmpeg -f concat -i filelist.txt -c copy output.mp4
```
### 按照时间切割音频/视频
- 切割音频   

在存放input.mp3文件的目录下进入cmd执行以下命令行
```bash
ffmpeg -i input.mp3 -ss 00:03:10 -to 00:05:10 output.mp3
```
- 切割视频   

在存放input.mp4文件的目录下进入cmd执行以下命令行   
```bash
ffmpeg -i input.mp4 -ss 00:03:10 -to 00:05:10 -c copy output.mp4
```

### 视频转GIF
- 将视频中的一部分转换为GIF,从视频中第3秒开始截取时长为3秒的片段转化为 gif

```
ffmpeg -t 3 -ss 00:00:03 -i test.mp4 test-clip.gif
```

- 转化高质量 GIF

```
ffmpeg -i test.mp4 -b 2048k test.gif
```

- 将 GIF 转换为 MP4

```
ffmpeg -f gif -i test.gif test.mp4
```

- 移除视频中的音频(静音),-an 就是禁止音频输出

```
ffmpeg -i input.mov -an mute-output.mov
```
