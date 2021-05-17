# Gitproxy

# git设置



通过设置git代理,你clone和push的操作处理速度会由20kb/s变成7.8Mb/s,因此有代理的设置这个很有必要

{{< admonition type=failure title="注意!" open=ture >}}
以后每次使用git命令或者git脚本记得代理是开启的
{{< /admonition >}}



## 设置代理

### 全局代理

```sh
git config --global http.proxy 127.0.0.1:7890
```

这里的7890是clash的代理端口,如果你不是clash,设置你自己的代理端口

### 局部代理 在仓库内使用

```sh
git config --local http.proxy 127.0.0.1:7890
```



## 取消设置代理

### 取消全局代理

```sh
git config --global --unset http.proxy
```

### 取消局部代理

```sh
git config --local --unset http.proxy
```



## 设置用户名密码

hugo每次push都需要输入用户名密码,我们可以生成用户名密码信息,以后都不需要输入

### 具体代码

```shell
git config --global credential.helper store
```

在你的本地仓库文件夹执行这条命令,然后push一次,输入一次用户名和密码,以后都不需要了

## 设置换行处理

```sh
git config --global core.autocrlf false
```

## git bash配置
在 `C:\Users\y2010\.minttyrc`中修改
```
BoldAsFont=-1

CursorType=block
CursorBlinks=no
Font=Fira Code


FontHeight=13
Transparency=low
FontSmoothing=default
Locale=C
Charset=UTF-8
Columns=88
Rows=26
OpaqueWhenFocused=no
Scrollbar=none
Language=zh_CN

ForegroundColour=131,148,150
BackgroundColour=0,43,54
CursorColour=220,130,71

BoldBlack=128,128,128
Red=255,64,40
BoldRed=255,128,64
Green=64,200,64
BoldGreen=64,255,64
Yellow=190,190,0
BoldYellow=255,255,64
Blue=0,128,255
BoldBlue=128,160,255
Magenta=211,54,130
BoldMagenta=255,128,255
Cyan=64,190,190
BoldCyan=128,255,255
White=200,200,200
BoldWhite=255,255,255

BellTaskbar=no
Term=xterm
FontWeight=400
FontIsBold=no
```


