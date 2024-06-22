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

## git bash配置(windows)

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

# git删除仓库历史中的文件
> 有这样的场景，在使用git进行代码push的时候，不小心把本地的编译好的可执行文件或者一些忘记加入到.gitignore的文件也上传到远程仓库，如何解决呢？

- 保留本地工作区的这个目录，从暂存区删除这个文件夹，这样就取消这个文件夹的关联了，没关联的文件加入.gitignore才会生效
```shell
git rm --cached /tmp -r
```

- 如果本地的也不需要保留,这会将工作区和暂存区的此目录取消关联、删除
```shell
git rm /tmp -r
```

- 接下来你只需要重新提交代码到仓库并push到远程仓库就OK

## 远程仓库和本地仓库的清理
- 上面的步骤执行完了后，真的ok了吗？其实没有，这个文件依然存在于你的.git本地和远程仓库里的历史版本，只是当前最新版本没有这个文件罢了

因此如果这个文件确认是不需要进行版本管理的文件，而且是错误的加入到了git仓库管理，那么为了仓库的容量大小和避免所有pull的人都拉取这样没有意义的文件，我们需要在本地仓库和远程仓库彻底清理这样的文件

以下有两种方案，如果是个人项目，不关心历史提交的版本和tag之类的，直接删除本地仓库.git,重新git init再强制推送就行，如果是团队项目就用下面的那种，从所有提交历史中删除要的文件夹
### 个人项目（粗糙的解决方法）
#### 步骤 1：删除本地 .git 目录

首先，在项目根目录中删除 `.git` 目录：

```shell
rm -rf .git
```

#### 步骤 2：重新初始化 Git 仓库

重新初始化一个新的 Git 仓库：
```shell
git init
```

#### 步骤 3：添加所有文件并进行首次提交

添加所有文件并进行首次提交：
```shell
git add .
git commit -m "Initial commit after cleanup"
```

#### 步骤 4：重新关联远程仓库

将新的本地仓库与远程仓库关联：
```shell
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
```

#### 步骤 5：强制推送到远程仓库

将本地仓库的内容强制推送到远程仓库（这会覆盖远程仓库的所有内容）：
```shell
git push --force origin master
```


### 团队项目（最终的方案）

下面是使用 `filter-branch` 从历史记录中移除 `/tmp` 目录的步骤：

#### 步骤 1：备份你的仓库

首先，备份你的仓库，以防任何操作出现问题：
```shell
git clone /path/to/your/local/repository /path/to/your/backup/repository

```


#### 步骤 2：使用 `filter-branch` 移除目录 /tmp
```shell
git filter-branch --force --index-filter 'git rm -r --cached --ignore-unmatch ./tmp/*' --prune-empty --tag-name-filter cat -- --all

```

解释：

- `--force` 强制重写。
- `--index-filter` 运行一个命令以更新每个提交的索引。
- `git rm -r --cached --ignore-unmatch /tmp` 从索引中删除 `/tmp` 目录。
- `--prune-empty` 删除所有没有文件变化的提交。
- `--tag-name-filter cat` 保持标签名称不变。
- `-- --all` 应用于所有分支。

#### 步骤 3：清理和重新打包

在重写历史后，运行以下命令以清理和重新打包 Git 仓库：

```shell

rm -rf .git/refs/original
git reflog expire --expire=now --all
git gc --prune=now
git gc --aggressive --prune=now

```


#### 步骤 4：强制推送修改到远程仓库

由于历史已经被重写，你需要强制推送这些更改到远程仓库：

```shell
git push --force --all
git push --force --tags

```


