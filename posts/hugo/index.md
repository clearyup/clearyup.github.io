# Hugo 建站




# 使用hugo搭建网站

首先,这里介绍一下传统的网站搭建,你需要的是

* 一台服务器(一般是阿里云之类的云服务器,搭载的是Linux系统)
* 一个开源的网站框架或者模板(或选择自己开发网站前后端)
* 一个网站域名(域名国内需要备案,配置后域名可以访问网站,或者不用域名直接服务器公网ip访问网站)

你通过对云服务器进行端口,防火墙,ssh等一些配置,安装数据库,tomcat等软件后,将开源网站框架部署到云服务器上,域名可选或不选,这里你通过云服务器的公网ip就可以访问到网站,有域名将域名指向网站,添加dns解析就可以通过域名访问网站.

但是问题有以下几个

* 云服务器需要进行一系列繁琐的安全配置,常用端口可能需要经常更换,因为公网服务器一直在被世界各地的人攻击
* 服务器需要一直续费,不然到期后网站挂掉
* 图片,js等静态资源加载慢(服务器配置好忽略)
* 手动前后端管理

# 另一种选择 hugo

hugo是一个基于`go语言`的框架,拥有最快速的静态网页生成速度,这里我们用它来搭建博客网站,使用github pages作为网站的托管,部署到vercel进行网站页面更新以及cdn加速,文章图片使用picgo+github仓库+jsdelivr组合图床,使用开源的utterances作为评论模块,使用algolia作为网站内部搜索引擎

这样做的好处

* 之后你可以专注写本地markdown文章,然后通过hugo生成html静态网页,使用git推送到github仓库
* 无需花时间在服务器上,无需续费网站一直存在,用户名.github.io永久域名访问网站
* vercel帮你完成网页更新以及cdn网站加速
* 静态资源加速访问,图片在github仓库通过jsdelivr加速访问
* 写的博客直接发给别人也不需要打包图片,图片是超链接
* 拥有评论和搜索功能 

基本上满足了我对于博客网站所有的需求,这里挂一张本网站在谷歌网页分析的评分,满分!

![image-20210416220534186](https://cdn.jsdelivr.net/gh/clearyup/picgo/img/20210416220541.png)

# 配置步骤

## 第一步创建仓库

创建github pages仓库,到github新建一个public仓库,仓库名为**你的用户名.github.io**,例如我的GitHub用户名为clearyup,那我的仓库名为clearyup.github.io

## 第二步下载hugo

[hugoGitHub仓库](https://github.com/gohugoio/hugo)在下载hugo 0.81.extended版本,将解压后的bin目录加载到系统path路径,cmd检验hugo是否安装成功

![image-20210416221237155](https://cdn.jsdelivr.net/gh/clearyup/picgo/img/20210416221237.png)


![image-20210416221331657](https://cdn.jsdelivr.net/gh/clearyup/picgo/img/20210416221332.png)

##  第三步 创建网站
* 在你的hugo目录下进入cmd 新建一个网站根目录
```bash
hugo new site myblog
```

![c](https://cdn.jsdelivr.net/gh/clearyup/picgo/img/20210417102917.png)

* cmd中使用hugo命令创建第一篇文章
```bash
cd myblog
hugo new posts/myfirst.md
```
这样在你的myblog/content/posts文件夹下就生成了myfirst.md文件,这个文件是根据myblog/archetypes目录下的default.md模板来创建新的md文件,你可以修改这个default.md文件作为新建文件的模板

{{< admonition type=note title="注意!" open=ture >}}
这里的文章头部有一个draft:true表示文章是草稿,使用hugo构建网站的时候不会将此文章生成静态html,建议设置成false
{{< /admonition >}}

* 设置主题

你可以到[hugo主题库](https://themes.gohugo.io/)选择一个主题,找到它的git仓库地址,然后使用git命令安装(git要设置好了代理)
```bash
cd themes
git submodule add https://github.com/luizdepra/hugo-coder.git themes/hugo-coder
```
这里是 git submodule add 主题仓库地址,你可以选择你喜欢的主题更换这个命令

* 配置主题

在myblog的config.toml中设置主题为你的主题,建议按照主题文档[配置文件](https://github.com/luizdepra/hugo-coder/blob/master/exampleSite/config.toml)进行详细配置
```toml
theme = "hugo-coder"
baseURL = "https://你的github用户名.github.io"
```
* cmd本地运行
```bash
hugo server
```
运行成功之后到http://localhost:1313/本地浏览网站

##  部署到github pages

* 生成静态网页
在myblog根目录下使用cmd命令
```bash
hugo -D
```
为确保所有文章都生成静态网页,使用hugo -D是草稿也生成静态网页

* 推送到远程仓库
```bash
cd public
git init
git add .
git commit -m "first commit"
git remote add origin 仓库地址
git push -u origin master
```
之后输入github用户名和密码就可以进行push,git有关配置参考我另一篇博客**git配置**

最后用户名.github.io访问你的网站

## 部署到Vercel
部署到vercel后网站拥有cdn加速,当你用git推送新文章到github仓库,vercel会帮你重新发布新页面,网站刷新后就可以看到新页面的内容   

这里的仓库建议不设置为`用户名.github.io`这个仓库, 因为这个仓库是`hugo`生成的`public`文件 push 上来的, 没有`config.toml`配置文件, `vercel`不能识别为hugo类型的仓库, 就算你上传 `config.toml`文件到
`用户名.github.io`这个仓库,还是不能识别

建议新建一个`blogBackup`仓库,将hugo本地网站带有`config.toml`文件的根目录设为`git仓库`,推送到这个 `blogBackup`仓库, 这个是可以被`vercel`识别为hugo类型的仓库

### 第一步
注册Vercel账号,直接使用GitHub账号登录就可以了,然后允许访问所有仓库

### 第二步
选择`Import Git Repository`，选择`blogBaskup`这个仓库即可。在`Environment Variables`这里，添加一个变量，`HUGO_VERSION = 0.80.0` 以便正确编译,随后直接部署就可以了

### 第三步
如果你想自定义域名,在`domain`那里添加你要自定义的域名,更改域名的`dns解析`为`vercel`的就可以了   


> 参考视频   
(https://www.bilibili.com/video/BV147411M7C7)  
(https://www.bilibili.com/video/BV1q4411i7gL/?spm_id_from=333.788.recommend_more_video.-1)
>

## 添加 .nojekyll 文件 
>避免 github 认为你使用了jekyll构建项目，添加` .nojekyll`文件到`用户名.github.io`仓库, 否则可能受到 build failure 的邮件

## 重构GithubPages
>主要是删除 `public`文件,使用`hugo -D`重新生成一个新的`public` ,再进行推送, 否则还是会重构失败,`githubpages`会显示`404`



