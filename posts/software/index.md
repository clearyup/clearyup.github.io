# Software

## 常用软件
>[Chrome](https://www.google.com/intl/zh-CN/chrome/)

>Bandizip,防火墙设置拦截更新程序,`U盘安装`   
>[7zip](https://sparanoid.com/lab/7z/)    
  
>[Snipaste](https://zh.snipaste.com/)

>[githubClash](https://github.com/Fndroid/clash_for_windows_pkg),设置自动更新订阅    
> [蓝奏云Clash](https://clearstack.lanzous.com/icd96oppxvi)   
> mixin开启增强模式
```yaml
mixin: 
   dns:
     enable: true
     listen: 127.0.0.1:53
     default-nameserver:
       - 223.5.5.5
       - 1.0.0.1
     ipv6: false
     enhanced-mode: fake-ip #redir-host
     nameserver:
       - tls://223.5.5.5:853
       - https://dns.pub/dns-query
       - https://dns.alidns.com/dns-query
     fallback: #if https-dns not working,try (DOT)tls://
       - https://1.0.0.1/dns-query  #tls://1.0.0.1:853
       - https://public.dns.iij.jp/dns-query
       - https://dns.google/dns-query #tls://8.8.4.4:853
     fallback-filter:
       geoip: true
       ipcidr:
         - 240.0.0.0/4
         - 0.0.0.0/32
         - 127.0.0.1/32
       domain:
         - +.google.com
         - +.facebook.com
         - +.youtube.com
         - +.xn--ngstr-lra8j.com
         - +.google.cn
         - +.googleapis.cn
```
![](https://cdn.jsdelivr.net/gh/clearyup/picgo/img/20210503192522.png)




>[Potplayer](https://clearstack.lanzous.com/iGFFlorle5g) 无边框设置

>[腾讯桌面管理软件](https://guanjia.qq.com/product/zmzl/)  
>[Coodesker](https://github.com/coodesker/coodesker-desktop/releases)

>[OBS](https://obsproject.com/) 
>[Gif动图录制软件](https://clearstack.lanzous.com/iUsZQormcoj)

>[uTools](https://u.tools/)
内部安装everything

>XMind ZEN `u盘安装`

>[百度网盘](https://pan.baidu.com/download#pan)

>[IDM](https://clearstack.lanzous.com/ijbryoppzze)  
>[绿色版IDM](https://clearstack.lanzoui.com/ihB7Qp1tgpg)   
>[手动破解版IDM](https://clearstack.lanzoui.com/iJpX9p1tgub)   
>[Motrix](https://github.com/agalwood/Motrix/releases) [dht.bat](https://clearstack.lanzous.com/iWfJrorgccb)    
>[YAAMconfig](chrome-extension://dennnbdlpgjgbcjfgaohdahloollfgoc/options.html)   
>[Motrix-extension](https://github.com/gautamkrishnar/motrix-webextension/releases)    

 
>[Typora](https://typora.io/)   
>[Obsidian](https://obsidian.md/)   Obsidian Nord主题  
>[Notion](https://www.notion.so/)   
>[picgo](https://github.com/Molunerfinn/PicGo/releases)

>[Capslock+](https://capslox.com/capslock-plus/)

>[IObit Uninstaller Portable](https://clearstack.lanzous.com/ihy6Eoricri)

>[BookxNote pro](http://www.bookxnote.com/)

>[vscode](https://code.visualstudio.com/)      
>`Nord Operator Theme`  
>c:/user/.vscode下更改`theme.json文件`fontstyle为“”，去除斜体样式     
>[sublimeText3](https://www.sublimetext.com/3)   
>idea   

  
>[git](https://git-scm.com/)

>Vmware workstation `U盘安装`

>Termius `U盘安装`




## 开发环境
>go

>java jdk1.8

>c/c++  MinGW/Clang

>python3   
>you-get 
```bash
you-get -x 127.0.0.1:7890 " "
```
>scoop 包管理工具   
```bash
scoop config proxy 127.0.0.1:7890
```
>annie

```bash
scoop install annie

```

>git

>MySQL

>Redis

>node npm

>[hugo](https://github.com/gohugoio/hugo/releases)

>[Fira Code字体](https://github.com/tonsky/FiraCode/releases)


>字体配置
```json
 "editor.fontFamily": "Fira Code",

 "editor.fontLigatures": false, //不开启连字

 "editor.fontSize": 15,

 "editor.fontWeight": "normal",
```

>[Noto Serif字体](https://www.google.com/get/noto/#serif-lgc)

>[Mactype](https://mactype.net/)   


>cmd设置代理

```bash
set http_proxy=http://127.0.0.1:7890
set https_proxy=http://127.0.0.1:7890
```

## 网络问题
谷歌浏览器无法上网时的3种解决方案
>1.更改dns为阿里的223.5.5.5   

>2.cmd执行以下命令   
```bash
netsh winsock reset  :: 重置winsock目录
netsh int ip reset  :: 重置tcp/ip各组件到初始状态
ipconfig /release  :: DHCP客户端手工释放动态IP地址
ipconfig /renew  :: DHCP客户端手工向服务器刷新请求
ipconfig /flushdns  :: 清除本地DNS缓存
```

>3.设置-->网络和Internet-->状态-->网络重置

谷歌浏览器重定向到www.google.com.hk 的解决方案   
> 安装浏览器扩展`URLRedirector`   

>添加用户规则
![](https://cdn.jsdelivr.net/gh/clearyup/picgo/img/20210501214358.png)

>或者在广告拦截插件`uBlock Origin`中加入   
>||google.com.hk$script
![](https://cdn.jsdelivr.net/gh/clearyup/picgo/img/20210501220148.png)

