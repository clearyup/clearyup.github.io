# 导数微分和积分

# 导数
## 什么是导数
>导数是函数值改变值比上自变量改变值,衡量的是变化率,是自变量的变化引起的函数的变化率,表示为   
>
>$\Large f(x)^{'}=\frac{df}{dx}=\frac{f(x+dx)-f(x)}{dx}$ ,并强调是$\Large dx\rightarrow 0$    
>  
>几何上表示经过图像上某一点切线上的斜率   

>某点的导数以及瞬时变化率其实都是两个点之间的变化,只是将这两个点之间的距离 $\large dx \rightarrow 0$ ,就将`一个点`的变化近似为`两个点`之间的变化,

![](https://cdn.jsdelivr.net/gh/clearyup/picgo/img/20210601111127.png)



# 微分
## 什么是微分
 $\large \varDelta f=f(x+\varDelta x)-f(x)$ 表示`微分`,衡量函数值变化量,实质是用 $\large df=f(x)dx$ 近似的表示`微分`

# 积分
## 什么是积分
>从几何上来看，要找到一个函数 $\large A(x)$ , 代表 $\large f(x)在[0,x]$ 和坐标轴围成的面积,这个函数 $\large A(x)$ 就是$\large f(x)$的 `积分`,将区间 $\large [0,x]$ 划分为一个个相等的间隔 $\large dx$,且 $\large dx \rightarrow 0$ 它们满足这个关系   
>
>$\Large A(x)^{'}=\frac{dA}{dx} = f(x)$   
>   
>这个 $\large A(x)$可以表示为   
>   
> $\Large \int_{0}^{x}f(x)dx$   
>    
> 可以理解为将  $\large [0,x]$  划分为的每个 $\large dx$ 近似的面积$\large f(x)dx$相加   
> 通过分割,近似,求和,取极限得到的

>如下图中取 $\large dx$  作为自变量 `x`的微小变化量, $\large dA$作为函数	`y`的微小变化量, 当 $dx$ 足够小, 就可以将这个增加的面积`近似`成为一个`长方形`, 可以得到 $\Large dA\approx x^2dx$ ,  将$dx$除过来就得到$\Large \frac{dA}{dx}\approx x^2$, 说明要求的积分$\large A(x)$ 求导后变成了原函数 $\large x^2$, 要求积分$\large A(x)$ 就是求 $\large x^2$ 的原函数   


![](https://cdn.jsdelivr.net/gh/clearyup/picgo/img/20210601100712.png)


