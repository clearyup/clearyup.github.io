# Algolia


# algolia配置

博客本身支持lunr和algolia两种搜索引擎,为了高性能和网站加载速度我选择algolia

* lunr无需同步**index.json**文件,但是lunr需要加载本地js文件,这会使得网站加载缓慢,并且性能低(需要检索中文时需要一个较大的分词依赖库)
* algolia,高性能占用带宽低,网站加载快,配置麻烦,更新网站需要同步**index.json**文件

## 配置步骤

### 第一步

你需要在algolia官网注册,并create一个新的app,然后在你的hugo config.toml文件中配置algolia

```toml
[params.search]
    enable = true
    # 搜索引擎的类型 ("lunr", "algolia")
    type = "algolia"
    # 文章内容最长索引长度
    contentLength = 4000
    # 搜索框的占位提示语
    placeholder = ""
    # LoveIt 新增 | 0.2.1 最大结果数目
    maxResultLength = 10
    # LoveIt 新增 | 0.2.3 结果内容片段长度
    snippetLength = 50
    # LoveIt 新增 | 0.2.1 搜索结果中高亮部分的 HTML 标签
    highlightTag = "em"
    # LoveIt 新增 | 0.2.4 是否在搜索索引中使用基于 baseURL 的绝对路径
    absoluteURL = false
    [params.search.algolia]
    #index是algolia注册的名字,不是index.json,是个坑. 
    #在algolia中 Duplication and Grouping Distinct设置为true，然后Attribute for Distinct设置为url，或者title
      index = ""
      appID = ""
      searchKey = ""
```



### 第二步

这时候你到博客搜索会发现algolia搜不到任何东西,这是因为algolia是根据hugo对你的网站生成的index.json进行搜索匹配文章的,你需要上传index.json文件,这会引发一个问题,你上传一次新文章或者修改一次文章就要重新到algolia控制台上传一次你的新生成的网站index.json文件,毫无疑问这是非常麻烦的,对此我们提供两种方法

* 使用algolia-automatic npm包,配置好后每次npm run algolia上传文件
* 自己根据官网写一个node.js脚本,运行node push_index.js命令行

我自己用的第二种,因为node push_index.js命令行可以加到我的文章git发布脚本和文章发布一起运行,而npm run命令会使得git部署发布脚本中断,因为它重开了一个bash,而且需要上传你的密钥到vercel,不安全.

push_index.js脚本 indexName是你创建的algolia app名字

```js
const algoliasearch = require('algoliasearch');

const appID = " "
const indexName = " " 
const adminKey = " "
const indexFile = "./public/index.json"


const client = algoliasearch(appID, adminKey);
const index = client.initIndex(indexName);
const indexJson = require(indexFile);

index.saveObjects(indexJson, {
  autoGenerateObjectIDIfNotExist: true
}).then(({ objectIDs }) => {
  console.log(objectIDs);
});
```



### 第三步

将 node push_index.js命令行加到deploy.sh脚本中,和git命令一起执行,完成git仓库push,网站页面更新,以及algolia的index.json更新

因此你在新文章发布后,algolia也有了最新的index.json脚本,新文章不会出现检索不到的情况

{{< admonition type=note title="注意!" open=ture >}}
你最好将push_index.js文件加入到.gitignore文件中,因为其中的密钥上传到仓库可能不安全!
{{< /admonition >}}







