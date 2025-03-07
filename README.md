### 这是一个基于Murainbot2qq机器人框架的可视化界面和一些插件的开发

MBR2:https://github.com/MuRainBot/MuRainBot2

MRB2本身**不具备任何**实际功能
具体的功能（如通过关键词回复特定消息等）都需要插件来实现，有任何功能的需要可以自行阅读MRB2的文档编写。

本项目在原有基础上打算做得更适合不懂编程之类的代码的小白使用插件或者调试等

### 目录结构（暂时）


<details>
<summary>查看基本看目录结构</summary>

```
├─ data                MRB2及插件的临时/缓存文件
│   ├─ ...
├─ Lib                 MRB2的Lib库，插件和MRB2均需要依赖此Lib
│   ├─ __init__.py     MRB2Lib的初始化文件
│   ├─ core            核心模块，负责配置文件读取、与实现端通信、插件加载等
│   |   ├─ ...
│   ├─ utils           工具模块，实现一些偏工具类的功能，例如QQ信息缓存、日志记录、事件分类等
│   |   ├─ ...
│   ...
├─ logs
│   ├─ latest.log      当日的日志
│   ├─ xxxx-xx-xx.log  以往的日志
│   ...
├─ plugins
│   ├─ xxx.py           xxx插件代码
│   ├─ yyy.py           yyy插件代码 
│   ...
├─ plugin_configs
│   ├─ xxx.yml          xxx插件的配置文件
│   ├─ yyy.yml          yyy插件的配置文件
│   ...
├─ config.yml           MRB2配置文件
├─ main.py              MRB2的入口文件
└─ README.md            这个文件就不用解释了吧（？）
```

</details>


