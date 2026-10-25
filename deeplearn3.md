---
title: jupyter安装
date: 2026-09-26 19:34:53
tags:
---

介绍一个工具
Jupyter Notebook
Jupyter Notebook 是一个基于 Web 的交互式计算环境，常用于数据分析、机器学习、科学计算、教学和文档编写。它允许你在一个 .ipynb 文件中混合编写：

代码：Python、R、Julia 等

说明文字：Markdown、公式、图片、链接

运行结果：表格、图表、图像、输出文本

名字来源于 Julia + Python + R，但现在支持多种内核



Notebook 文档	后缀为.ipynb本质是JSON文件
单元格Cell	 分为 Code、Markdown、Raw
内核Kernel	实际执行代码的进程，默认是 Python
JupyterLab	更现代的下一代界面，功能类似 IDE
Notebook 7	新版 Notebook，基于 JupyterLab 组件

常用快捷键：
Shift + Enter：运行当前单元格并跳到下一个
Ctrl + Enter：运行当前单元格
A / B：在上方/下方插入单元格
M / Y：切换 Markdown / Code
Esc：进入命令模式
Enter：进入编辑模式

安装教程
一，
# 创建虚拟环境，可选
python -m venv .venv

# 激活虚拟环境
# Windows:
.venv\Scripts\activate
# macOS / Linux:
source .venv/bin/activate

# 升级 pip
python -m pip install --upgrade pip

# 安装 Jupyter Notebook
pip install notebook

# 启动
jupyter notebook

如果 jupyter 命令找不到，可以用：
python -m notebook

安装 JupyterLab：
pip install jupyterlab
jupyter lab


方式二：conda / Miniconda 安装
conda create -n jupyter python=3.11 -y
conda activate jupyter
conda install -c conda-forge notebook -y

jupyter notebook

安装 JupyterLab：
conda install -c conda-forge jupyterlab -y
jupyter lab

方式三：Anaconda
如果你安装了 Anaconda，通常已经自带 Jupyter Notebook 和 JupyterLab。直接运行：

jupyter notebook

或：
jupyter lab


启动与基本使用
启动后，终端会显示类似地址
http://localhost:8888/tree?token=...
浏览器通常会自动打开。如果没有，手动复制该地址访问。


新建 Notebook：
点击右上角 New
选择 Python 3
在单元格中输入代码
按 Shift + Enter 运行
保存文件，默认后缀为 .ipynb
关闭服务：在终端按 Ctrl + C，再按 y 确认。
指定端口或目录：
jupyter notebook --port 8889 --notebook-dir=/path/to/notebooks


在虚拟环境中使用内核
如果你在某个虚拟环境安装了 Jupyter，但新建 Notebook 时找不到该环境，可以注册内核：

# 先激活目标环境
conda activate myenv

# 安装 ipykernel
pip install ipykernel

# 注册内核
python -m ipykernel install --user --name=myenv --display-name "Python (myenv)"
之后在 Notebook 的 Kernel 菜单中即可选择该环境。


常见问题
1. jupyter: command not found
尝试：


python -m notebook
或检查 Python Scripts 目录是否加入 PATH。

2. 端口被占用
jupyter notebook --port 8889

3. 远程服务器访问
不推荐直接暴露公网。建议使用 SSH 隧道：
ssh -L 8888:localhost:8888 user@server
然后在服务器运行：
jupyter notebook --no-browser --port=8888
本地浏览器访问：
http://localhost:8888

4.Notebook 中导入包失败
通常是内核环境不对。确认 Notebook 右上角选择的内核，与你安装包的环境一致。

5.安装完成后，运行：
jupyter --version
jupyter notebook --version