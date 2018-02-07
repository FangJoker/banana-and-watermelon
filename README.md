# 机器学习初尝试 #
**尝试项目： 利用监督学习来辨别“香蕉”与“苹果”**
## 第一步，数据收集 ##
因为是识别图片，当然第一步是要编写简单爬虫一个简单的爬虫来扒出大量图片,基于<br>

- python 3.6
- BeautifulSoup

Beautiful Soup 是一个可以从HTML或XML文件中提取数据的Python库，简单来说，它能将HTML的标签文件解析成树形结构，然后方便地获取到指定标签的对应属性。能够非常优雅的编写爬虫，可以在cmd下使用pip 安装
    
	pip install BeautifulSoup

示例代码：<br>

	import time
	from urllib import request
	from bs4 import BeautifulSoup
	import re
	url = 'http://www.quanjing.com/category/118082.html'
	headers = {'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:57.0) Gecko/20100101 Firefox/57.0'}
	page = request.Request(url, headers=headers)
	page_info = request.urlopen(page).read()
	
	# 将获取到的内容转换成BeautifulSoup格式，并将html.parser作为解析器
	soup = BeautifulSoup(page_info, 'html.parser')  
	
	#Beautiful Soup和正则表达式结合，提取出所有图片的链接（img标签中，以.jpg结尾的链接,http开头） 
	links = soup.find_all('img',src=re.compile(r"^http|.jpg$>")) 
	# 设置保存的路径，否则会保存到程序当前路径
	local_path=r'D:\pip\xiangjiao'
	num = 1
	for link in links:
	    downLoda ="正在下载第"+str(num)+"张"+link.attrs['src']
	    print(downLoda)
	    num +=1
	     # 保存链接并命名，time防止命名冲突
	    request.urlretrieve(link.attrs['src'], local_path+r'\%s.jpg' % time.time())

![](https://i.imgur.com/7KEfRle.png)
这样 一堆banana的图就轻松存在了本地
![](https://i.imgur.com/6rhx77z.png)
**值得一提的是，python存文件如果所写路径不存在会报错，不会说新建一个文件夹**