
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










