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
## 计算图形直方图 ##
**图像直方图是用以表示数字图像中亮度分布的直方图，标绘了图像中每个亮度值的像素数，可以借助观察该直方图了解需要如何调整亮度分布。**<br>
**计算机通常将图像表示为RGB值，或者再加上alpha值（通透度，透明度），称为RGBA值。在Pillow中，RGBA的值表示为由4个整数组成的元组，分别是R、G、B、A。整数的范围0~255。RGB全0就可以表示黑色，全255代表黑色。(255, 0, 0, 255)代表红色**

### 分离R G B（red green blue）三通道 ###

	import cv2  
	import numpy as np 
	
	
	
	'''计算每个通道直方图'''
	def calcAndDrawHist(image, color):
	    '''hist = cv2.calcHist([image],  
	     img必须用[]括起来
	     [0], #使用的通道  
	     None, #没有使用mask  
	    [256], #HistSize  表示这个直方图分成多少份（即多少个直方柱）
	    [0.0,255.0]) #直方图柱的范围      '''
	    hist= cv2.calcHist([image], [0], None, [256], [0.0,255.0])    
	    '''minMaxLoc寻找矩阵(一维数组当作向量,用Mat定义) 中最小值和最大值的位置'''
	    minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(hist)    #结果为 (0.0, 17288.0, (0, 0), (0, 1))
	    '''np.zeros();返回来一个给定形状和类型的用0填充的数组； eg: np.zeros(5)->array([ 0.,  0.,  0.,  0.,  0.])
	    zeros(shape, dtype=float, order='C') shape:形状  dtype:数据类型，可选参数，默认numpy.float64   order:可选参数，c代表与c语言类似，行优先；F代表列优先'''
	    histImg = np.zeros([256,256,3], np.uint8)  #300 个 300行 3列
	    hpt = int(0.9* 256)
	       
	    
	    for h in range(300):  
	        '''计算直方图的每个点的值 '''
	        intensity = int(hist[h]*hpt/maxVal)
	        '''绘制直线'''
	        cv2.line(histImg,(h,256), (h,256-intensity), color) 
	        '''void line(Mat& img, Point pt1, Point pt2, const Scalar& color, int thickness=1, int lineType=8, int shift=0)
	        img: 要绘制线段的图像。
			pt1: 线段的起点。
			pt2: 线段的终点。
			color: 线段的颜色，通过一个Scalar对象定义。
			hickness: 线条的宽度。
			lineType: 线段的类型。可以取值8， 4， 和CV_AA， 分别代表8邻接连接线，4邻接连接线和反锯齿连接线。默认值为8邻接。为了获得更好地效果可以选用CV_AA(采用了高斯滤波)。
			shift: 坐标点小数点位数。  '''
	    return histImg; 
	
	if __name__ == '__main__':    
	    img = cv2.imread("D:/pip/xiangjiao/1517974165.9889214.jpg")    
	    b, g, r = cv2.split(img)    
	    
	    B = calcAndDrawHist(b, [255, 0, 0])    
	    G = calcAndDrawHist(g, [0, 255, 0])    
	    R = calcAndDrawHist(r, [0, 0, 255])    
	        
	    cv2.imshow("histImgB", B)    
	    cv2.imshow("histImgG", G)    
	    cv2.imshow("histImgR", R)    
	    cv2.imshow("Img", img)    
	    '''waitKey()函数的功能是不断刷新图像，频率时间为delay，单位为ms。'''
	    cv2.waitKey(0)    
	    '''关闭所有的HighGUI窗口'''
	    cv2.destroyAllWindows()   


![](https://i.imgur.com/vrxQf8p.png)
![](https://i.imgur.com/NJB4SLM.png)
![](https://i.imgur.com/nBGKoSy.png)
## 合并RGB通道 ##

	import cv2 as cv2
	import numpy as np    
	         
	img = cv2.imread('D:/pip/xiangjiao/1517974165.9889214.jpg')    
	 '''作为折线输入图  全0图像  '''  
	h = np.zeros((300,300,3),np.uint8) 
	'''直方图中各bin的顶点位置   (生成一个256行1列的数组) '''        
	bins = np.arange(256).reshape(256,1) 
	'''BGR三种颜色'''    
	color = [ (255,0,0),(0,255,0),(0,0,255) ] 
	
	'''对三个通道都遍历一遍（枚举3个通道）'''
	for ch, col in enumerate(color):    
	    '''计算ch通道直方图'''
	    originHist = cv2.calcHist([img],[ch],None,[256],[0,256]) 
	    '''Opencv2的归一化函数normalize()，使得直方图的范围限定在0-255×0.9之间
	    void normalize(InputArray src,OutputArray dst, double alpha=1, doublebeta=0, int norm_type=NORM_L2, int dtype=-1, InputArray mask=noArray() )
	    src  输入数组   dst 输出数组，支持原地运算
	    归一类型：
	    NORM_MINMAX:数组的数值被平移或缩放到一个指定的范围，线性归一化，一般较常用。
	    NORM_INF: 此类型的定义没有查到，根据Opencv2 1的对应项，可能是归一化数组的C-范数(绝对值的最大值)
	    NORM_L1 :  归一化数组的L1-范数(绝对值的和)
	    NORM_L2: 归一化数组的(欧几里德)L2-范数
	    '''
	    cv2.normalize(originHist, originHist,0,255*0.9,cv2.NORM_MINMAX)
	    '''使用around对矩阵中每一个元素取整（四舍五入）
	    因为calcHist函数返回的是float32类型的数组所以将整数部分转成np.int32类型。 例如66.666->66.0->66
	    注意：Python的int(...)只能转换一个元素。这里是很多个元素，numpy的转换函数可以对数组中的每个元素都进行转换
	    '''
	    hist=np.int32(np.around(originHist)) 
	    '''
	    column_stack()取一组一维数组并将它们堆叠成一列以构成一个二维数组。
	    将binds与hist转成相应的坐标，然后一一对应
	    '''
	    pts = np.column_stack((bins,hist))
	    '''绘制出折线 
        第二个参数包含多角形各個曲线点的阵列。即pts
	    第三个False参数指出不是封闭图形
	    第四个参数指定了折线的颜色。 3个通道3个颜色
	    cv2.polylines(h,[pts],False,col)
	    '''
	    
	'''反转绘制好的直方图，因为绘制时，[0,0]在会图像的左上角。'''
	h = np.flipud(h)
	cv2.imshow('img',h)  
	cv2.waitKey(0)


![](https://i.imgur.com/0Wid7BI.png)  