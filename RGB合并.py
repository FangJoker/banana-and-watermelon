import cv2 as cv2
import numpy as np    
         
img = cv2.imread('D:/pip/xiangjiao/1517974165.9889214.jpg')    
h = np.zeros((300,300,3),np.uint8)  #作为折线输入图  全0图像    
         
bins = np.arange(300).reshape(300,1) #直方图中各bin的顶点位置   (生成一个256行1列的数组) 
color = [ (255,0,0),(0,255,0),(0,0,255) ] #BGR三种颜色    
#对三个通道都遍历一遍（枚举3个通道）

for ch, col in enumerate(color):    
    #计算ch通道直方图
    originHist = cv2.calcHist([img],[ch],None,[256],[0,256]) 
    #Opencv2的归一化函数normalize()，使得直方图的范围限定在0-255×0.9之间
    #void normalize(InputArray src,OutputArray dst, double alpha=1, doublebeta=0, int norm_type=NORM_L2, int dtype=-1, InputArray mask=noArray() )
    #src  输入数组   dst 输出数组，支持原地运算
    #归一类型：
    #NORM_MINMAX:数组的数值被平移或缩放到一个指定的范围，线性归一化，一般较常用。
    #NORM_INF: 此类型的定义没有查到，根据Opencv2 1的对应项，可能是归一化数组的C-范数(绝对值的最大值)
    #NORM_L1 :  归一化数组的L1-范数(绝对值的和)
    #NORM_L2: 归一化数组的(欧几里德)L2-范数
    cv2.normalize(originHist, originHist,0,255*0.9,cv2.NORM_MINMAX)
    #使用around对矩阵中每一个元素取整（四舍五入）
    #因为calcHist函数返回的是float32类型的数组所以将整数部分转成np.int32类型。 例如66.666->66.0->66
    #注意：Python的int(...)只能转换一个元素。这里是很多个元素，numpy的转换函数可以对数组中的每个元素都进行转换
    hist=np.int32(np.around(originHist)) 
    #column_stack()取一组一维数组并将它们堆叠成一列以构成一个二维数组。
    #将binds与hist转成相应的坐标，然后一一对应
    pts = np.column_stack((bins,hist))
    #绘制出折线 
    #第三个False参数指出不是封闭图形
    #第四个参数指定了折线的颜色。 3个通道3个颜色
    cv2.polylines(h,[pts],False,col)
    
#反转绘制好的直方图，因为绘制时，[0,0]在会图像的左上角。
h = np.flipud(h)
cv2.imshow('img',h)  
cv2.waitKey(0)  