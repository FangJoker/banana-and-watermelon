from PIL import Image
import cv2  
import numpy as np 



#计算每个通道直方图
#图像直方图是用以表示数字图像中亮度分布的直方图，标绘了图像中每个亮度值的像素数，可以借助观察该直方图了解需要如何调整亮度分布。
def calcAndDrawHist(image, color):
    #hist = cv2.calcHist([image],  
    # img必须用[]括起来
    # [0], #使用的通道  
    # None, #没有使用mask  
    #[256], #HistSize  表示这个直方图分成多少份（即多少个直方柱）
    #[0.0,255.0]) #直方图柱的范围      
    hist= cv2.calcHist([image], [0], None, [256], [0.0,256.0])    
    #minMaxLoc寻找矩阵(一维数组当作向量,用Mat定义) 中最小值和最大值的位置
    minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(hist)    #结果为 (0.0, 17288.0, (0, 0), (0, 1))
    #np.zeros();返回来一个给定形状和类型的用0填充的数组； eg: np.zeros(5)->array([ 0.,  0.,  0.,  0.,  0.])
    #zeros(shape, dtype=float, order='C') shape:形状  dtype:数据类型，可选参数，默认numpy.float64   order:可选参数，c代表与c语言类似，行优先；F代表列优先
    histImg = np.zeros([256,256,3], np.uint8)  #300 个 300行 3列
    hpt = int(0.9* 256)
       
    
    for h in range(256):  
        #计算直方图的每个点的值 
        intensity = int(hist[h]*hpt/maxVal)
        #绘制直线  
        cv2.line(histImg,(h,256), (h,256-intensity), color) 
        #void line(Mat& img, Point pt1, Point pt2, const Scalar& color, int thickness=1, int lineType=8, int shift=0)
        #img: 要绘制线段的图像。
		#pt1: 线段的起点。
		#pt2: 线段的终点。
		#color: 线段的颜色，通过一个Scalar对象定义。
		#hickness: 线条的宽度。
		#lineType: 线段的类型。可以取值8， 4， 和CV_AA， 分别代表8邻接连接线，4邻接连接线和反锯齿连接线。默认值为8邻接。为了获得更好地效果可以选用CV_AA(采用了高斯滤波)。
		#shift: 坐标点小数点位数。   
    return histImg; 

if __name__ == '__main__':    
    img = cv2.imread("D:/pip/xiangjiao/1517974165.9889214.jpg")    
    b, g, r = cv2.split(img)    
    
    histImgB = calcAndDrawHist(b, [255, 0, 0])    
    histImgG = calcAndDrawHist(g, [0, 255, 0])    
    histImgR = calcAndDrawHist(r, [0, 0, 255])    
        
    cv2.imshow("histImgB", histImgB)    
    cv2.imshow("histImgG", histImgG)    
    cv2.imshow("histImgR", histImgR)    
    cv2.imshow("Img", img)    
    #waitKey()函数的功能是不断刷新图像，频率时间为delay，单位为ms。
    cv2.waitKey(0)    
    #关闭所有的HighGUI窗口
    cv2.destroyAllWindows()   