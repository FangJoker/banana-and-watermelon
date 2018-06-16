import cv2  
import numpy as np 
import matplotlib.pyplot as plt  

#计算每个通道直方图  import matplotlib.pyplot as plt
#图像直方图是用以表示数字图像中亮度分布的直方图，标绘了图像中每个亮度值的像素数，可以借助观察该直方图了解需要如何调整亮度分布。
def calcAndDrawHist(image, color,rgbList):
    #hist = cv2.calcHist([image],  
    # img必须用[]括起来
    # [0], #使用的通道  
    # None, #没有使用mask  
    #[256], #HistSize  表示这个直方图分成多少份（即多少个直方柱）9
    #[0.0,255.0]) #直方图柱的范围      
    hist= cv2.calcHist([image], [0], None, [256], [0.0,256.0])    
    #minMaxLoc寻找矩阵(一维数组当作向量,用Mat定义) 中最小值和最大值的位置
    minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(hist)    #结果为 (0.0, 17288.0, (0, 0), (0, 1))
    hpt = int(0.9* 256)
  
    
    for h in range(256):  
        #计算直方图的每个点的值 
        intensity = int(hist[h]*hpt/maxVal)
        rgbList [h] = intensity
     
    return rgbList



if __name__ == '__main__':    
    img = cv2.imread("D:/pip/xiangjiao/1517974165.9889214.jpg")    
    b, g, r = cv2.split(img)    
    
    rList = [None]*256
    glist = [None]*256
    blist = [None]*256

    glist2 = [None]*256
    
    #x, y = [], []

    #初始化x
    _x = [None]*256
    for i in range(256):
        _x[i]=i

    b = calcAndDrawHist(b, [255, 0, 0],rList)
    g = calcAndDrawHist(g, [0, 255, 0],glist)  
    r = calcAndDrawHist(r, [0, 0, 255],rList) 


   #曲线拟合，8阶.  结果返回m次拟合多项式系数, 从高次到低次存放在向量p1中.
   
    p1 = np.polyfit(_x, g,29)
    fig = plt.figure()
    #描绘散点
    plt.scatter(_x, g, c="r")
    #初始化x数轴
    x = np.linspace(0,256, 256)
    #np.polyval(p1, x1)求得x1处的值
    plt.plot(x, np.polyval(p1, x), c='b')
    plt.show()

   
  

   
  
    


    
   
   