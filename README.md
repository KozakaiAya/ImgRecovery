# Description

受损图像 ![](http://chart.googleapis.com/chart?cht=tx&chl=x)  是由原始图像 ![](http://chart.googleapis.com/chart?cht=tx&chl=I\in R^{H*W*C}) 添加了不同噪声遮罩(noise masks)![](http://chart.googleapis.com/chart?cht=tx&chl= M\in R^{H*W*C}) 得到的 ![](http://chart.googleapis.com/chart?cht=tx&chl=  X=I\odot   M )  ，其中 ![](http://chart.googleapis.com/chart?cht=tx&chl= \odot) 是逐元素相乘。

噪声遮罩仅包含{0,1}值。噪声遮罩的每行是用噪声比率产生的，即噪声遮罩每个通道每行有固定比率的像素值为0，其他为1。



# Dependency

PYTHON_VERSION="2.7"

TensorFlow

Numpy

Scipy

Pillow



# Arguments

Program must be run with three arguments :

channel(int) channel dimension of input image

input(str) input image name, it must be put in data folder and with png format

percent(float) noise percent



Other optional arguments :

epoch(int) epoch num for training, default is 50

layer(int) number of CNN layers in hidden layer, default is 10

batch(int) batch size, default is 32



Example :

If you want to recover a noised image(data/abc.png) with 60% percent noise rate, you can use following command in folder src :

python main.py --channel 3 -- input abc --percent 0.6



# Example

A gray image with 80% noise rate.

Fix it by using 8 CNN layers and 40 epoches.

![](https://github.com/CKCZZJ/ImgRecovery/blob/master/data/A.png)![](https://github.com/CKCZZJ/ImgRecovery/blob/master/result/3150104669_A.png)



A RGB image with 40% noise rate.

Fix it by using 10 CNN layers and 50 epoches.

![](https://github.com/CKCZZJ/ImgRecovery/blob/master/data/B.png) ![](https://github.com/CKCZZJ/ImgRecovery/blob/master/result/3150104669_B.png)



A RGB image with 60% noise rate.

Fix it by using 10 CNN layers and 50 epoches.

![](https://github.com/CKCZZJ/ImgRecovery/blob/master/data/C.png) ![](https://github.com/CKCZZJ/ImgRecovery/blob/master/result/3150104669_C.png)





# Author

CKCZZJ
