from model import *
import cPickle
import numpy as np
import tensorflow as tf
import argparse
from scipy.misc import imread,imsave

parser=argparse.ArgumentParser()

parser.add_argument('--channel',dest='channel',type=int,required=True)
parser.add_argument('--input',dest='input',type=str,required=True)
parser.add_argument('--batch',dest='batch',type=int,default=32)
parser.add_argument('--percent',dest='percent',type=float,required=True)
parser.add_argument('--epoch',dest='epoch',type=int,default=50)
parser.add_argument('--layer',dest='layer',type=int,default=10)

args=parser.parse_args()


data=[]

if args.channel==3:
    with open('../data/cifar-10-batches-py/data_batch_1','rb') as fo:
        dict=cPickle.load(fo)
    data1=dict['data']
    data1=data1.reshape(-1,3,32,32)

    data=np.zeros([3000,32,32,3])
    for num in range(3000):
        for i in range(32):
            for j in range(32):
                for c in range(3):
                    data[num][i][j][c]=data1[num][c][i][j]

elif args.channel==1:
    data=[]
    for i in range(1,401):
        img=imread("../data/train/train_{:0>3d}.png".format(i))
        data.append(img.reshape([img.shape[0],img.shape[1],1]))
    data=np.array(data)


lr=0.001*np.ones([args.epoch])
lr[30:]=lr[0]/10.0

tf.device('/gpu:0')
with tf.Session() as sess:
    model=denoiser(sess,args.percent,args.channel,args.layer,args.batch)
    model.train(data,lr,args.epoch)
    img=np.array(imread("../data/"+args.input+".png"))
    img=img.reshape([1,img.shape[0],img.shape[1],args.channel])
    denoised=model.denoise(img)
    if args.channel==1:
        denoised=denoised.reshape([denoised.shape[1],denoised.shape[2]])
    elif args.channel==3:
        denoised=denoised.reshape([denoised.shape[1],denoised.shape[2],3])
    imsave("../result/3150104669_"+args.input+".png",denoised)
