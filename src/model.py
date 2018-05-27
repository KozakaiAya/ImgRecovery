import tensorflow as tf
import numpy as np
import time
from corrupt import *

def interface(input,output_channels,layer,is_training=True):
    with tf.variable_scope('block1'):
        output=tf.layers.conv2d(input,64,3,padding='same',activation=tf.nn.relu)
        #  'same' : keep output be the same size with input after conv by padding 0 to input
    for layers in range(2,layer+2):
        with tf.variable_scope('block%d' % layers):
            output=tf.layers.conv2d(output,64,3,padding='same',name='conv%d' % layers,use_bias=False)
            output=tf.nn.relu(tf.layers.batch_normalization(output,training=is_training))
    with tf.variable_scope('block'+str(layer+2)):
        output=tf.layers.conv2d(output,output_channels,3,padding='same')
    return output

class denoiser(object):
    def __init__(self,sess,percent,input_c_dim,layer,batch_size):
        self.sess=sess
        self.percent=percent
        self.input_c_dim=input_c_dim  # channel dimension
        self.layer=layer
        self.batch_size=batch_size
        # build model
        self.origin=tf.placeholder(tf.float32,[None,None,None,self.input_c_dim])   # origin
        self.is_training=tf.placeholder(tf.bool,name='is_training')

        self.X=tf.placeholder(tf.float32,[None,None,None,self.input_c_dim])   # corrupted
        self.Y=interface(self.X,output_channels=self.input_c_dim,layer=self.layer,is_training=self.is_training)
        self.loss=tf.Variable(0.0,tf.float32)
        for batch in range(batch_size):
            for channel in range(input_c_dim):
                self.loss=self.loss+tf.norm(tf.reshape(self.Y[batch,:,:,channel],[-1])-tf.reshape(self.origin[batch,:,:,channel],[-1]),2)
        self.loss=self.loss/self.batch_size
        self.lr=tf.placeholder(tf.float32,name='learning_rate')

        optimizer=tf.train.AdamOptimizer(self.lr,name='AdamOptimizer')
        update_ops=tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.train_op=optimizer.minimize(self.loss)
        init=tf.global_variables_initializer()
        self.sess.run(init)

        print("[*] Initialize model successfully...")

    def train(self,data,lr,epoch):
        numBatch=int(data.shape[0]/self.batch_size)
        iter_num=0
        start_epoch=0
        start_step=0

        print("[*] Start training, with start epoch %d start iter %d : " % (start_epoch, iter_num))
        start_time = time.time()
        for epoch_ in range(start_epoch,epoch):
            np.random.shuffle(data)
            for batch_id in range(start_step,numBatch):
                batch_images=data[batch_id*self.batch_size:(batch_id+1)*self.batch_size,:,:,:]
                corrupt_images=tf_corrupt_img(batch_images,self.percent)
                _,loss= self.sess.run([self.train_op, self.loss],
                                                 feed_dict={self.origin:batch_images,
                                                            self.X:corrupt_images,
                                                            self.lr:lr[epoch_],
                                                            self.is_training:True})
                print("Epoch: [%2d] [%4d/%4d] time: %4.4f, loss: %.6f"
                      % (epoch_+1,batch_id+1,numBatch,time.time()-start_time,loss))
                iter_num+=1

        print("[*] Finish training.")

    def denoise(self,data):
        output_clean_image=self.sess.run(self.Y,feed_dict={self.X:data,self.is_training:False})
        return output_clean_image
