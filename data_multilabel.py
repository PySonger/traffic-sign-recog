'''
@Author: Ding Song
@Date: 2019-10-21 12:14:05
@LastEditors: Ding Song
@LastEditTime: 2019-10-21 18:55:47
@Description: 
'''
import os
import cv2
import caffe
import random
import numpy as np 
import sys

def CalMean(array):
    return np.mean(array,axis=0)

class Transformer(object):

    def __init__(self):
        pass

    def preprocess(self,img,degree,crop_size,crop_rate=0.5,flip_rate=0.5):
        #img = np.float32(img)
        #mean = CalMean(img)
        #img -= mean
        #if crop_rate > random.random():
        img = self.random_crop(img,crop_size)
        #if flip_rate > random.random():
        #    img = self.flip_img(img)
        #img = self.color_augumentation(img)
        img = cv2.resize(img,crop_size)
        
        return img

    def rot_img(self,img,degree):
        rows,cols,_ = img.shape
        center = (cols / 2, rows / 2)
        deg = random.uniform(-degree,degree)
        M = cv2.getRotationMatrix2D(center,deg,1)
        top_right = np.array((cols - 1,0)) - np.array(center)
        bottom_right = np.array((cols - 1,rows - 1)) - np.array(center)
        top_right_after_rot = M[0:2,0:2].dot(top_right)
        bottom_right_after_rot = M[0:2,0:2].dot(bottom_right)
        new_width = max(int(abs(bottom_right_after_rot[0] * 2) + 0.5),int(abs(top_right_after_rot[0] * 2) + 0.5))
        new_height = max(int(abs(top_right_after_rot[1] * 2) + 0.5),int(abs(bottom_right_after_rot[1] * 2) + 0.5))
        offset_x = (new_width - cols) / 2
        offset_y = (new_height - rows) / 2
        M[0,2] += offset_x
        M[1,2] += offset_y
        dst = cv2.warpAffine(img,M,(new_width,new_height))
        return dst

    def center_crop(self,img,crop_size):
        h,w,c = img.shape
        if crop_size[0] > w or crop_size[1] > h:
            #raise Exception('crop_size is too large')
            img = cv2.resize(img,crop_size)
            return img
        else:
            center_y,center_x = h/2,w/2
            return img[center_y-crop_size[1]/2:center_y+crop_size[1]/2,
                    center_x-crop_size[0]/2:center_y+crop_size[0]/2]

    def flip_img(self,img,flip_type=-1): 
        return cv2.flip(img,flip_type)

    def random_crop(self,img,crop_size):
        h,w,c = img.shape
        x,y = random.randint(0,w-crop_size[0]),random.randint(0,h-crop_size[1])
        return img[y:y+crop_size[1],x:x+crop_size[0]]

    def color_augumentation(self,img):
        hsv_img = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
        s = hsv_img[...,1]
        s_var = np.var(s)
        new_s = s if s_var > 500 else s*2 
        hsv_img[...,1] = new_s
        img = cv2.cvtColor(hsv_img,cv2.COLOR_HSV2BGR)
        return img

class TrafficLightDataLayer(caffe.Layer):

    def setup(self,bottom,top):
        self.top_names = ['data','label']   

        params = eval(self.param_str)
        self.batch_size = params['batch_size']
        self.batch_loader = BatchLoader(params,None)
        self.crop_size = params['crop_size']
        self.num_labels = params['num_labels']                #the number of labels

        top[0].reshape(
            self.batch_size,3,params['crop_size'][1],params['crop_size'][0]
        )
        top[1].reshape(self.batch_size,self.num_labels)

    def forward(self,bottom,top):

        for idx in range(self.batch_size):
            img,label = self.batch_loader.load_next_image()
            if img is None:
                continue
            top[0].data[idx,...] = img
            top[1].data[idx,...] = label

    def reshape(self,bottom,top):
        pass

    def backward(self,top,propagate_down,bottom):
        pass

class BatchLoader(object):

    def __init__(self,params,result):
        self.result = result
        self.batch_size = params['batch_size']
        self.img_size = params['img_size']
        self.crop_size = params['crop_size']
        self.degree = params['degree']

        self.num_labels = params['num_labels']         #the total number of all kinds of labels
        
        data_file = params['data_file']
        with open(data_file,'r') as f:
            self.data_lines = f.readlines()
            random.shuffle(self.data_lines)
        self._cur = 0

        self.transformer = Transformer()

    def load_next_image(self):
        if self._cur == len(self.data_lines):
            self._cur = 0
            random.shuffle(self.data_lines)

        img_path,labels = self.data_lines[self._cur].strip().split('#')
        name = os.path.split(img_path)[-1]
        img = cv2.imread(img_path)
        if img is None:
            print '{} has problem'.format(img_path)
            self._cur += 1
            return None,None
        img = cv2.resize(img,self.img_size)
        img = self.transformer.preprocess(img,degree=self.degree,crop_size=self.crop_size)
        labels = map(int,labels.split(' '))
        label_array = np.array(labels)
        img = np.transpose(img,(2,0,1))
        self._cur += 1
        return img,label_array