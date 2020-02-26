'''
@Author: Ding Song
@Date: 2020-02-27 00:36:59
@LastEditors: Ding Song
@LastEditTime: 2020-02-27 01:19:06
@Description: This file is used to test/evalute the performance of 
              multi-label classification caffemodel.
'''
import os
import cv2
import numpy as np 
import caffe

class TestMultiLabel(object):

    def __init__(self,deploy,caffemodel,save_dir):

        self.model = caffe.Net(deploy,caffemodel,caffe.TEST)
        self.transformer = caffe.io.Transformer({'data':self.model.blobs['data'].data.shape})
        self.transformer.set_transpose('data',(2,0,1))
        self.save_dir = save_dir

    def get_img_path(self,dirname):
        path_list = []
        for root,dirs,files in os.walk(dirname):
            for filename in files:
                path_list.append(os.path.join(root,filename))

        print "there are {}imgs totally.".format(len(path_list))
        return path_list

    def test(self,img_dir):
        img_path_list = self.get_img_path(img_dir)
        for idx, img_path in enumerate(img_path_list):
            img = cv2.imread(img_path)
            self.model.blobs['data'].data[...] = self.transformer.preprocess('data',img)
            output = self.model.forward()
            
    def draw_in_origin_img(self,txt_file,origin_img_path):
        """
        origin_img_path: the path of the original img.
        txt_file: name of this file is the same as the original img. it contains the bounding box of cars in the original img.
                  the format is:
                  class_id, x, y, w, h
                  class_id, x, y, w, h
                  ........
        """
        img = cv2.imread(origin_img_path)
        with open(txt_file,'r') as f:
            lines = f.readlines()
        for line in lines:
            _,x,y,w,h = line.strip().split(',')
            x,y,w,h = int(x),int(y),int(w),int(h)
            car_img = img[y-h/2:y+h/2,x-h/2:x+w/2]
            self.model.blob['data'].data.[...] = self.transformer.preprocess('data',img)
            output = self.model.forward()
