'''
@Author: Ding Song
@Date: 2020-02-27 00:36:59
@LastEditors: Ding Song
@LastEditTime: 2020-02-28 15:30:27
@Description: This file is used to test/evalute the performance of 
              multi-label classification caffemodel.
'''
import os
import cv2
import numpy as np 
import caffe
from PIL import Image,ImageDraw,ImageFont

class AvgMea(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.total = 0
        self.right = 0

    def append(self,pred,label):
        self.total += 1
        self.right += 1 if pred == label else 0

    def cal(self):
        return self.right / self.total

class TestMultiLabel(object):

    def __init__(self,deploy,caffemodel,save_dir):

        self.model = caffe.Net(deploy,caffemodel,caffe.TEST)
        self.transformer = caffe.io.Transformer({'data':self.model.blobs['data'].data.shape})
        self.transformer.set_transpose('data',(2,0,1))
        self.save_dir = save_dir
        self.avgmea = AvgMea()
        self.makedirs(save_dir)

    def makedirs(self,dir_name):
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)

    def rcb2bgr(self,img):
        r,c,b = cv2.split(img)
        g = (c - r * 0.299 - b * 0.114 - 0.5) / 0.587
        g[g > 255] = 255
        g[g < 0] = 0
        g = g.astype(np.uint8)
        bgr = cv2.merge((b,g,r))
        return bgr

    def evalution(self,data_file):
        with open(data_file,'r') as f:
            lines = f.readlines()
        for idx,line in enumerate(lines):
            path,label = line.strip().split('#')
            label = [int(i) for i in label]
            img = cv2.imread(path)
            self.model.blobs['data'].data[...] = self.transformer.preprocess('data',img)
            output = self.model.forward()
            pred = output['prob'].squeeze()
            pred = list(np.argmax(pred,axis=1))
            self.avgmea.append(pred,label)
        accuracy = self.avgmea.cal()
        print("the accuracy is {}".format(accuracy))

    def draw_in_origin_img(self,txt_file,origin_img_path):
        """
        origin_img_path: the path of the original img.
        txt_file: name of this file is the same as the original img. it contains the bounding box of cars in the original img.
                  the format is:
                  class_id, x, y, w, h
                  class_id, x, y, w, h
                  ........
        """
        draw_dict = {
            (0,0):'No Turn Signal',
            (0,1):'Right Turn',
            (1,0):'Left Turn',
            (1,1):'Double Shining'
        }
        img = cv2.imread(origin_img_path)
        img = self.rcb2bgr(img)
        pil_img = Image.fromarray(img[...,::-1].astype('uint8')).convert('RGB')
        draw = ImageDraw.Draw(pil_img)
        with open(txt_file,'r') as f:
            lines = f.readlines()
        for line in lines:
            _,x,y,w,h = line.strip().split()
            x,y,w,h = int(float(x)),int(float(y)),int(float(w)),int(float(h))
            if _ not in ['0','1','2'] or h < 50:
                continue
            car_img = img[y-h//2:y+h//2,x-w//2:x+w//2]
            self.model.blobs['data'].data[...] = self.transformer.preprocess('data',car_img)
            output = self.model.forward()
            pred = output['prob'].squeeze()
            pred[pred > 0.5] = 1
            pred[pred <= 0.5] = 0
            pred = tuple(pred)
            word = draw_dict[pred]
            print(word)
            #cv2.rectangle(img,(x-w//2,y-h//2),(x+w//2,y+h//2),[0,255,0],1)
            #cv2.putText(img,word,(x-w//2,y-h//2),cv2.FONT_HERSHEY_COMPLEX,1,[0,0,255],1)
            draw.rectangle((x-w//2,y-h//2,x+w//2,y+h//2),outline='red')
            draw.text((x-w//2,y-h//2),word,(255,255,0))
        img_name = origin_img_path.split(os.sep)[-1]
        date = origin_img_path.split('/')[-2]
        save_dir = os.path.join(self.save_dir,date)
        self.makedirs(save_dir)
        save_path = os.path.join(save_dir,img_name)
        pil_img.save(save_path)

def main():
    deploy = 'resnet18_deploy.prototxt'
    caffemodel = 'models/resnet18_multilabel_112x112_0228_iter_3031.caffemodel'
    save_dir = 'test_results'
    img_dir = 'changtai1118/'
    anno_dir = 'Annotations/changtai1118'
    test = TestMultiLabel(deploy,caffemodel,save_dir)
    img_path_list = []
    for root,dirs,files in os.walk(img_dir):
        for filename in files:
            img_path = os.path.join(root,filename)
            img_path_list.append(img_path)
    for img_path in img_path_list:
        dir_list = img_path.split('/')
        txt_name = dir_list[-1].split('.')[0] + '.txt'
        txt_path = anno_dir + '/' + '/'.join(dir_list[1:-1]) + '/' + txt_name
        test.draw_in_origin_img(txt_path,img_path)
    

if __name__ == '__main__':
    """
    deploy = 'resnet18_deploy.prototxt'
    caffemodel = 'models/resnet18_multilabel_112x112_0228_iter_3031.caffemodel'
    save_dir = 'test_results'
    test = TestMultiLabel(deploy,caffemodel,save_dir)
    test.draw_in_origin_img('frame_vc1_11439_rcb.txt','frame_vc1_11439_rcb.jpg')
    """
    main()