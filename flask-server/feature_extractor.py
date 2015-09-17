#!/usr/bin/env python
import os
import sys
import Image
import numpy as np

# caffe_root = '/opt/caffe/'
caffe_root = '/home/sean/Sean/DeepDev/caffe/'
sys.path.insert(0, caffe_root + 'python')
import caffe


class Feature:
    def __init__(self, model, caffemodel, dim, layer):
        self.model = model
        self.caffemodel = caffemodel
        self.dimension = dim #50
        self.layer = layer  #pool5

        self.net = caffe.Net(self.model, self.caffemodel, caffe.TEST)
        print [(k, v.data.shape) for k, v in self.net.blobs.items()]

    def get_feature(self, image_file):
        face = Image.open(image_file).resize((self.dimension, self.dimension), Image.BILINEAR).convert('L')
        face_arr = np.array(face, order="c") / 256.
        # n x c x h x w
        face_arr = face_arr.reshape(1, 1, face_arr.shape[0], face_arr.shape[1])
        # print face_arr.flags['C_CONTIGUOUS']
        label = np.array([1])  # Don't care

        self.net.set_input_arrays(face_arr.astype(np.float32), label.astype(np.float32))
        self.net.forward()

        feature = self.net.blobs[self.layer].data[0].flat
        return feature
