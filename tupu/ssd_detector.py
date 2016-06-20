import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../python'))
import cv2
import caffe
import numpy
from utils.timer import Timer

def gpu_init(id):
    caffe.set_mode_gpu()
    caffe.set_device(id)

class SSDDetector(object):
    def __init__(self, caffemodel, prototxt, classes):
        self.caffemodel = caffemodel
        self.prototxt = prototxt
        self.mean = np.array([104, 117, 123], np.uint8)
        self.net = caffe.Detector(prototxt, caffemodel,mean=self.mean)

    def detect(self, im_data):
    	h = im.shape[0]
    	w = im.shape[1]
    	r_im = cv2.resize(im, (300, 300))
    	blob = np.zeros((1, 3, 300,300), np.float32)
    
    	in_ = net.inputs[0]
    	pre_time = time.time()
    	blob[0] = net.transformer.preprocess(in_, r_im)
    	forward_kwargs = {in_: blob}
    	blobs_out = net.forward(**forward_kwargs)
    	res = blobs_out[net.outputs[0]] 
        dets = {}
    	for i in range(res.shape[2]):
            box = res[0,0,i]
            label = str(int(box[1]))
            score = box[2]
            if score < 0.1:
                continue
            xmin = int(box[3]*w)
            ymin = int(box[4]*h)
            xmax = int(box[5]*w)
            ymax = int(box[6]*h)
            det = [xmin, ymin, xmax, ymax, score]
            if label not in dets:
                dets[label] = []
            dets[label].append(det)
        return dets
       
     def draw_detection_result(self, im_data, class_name, dets):
        font = cv2.FONT_HERSHEY_SIMPLEX
        for i in range(dets.shape[0]):
            bbox = dets[i, :4]
            score = dets[i, -1]
            cv2.rectangle(im_data, (int(bbox[0]), int(bbox[1])),(int(bbox[2]), int(bbox[3])),\
                                  (0, 255, 0), 2)
            cv2.putText(im_data, class_name, (int(bbox[0]), int(bbox[1])), font, 0.5, (0,0,255), 1)

    def detect_image(self, im_data):
        class_res = self.detect(im_data)
        for k, v in class_res.items():
            self.draw_detection_result(im_data, k, v)
        r, i = cv2.imencode('.jpg', im_data)
        return i.data
