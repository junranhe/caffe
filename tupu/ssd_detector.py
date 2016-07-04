import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../python'))
import cv2
import caffe
import numpy as np

def gpu_init(id):
    caffe.set_mode_gpu()
    caffe.set_device(id)

class SSDDetector(object):
    def __init__(self, caffemodel, prototxt, classes):
        self.caffemodel = caffemodel
        self.prototxt = prototxt
        self.mean = np.array([104, 117, 123], np.uint8)
        self.net = caffe.Detector(prototxt, caffemodel,mean=self.mean)

    def detect(self, im):
    	h = im.shape[0]
    	w = im.shape[1]
    	r_im = cv2.resize(im, (300, 300))
    	blob = np.zeros((1, 3, 300,300), np.float32)
    	in_ = self.net.inputs[0]
    	blob[0] = self.net.transformer.preprocess(in_, r_im)
    	forward_kwargs = {in_: blob}
    	blobs_out = self.net.forward(**forward_kwargs)
    	res = blobs_out[self.net.outputs[0]]
        dets = {}
    	for i in range(res.shape[2]):
            box = res[0,0,i]
            label = str(int(box[1]))
            score = box[2]
            if score < 0.1:
                continue
            xmin = float(int(box[3]*w))
            ymin = float(int(box[4]*h))
            xmax = float(int(box[5]*w))
            ymax = float(int(box[6]*h))
            if xmin < 0 or xmin >= w or xmax < 0 or xmax >= w:
                continue
            if ymin < 0 or ymin >= h or ymax < 0 or ymax >= h:
                continue
            if xmin >= xmax or ymin >= ymax:
                continue
            det = [xmin, ymin, xmax, ymax, score]
            if label not in dets:
                dets[label] = []
            dets[label].append(det)
        np_dets = {}
        for k, v in dets.items():
            np_dets[k] = np.array(v, np.float32)
	#print 'np_dets:', np_dets
        return np_dets
    def draw_detection_result(self, im_data, class_name, dets):
        font = cv2.FONT_HERSHEY_SIMPLEX
        #print self.ocr_group(dets)
        for i in range(dets.shape[0]):
            bbox = dets[i, :4]
            score = dets[i, -1]
            cv2.rectangle(im_data, (int(bbox[0]), int(bbox[1])),(int(bbox[2]), int(bbox[3])),\
                                  (0, 255, 0), 2)
            cv2.putText(im_data, str(i), (int(bbox[0]), int(bbox[1])), font, 0.5, (0,0,255), 1)

    def ocr_group(self, dets):
	l = dets.shape[0]
        used = {}
        res = []
        while len(used) < l:
	    max_line = []
	    for i in range(l):
		if i in used:
		    continue
		line = [i]
                ymin = dets[i][1]
	        ymax = dets[i][3]
                r = (ymax - ymin)/2
		y_middle = (ymin + ymax) /2
		for j in range(l):
		    if i == j or j in used:
			continue
		    other_ymin = dets[j][1]
		    other_ymax = dets[j][3]
		    other_y_middle = (other_ymin + other_ymax)/2
		    other_r = (other_ymax - other_ymin)/2
		    distance = abs(y_middle - other_y_middle)
		    if distance < r/2 and distance < other_r/2:
			line.append(j)
		if len(line) > len(max_line):
		    max_line = line
		    def x_cmp(a, b):
			x1 = dets[a][0]
			x2 = dets[b][0]
			if x1 < x2:
			    return -1
			elif x1 > x2:
			    return 1
			return 0
		    max_line.sort(x_cmp)
		xmin = dets[i][0]
		xmax = dets[i][2]
		x_r = (xmax - xmin)/2
		x_middle = (xmin + xmax) / 2 
		line = [i]
		for j in range(l):
		    if i == j or j in used:
			continue
		    other_xmin = dets[j][0]
		    other_xmax = dets[j][2]
		    other_x_middle = (other_xmin + other_xmax)/2
		    other_x_r = (other_xmax - other_xmin)/2
		    distance = abs(x_middle - other_x_middle)
		    if distance < x_r/2 and distance < other_x_r/2:
			line.append(j)
		if len(line) > len(max_line):
		    max_line = line
		    def y_cmp(a, b):
			y1 = dets[a][1]
			y2 = dets[b][1]
			if y1 < y2:
			    return -1
			elif y1 > y2:
			    return 1
			return 0
		    max_line.sort(y_cmp)

	    assert len(max_line) > 0
	    for item in max_line:
		used[item] = True
	    res.append(max_line)
	def line_cmp(a,b):
	    assert len(a) > 0 and len(b) > 0
	    x_a = dets[a[0]][0]
	    y_a = dets[a[0]][1]
	    x_b = dets[b[0]][0]
	    y_b = dets[b[0]][1]
	    if y_a < y_b:
		return -1
	    elif y_a > y_b:
		return 1
	    elif x_a < x_b:
		return -1
	    elif x_a > x_b:
	   	return 1
	    return 0
	res.sort(line_cmp)
	return res

    def detect_image(self, im_data):
        class_res = self.detect(im_data)
        for k, v in class_res.items():
            self.draw_detection_result(im_data, k, v)
        r, i = cv2.imencode('.jpg', im_data)
        return i.data
