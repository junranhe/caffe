# -*- coding: utf-8 -*-
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../python'))
import cv2
import caffe
import numpy as np
import math
import ssd_util
import random
def gpu_init(id):
    caffe.set_mode_gpu()
    caffe.set_device(id)

class SSDDetector(object):
    def __init__(self, caffemodel, prototxt, classes):
        self.caffemodel = caffemodel
        self.prototxt = prototxt
        self.mean = np.array([104, 117, 123], np.uint8)
        self.net = caffe.Detector(prototxt, caffemodel,mean=self.mean)

    def internal_detect(self, im, has_angle = False):
        h = im.shape[0]
        w = im.shape[1]

        in_ = self.net.inputs[0]
        input_shape = self.net.blobs[in_].shape
        image_height = input_shape[2]
        image_width = input_shape[3]
        r_im = cv2.resize(im, (image_width, image_height))
        blob = np.zeros((1, 3, image_width, image_height), np.float32)
        blob[0] = self.net.transformer.preprocess(in_, r_im)
        forward_kwargs = {in_: blob}
        blobs_out = self.net.forward(**forward_kwargs)
        #print self.net.outputs[0]
        res = blobs_out[self.net.outputs[0]]
        dets = []
        clss = []
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
            if has_angle:
                if box.shape[0] == 8:
                    angle = float((box[7]))
                else:
                    angle = 0.0
                degree = ssd_util.angle2degree(angle, w, h)
                det = [xmin, ymin, xmax,ymax,score, degree]
            else:
                det = [xmin, ymin, xmax, ymax, score, 0.]
            dets.append(det)
            clss.append(label)
        return clss,dets

    def detect(self, im, has_angle = False):
        clss,dets = self.internal_detect(im, has_angle)
        dets_table = {}
        for i in range(len(clss)):
            k = clss[i]
            if k not in dets_table:
                dets_table[k] = []
            dets_table[k].append(dets[i])
        np_dets = {}
        for k, v in dets_table.items():
            np_dets[k] = np.array(v, np.float32)
	#print 'np_dets:', np_dets
        return np_dets

    def draw_detection_result(self, im_data, class_name, dets, color=None):
        if color is None:
            color = (0,255,0)
        font = cv2.FONT_HERSHEY_SIMPLEX
        #print self.ocr_group(dets)
        #print 'draw restult:'
        for i in range(dets.shape[0]):
            bbox = dets[i, :4]
            score = dets[i, 4]
            #print chr(int(class_name))
            cv2.rectangle(im_data, (int(bbox[0]), int(bbox[1])),(int(bbox[2]), int(bbox[3])),\
                                  color, 1)
            cv2.putText(im_data, str(class_name), (int(bbox[0]), int(bbox[1])), font, 0.5, (0,0,255), 1)


    def ocr_group(self, dets):
        import time
        pre_time = time.time()
        res = ssd_util.ocr_group(dets)
        #print res
        print 'ssd_ocr_group:', time.time() - pre_time
        return res
        dets = np.array(dets, np.float32)
        l = dets.shape[0]
        used = {}
        res = []
        #k_array = [math.tan(math.radians(degree)) for degree in [0,4,8,12,16,20,24,28,32,36,176,172,168,164,160, 156,152,148, 144]]
        def compute_distance(k,x0,y0, x1, y1):
            b = y0 - (k*x0)
            d = abs(k*x1 - y1 + b)/math.sqrt(k*k + 1)
            return d

        cache_point = {}
        def compute_center(x0,y0,x1,y1):
            x_c = (x0 + x1)/2
            y_c = (y0 + y1)/2
            x_d = (x1 - x0)
            y_d = (y1 - y0)
            r = math.sqrt(x_d*x_d + y_d*y_d)/2
            return x_c, y_c , r
        while len(used) < l:
            max_line = []
            for i in range(l):
                if i in used:
                    continue
                cur_degree = dets[i][5]
                k_array = [math.tan(math.radians(degree)) for degree in [cur_degree, cur_degree + 3, cur_degree-3]]
                for k in k_array:
                    line = [i]
                    for j in range(l):
                        if i == j or j in used:
                            continue
                        if i not in cache_point:
                            cache_point[i] = compute_center(dets[i][0], dets[i][1], dets[i][2], dets[i][3])
                        x0, y0, r0 = cache_point[i]
                        if j not in cache_point:
                            cache_point[j] = compute_center(dets[j][0], dets[j][1], dets[j][2], dets[j][3])
                        x1, y1, r1 = cache_point[j]
                        distance = compute_distance(k, x0, y0, x1, y1)
                        if distance < r0/4 and distance < r1/4:
                            line.append(j)
                    if len(line) > len(max_line):
                        max_line = line
                        def tm_cmp(a, b):
                            x1 = dets[a][0]
                            x2 = dets[b][0]
                            if x1 < x2:
                                return -1
                            elif x1 > x2:
                                return 1
                            return 0
                        max_line.sort(tm_cmp)
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
        print 'ocr group time:', time.time() - pre_time
        return res
    def nms(self, clss, dets, th):
        used = {}
        nms_index = []
        while len(used) < len(dets):
            max_score = 0
            max_index = -1
            for i in range(len(dets)):
                if i in used:
                    continue
                score = dets[i][4]
                if score > max_score:
                    max_score = score
                    max_index = i
            if max_index == -1:
                break
            nms_index.append(max_index)
            used[max_index] = True
            for i in range(len(dets)):
                if i in used:
                    continue
                def area(rec):
                    x1 = rec[0]
                    y1 = rec[1]
                    x2 = rec[2]
                    y2 = rec[3]
                    if x1 >= x2:
                        return 0
                    if y1 >= y2:
                        return 0
                    return (x2-x1)*(y2-y1)
                def interscet(a, b):
                    x1 = max(a[0], b[0])
                    y1 = max(a[1], b[1])
                    x2 = min(a[2], b[2])
                    y2 = min(a[3], b[3])
                    return [x1,y1,x2,y2,0]
                max_area = area(dets[max_index])
                other_area = area(dets[i])
                inter_det = interscet(dets[max_index], dets[i])
                inter_area = area(inter_det)
                if inter_area > th*max_area or inter_area > th*other_area:
                    used[i] = True
        new_clss = []
        new_dets = []
        for i in nms_index:
            new_clss.append(clss[i])
            new_dets.append(dets[i])
        return new_clss, new_dets

    def draw_rec(self, im_data, point, color, thickness = 1, degree = 0.0):
        xmin, ymin, xmax, ymax = point

        #point_list = [(xmin, ymin), (xmax,ymin), (xmax, ymax), (xmin, ymax)]
        #for i in range(len(point_list)):
        #    nxt_idx = (i + 1) % len(point_list)
        #    cv2.line(im_data, point_list[i], point_list[nxt_idx], color, thickness)
        point_list2 = ssd_util.compute_sub_rotate_rec(xmin,ymin, xmax, ymax, degree)
        point_list2 = [(int(x), int(y)) for x, y in point_list2]
        for i in range(len(point_list2)):
            nxt_idx = (i + 1) % len(point_list2)
            cv2.line(im_data, point_list2[i], point_list2[nxt_idx], color, thickness)
    def detect_image_ex(self, im_data, label_dict = None, has_angle = False):
        clss,dets = self.internal_detect(im_data, has_angle)
        #print 'dets:', dets
        #print 'old:', len(dets)
        #clss,dets = self.nms(clss, dets, 0.6)
        #prit 'nms:',len(dets)
        lines = ssd_util.ocr_group(dets)
        print lines
        font = cv2.FONT_HERSHEY_SIMPLEX
        for l in lines:
            #print u'text:',u','.join([label_dict[int(clss[a])] for a in l])
            left_point = dets[l[0]]
            right_point = dets[l[-1]]
            l_x = (left_point[0] + left_point[2])/2
            l_y = (left_point[1] + left_point[3])/2
            r_x = (right_point[0] + right_point[2])/2
            r_y = (right_point[1] + right_point[3])/2
            #degree = 0.
            #if r_x > l_x:
                #angle = math.atan((r_y - l_y)/ (r_x - l_x))
                #degree = (angle * 180.0)/math.pi

            #print 'degree:',degree
            import random
            color = (random.randint(0,255), random.randint(0,255), random.randint(0,255))
            for i in l:
                if label_dict is not None:
                    class_name = label_dict[int(clss[i])]
                else:
                    class_name = chr(int(clss[i]))
                bbox = dets[i][ :4]
                degree = dets[i][5]
                #print 'get degree'
                self.draw_rec(im_data,\
                     (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])), color, 2, degree)

                #degree = ssd_util.angle2degree()
                #cv2.rectangle(im_data, (int(bbox[0]), int(bbox[1])),(int(bbox[2]), int(bbox[3])),\
                #                  color, 2)
                #cv2.putText(im_data, str(angle), (int(bbox[0]), int(bbox[1])), font, 0.5, (0,0,255), 1)
        r, i = cv2.imencode('.jpg', im_data)
        return i.data

    def get_line_rotate_image_and_boxes(self,im_data,  l, dets):
        left_point = dets[l[0]]
        right_point = dets[l[-1]]
        l_x = (left_point[0] + left_point[2])/2
        l_y = (left_point[1] + left_point[3])/2
        r_x = (right_point[0] + right_point[2])/2
        r_y = (right_point[1] + right_point[3])/2
        degree = 0.
        if r_x > l_x:
            angle = math.atan((r_y - l_y)/ (r_x - l_x))
            degree = (angle * 180.0)/math.pi

        boxes = []
        rotate_boxes = []
        # degree not handle large than 45
        if degree >= 45.:
            for i in l:
                bbox = dets[i][:4]
                xmin = int(bbox[0])
                ymin = int(bbox[1])
                xmax = int(bbox[2])
                ymax = int(bbox[3])
                one_box = [(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)]
                boxes.append(one_box)
            return im_data, boxes, boxes, degree

        x_c = (l_x + r_x)/2
        y_c = (l_y + r_y)/2
        cols, rows = im_data.shape[:2]
        M = cv2.getRotationMatrix2D((x_c, y_c), degree,1)
        im_rotate = cv2.warpAffine(im_data,M,(rows,cols))

        for i in l:
            bbox = dets[i][:4]
            point_list = ssd_util.compute_sub_rotate_rec(float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]), degree)
            rotate_list = [ssd_util.rotate_point(x, y, x_c, y_c, -1*degree) for x, y in point_list]
            one_box = [(int(x), int(y)) for x, y in rotate_list]
            sub_rotate_box = [(int(x), int(y)) for x, y in point_list]
            boxes.append(one_box)
            rotate_boxes.append(sub_rotate_box)
        return im_rotate, boxes, rotate_boxes, degree

    def detect_image_rotate(self, im_data, label_dict = None, has_angle = False):
        clss,dets = self.internal_detect(im_data, has_angle)
        lines = ssd_util.ocr_group(dets)
        font = cv2.FONT_HERSHEY_SIMPLEX
        color = (random.randint(0,255), random.randint(0,255), random.randint(0,255))
        im_rotate, boxes, rotate_boxes, _= self.get_line_rotate_image_and_boxes(im_data, lines[0], dets)
        rotate_h = im_rotate.shape[0]
        rotate_w = im_rotate.shape[1]

        for index, box in enumerate(boxes):
            for k in range(len(box)):
                cv2.line(im_rotate, box[k], box[(k+1)%len(box)], color, 1)
            xmin, ymin = box[0]
            xmax, ymax = box[0]
            for x, y in box:
                xmin = min(x, xmin)
                ymin = min(y, ymin)
                xmax = max(x, xmax)
                ymax = max(y, ymax)
            h = ymax - ymin
            w = xmax - xmin
            diff = 0
            if h > w:
                diff = h - w
                xmax = xmax + int(diff/2)
                xmin = xmin - int(diff/2)
                print 'diff:', diff
            if xmax >= rotate_w:
                xmax = rotate_w -1
            if xmin < 0:
                xmin = 0
            if ymax >= rotate_h:
                ymax = rotwate_h -1
            if ymin < 0:
                ymin = 0
            crop_mat = im_rotate[ymin:ymax, xmin:xmax]

        r, i = cv2.imencode('.jpg', im_rotate)
        return i.data

    def detect_group(self, im_data, has_angle = False, is_crop = False):
        clss,dets = self.internal_detect(im_data, has_angle)
        clss,dets = self.nms(clss, dets, 0.9)
        lines = ssd_util.ocr_group(dets)
        font = cv2.FONT_HERSHEY_SIMPLEX
        new_clss = []
        new_dets = []
        new_groups = []
        new_rotate_boxes = []
        new_crop_mat = []
        new_degree = []
        for group, l in enumerate(lines):
            im_rotate, crop_boxes, rotate_boxes, degree = \
                self.get_line_rotate_image_and_boxes(im_data, l, dets)
            rotate_h = im_rotate.shape[0]
            rotate_w = im_rotate.shape[1]
            for index, i in enumerate(l):
                bbox = dets[i][ :4]
                score = dets[i][4]
                new_clss.append(clss[i])
                new_dets.append(dets[i])
                new_groups.append(group)
                new_rotate_boxes.append(rotate_boxes[index])
                new_degree.append(degree)
                if is_crop:
                   xmin , ymin= crop_boxes[index][0]
                   xmax , ymax = crop_boxes[index][0]
                   for x, y in crop_boxes[index]:
                       xmin = min(x, xmin)
                       ymin = min(y, ymin)
                       xmax = max(x, xmax)
                       ymax = max(y, ymax)
                   h = ymax - ymin
                   w = xmax - xmin
                   diff = 0
                   if h > w:
                       diff = h - w
                       #xmax = xmax + int(diff/6)
                       #xmin = xmin - int(diff/6)
                       xmax = xmax + int(diff/4)
                       xmin = xmin - int(diff/4)
                       ymax = ymax - int(diff/4)
                       ymin = ymin + int(diff/4)
                       print 'diff:', diff
                   if xmax >= rotate_w:
                       xmax = rotate_w -1
                   if xmin < 0:
                       xmin = 0
                   if ymax >= rotate_h:
                       ymax = rotwate_h -1
                   if ymin < 0:
                       ymin = 0

                   crop_mat = im_rotate[ymin:ymax, xmin:xmax]
                   new_crop_mat.append(crop_mat)
        return new_clss, new_dets, new_groups, new_rotate_boxes, new_crop_mat, new_degree

    def detect_image(self, im_data):
        class_res = self.detect(im_data)
        for k, v in class_res.items():
            self.draw_detection_result(im_data, k, v)
        r, i = cv2.imencode('.jpg', im_data)
        return i.data
