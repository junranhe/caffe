import matplotlib
matplotlib.use('pdf')

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../python'))
#import caffe
import cv2
import tornado
import tornado.ioloop
import tornado.httpclient
import tornado.web
import numpy as np
#caffe.set_mode_gpu()
#caffe.set_device(1)
#prototxt = '/world/data-c5/ssd_models/text_batch_64/deploy.prototxt'
#caffemodel = '/world/data-c5/ssd_models/text_batch_64/VGG_SSD_300x300_iter_80000.caffemodel'
prototxt = '/home/kevin/char_output/deploy.prototxt'
caffemodel = '/home/kevin/char_output/VGG_SSD_300x300_iter_30000.caffemodel'
#mean = np.array([104, 117, 123], np.uint8)
#net = caffe.Detector(prototxt, caffemodel,mean=mean)
import ssd_detector

ssd_detector.gpu_init(1)
net = ssd_detector.SSDDetector(caffemodel, prototxt, None)
'''
def im_detect(net, im):
    h = im.shape[0]
    w = im.shape[1]
    r_im = cv2.resize(im, (300, 300))
    blob = np.zeros((1, 3, 300,300), np.float32)
    
    in_ = net.inputs[0]
    import time
    pre_time = time.time()
    blob[0] = net.transformer.preprocess(in_, r_im)
    forward_kwargs = {in_: blob}
    blobs_out = net.forward(**forward_kwargs)
    res = blobs_out[net.outputs[0]] 
    print 'run time', time.time() - pre_time
    font = cv2.FONT_HERSHEY_SIMPLEX
    for i in range(res.shape[2]):
        box = res[0,0,i]
        label = box[1]
        score = box[2]
        if score < 0.1:
            continue
        xmin = int(box[3]*w)
        ymin = int(box[4]*h)
        xmax = int(box[5]*w)
        ymax = int(box[6]*h)
       
        cv2.rectangle(im,(xmin, ymin), (xmax, ymax), (0,255,0), 2)
        cv2.putText(im, str(label), (xmin, ymin), font, 0.5, (0, 0, 255), 1)
'''

class TestHandler(tornado.web.RequestHandler):
    def post(self):
        global net
        for f in self.request.files['file']:
            im = cv2.imdecode(np.asarray(bytearray(f['body']), dtype = np.uint8),\
                              cv2.IMREAD_COLOR)
            image_data = net.detect_image_ex(im)
	    print 'ok'
            if image_data:
                self.set_header('Content-Type', 'image/jpg')
                self.write(bytes(image_data))

class MainHandler(tornado.web.RequestHandler):
    def get(self):
        self.render('detect.html', search = '/test', writelog = '', returnupimg='')
settings = {
    'template_path': os.path.join(os.path.dirname(__file__) , 'template'),
    'debug': True
}

application = tornado.web.Application([('/', MainHandler),\
                                       ('/test',TestHandler)], **settings)

import signal
print 'server start'
def sig_handler(sig, frame):
    if sig in [signal.SIGINT, signal.SIGTERM]:
        #print 'kill svr %d' % sig
        tornado.ioloop.IOLoop.instance().stop()

if __name__ == '__main__':
    application.listen(20002)
    signal.signal(signal.SIGINT, sig_handler)
    signal.signal(signal.SIGTERM, sig_handler)
    tornado.ioloop.IOLoop.instance().start()

