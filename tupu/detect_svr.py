# -*- coding: utf-8 -*-
import matplotlib
matplotlib.use('pdf')
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../python'))
#import caffe
import cv2
import tornado
import tornado.ioloop
import tornado.httpclient
import tornado.web
import numpy as np
import json
#prototxt = '/world/data-c6/dl-data/57328ea10c4ac91c23d95f72/57b58a8ed444324f304cd7d4/deploy.prototxt'

#caffemodel = '/world/data-c6/dl-data/57328ea10c4ac91c23d95f72/57b58a8ed444324f304cd7d4/VGG_SSD_300x300_iter_70000.caffemodel'
prototxt = '/world/data-c5/ssd_test/angle_debug_output/deploy.prototxt'
caffemodel = '/world/data-c5/ssd_test/angle_debug_output/VGG_SSD_500x500_iter_70000.caffemodel'
#dict_json = json.load(open('/world/data-c5/ssd_test/full_char_exp_train_lmdb/dict.json','r'))
#label_dict = {v:k for k, v in dict_json.items()}
import ssd_detector

ssd_detector.gpu_init(6)
net = ssd_detector.SSDDetector(caffemodel, prototxt, None)
class TestHandler(tornado.web.RequestHandler):
    def post(self):
        global net
        for f in self.request.files['file']:
            im = cv2.imdecode(np.asarray(bytearray(f['body']), dtype = np.uint8),\
                              cv2.IMREAD_COLOR)
            image_data = net.detect_image_ex(im, None, True)
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

