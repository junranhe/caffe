import matplotlib
matplotlib.use('pdf')
import caffe
import sys
prototxt = '/home/kevin/ssd/caffe/examples/ssd/test.prototxt'

caffemodel = '/home/kevin/ssd/caffe/models/VGGNet/VOC0712/SSD_300x300/VGG_VOC0712_SSD_300x300_iter_60000.caffemodel'
#imagepath = sys.argv[1]
import cv2
#im = cv2.imread(imagepath)
#im = cv2.resize(im, (300, 300))

import numpy as np
caffe.set_mode_gpu()
caffe.set_device(0)
net = caffe.Net(prototxt, caffemodel, caffe.TEST)

def im_detect(net, im, shresh):
    w = im.shape[0]
    h = im.shape[1]
    mean = np.array([104, 117, 123], np.uint8)
    r_im = cv2.resize(im.copy(), (300, 300))
    r_im -= mean
    blob = np.zeros((1, 300, 300, 3), np.float32)
    blob[0] = r_im.astype(np.float32, copy=True)
    blob = blob.transpose(0,3,1,2)
    forward_kwargs = {'data': blob}
    blobs_out = net.forward(**forward_kwargs)
    res = net.blobs['detection_out'].data.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    for i in range(res.shape[2]):
        box = res[0,0,i]
        label = box[1]
        score = box[2]
        xmin = int(box[3]*w)
        ymin = int(box[4]*h)
        xmax = int(box[5]*w)
        ymax = int(box[6]*h)
        if score < shresh:
            continue
        cv2.rectangle(im,(xmin, ymin), (xmax, ymax), (0,255,0), 2)
        cv2.putText(im, str(label), (xmin, ymin), font, 0.5, (0, 0, 255), 1)
import tornado
import tornado.ioloop
import tornado.httpclient
import tornado.web
class TestHandler(tornado.web.RequestHandler):
    def post(self):
        global net
        for f in self.request.files['file']:
            im = cv2.imdecode(np.asarray(bytearray(f['body']), dtype = np.uint8),\
                              cv2.IMREAD_COLOR)
            new_im = im_detect(net, im, 0.6)
            r, i = cv2.imencode('.jpg', im)
            if i.data:
                self.set_header('Content-Type', 'image/jpg')
                self.write(bytes(i.data))

class MainHandler(tornado.web.RequestHandler):
    def get(self):
        self.render('detect.html', search = '/test', writelog = '', returnupimg='')
settings = {
    'template_path':'/home/kevin/tp_server/tpdetect/template',
    'debug': True
}

application = tornado.web.Application([('/', MainHandler),\
                                       ('/test',TestHandler)], **settings)

import signal

def sig_handler(sig, frame):
    if sig in [signal.SIGINT, signal.SIGTERM]:
        #print 'kill svr %d' % sig
        tornado.ioloop.IOLoop.instance().stop()

if __name__ == '__main__':
    application.listen(20002)
    signal.signal(signal.SIGINT, sig_handler)
    signal.signal(signal.SIGTERM, sig_handler)
    tornado.ioloop.IOLoop.instance().start()

