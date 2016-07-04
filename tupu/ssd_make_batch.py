import matplotlib
matplotlib.use('pdf')
import lmdb
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../python'))
import caffe
import caffe.proto.caffe_pb2 as pb
import cv2
import json
import PIL
import random

def read_file_datum(filepath, datum):
    datum.encoded = True
    datum.label = -1
    mat = cv2.imread(filepath, cv2.CV_LOAD_IMAGE_COLOR)
    h = mat.shape[0]
    w = mat.shape[1]
    r, i = cv2.imencode('.jpg', mat)
    assert r
    assert i.data
    datum.data = str(i.data)
    return h, w

def object2string(filepath, obj):
    an_datum = pb.AnnotatedDatum()
    an_datum.type = pb.AnnotatedDatum.BBOX
    h, w = read_file_datum(filepath, an_datum.datum)
    group = {}
    for box in obj:
        cls = box['name']
        if cls not in group:
            group[cls] = []
        group[cls].append(box)
    for k, v in group.items():
        g = an_datum.annotation_group.add()
        g.group_label = 15
        for i in range(len(v)):
            instance = g.annotation.add()
            instance.instance_id = i
            bbox = instance.bbox
            bbox.xmin = float(v[i]['xmin'])/w
            bbox.ymin = float(v[i]['ymin'])/h
            bbox.xmax = float(v[i]['xmax'])/w
            bbox.ymax = float(v[i]['ymax'])/h
            bbox.difficult = (int(v[i]['difficult']) <> 0)
    return an_datum.SerializeToString()

def gen_data_from_json(json_data, db_path):
    reload(sys)
    sys.setdefaultencoding('utf-8')
    label_path = json_data.get('fileListJSON')
    f = open(label_path, 'r')
    data_set = json.load(f)
    rooturis = json_data['rootUris']
    objs = {}
    for k, v in data_set.items():
        for item in v:
            if len(item) == 0:
                continue
            rootpath = rooturis[k]['rootUri']
            filepath = rootpath + '/' + item['file_name']
            if not os.path.exists(filepath):
                print 'image not exits:', filepath
                continue
            try:
                tm = cv2.imread(filepath)
                if tm is None:
                    print 'image None:', filepath
                    continue
            except Exception,ex:
                print 'image read error:', filepath, ' Exception: ', ex
                continue

            label = item['label']
            is_ok = True

            w,h = PIL.Image.open(filepath).size
            for box in label:
                if float(box['xmin']) > float(box['xmax']) \
                    or float(box['ymin']) > float(box['ymax']) \
                    or float(box['xmin']) < 1 or float(box['ymin']) < 1 \
                    or float(box['xmax']) >= w or float(box['ymax']) >= h:
                    print 'label error:', filepath
                    is_ok = False
                    break
            if is_ok:
                objs[filepath] = label
    print 'total image:%d' % len(objs)
    cnt = 0
    arr = objs.items()
    random.shuffle(arr)
    db = lmdb.open(db_path, map_size=int(1e12))
    with db.begin(write=True) as in_txn:
    	for k,v in arr:
            key = '%8d' % cnt
            in_txn.put(key, object2string(k, v))
            cnt += 1
    db.close()
    print 'finish:', db_path

def make_batch(config, util):
    batches_dir = config['batches_dir']
    if not os.path.exists(batches_dir):
        os.makedirs(batches_dir)
    gen_data_from_json(config, batches_dir)
'''
if __name__ == "__main__":
    #json_path = '/world/data-c6/dl-data/57328ea10c4ac91c23d95f72/57500cf408b3893435513c63/14648639883950.588590997736901.json'
    json_path = '/world/data-c6/dl-data/57328ea10c4ac91c23d95f72/575f6689a3c08454158bd197/14658699625140.9822487544734031.json'
    db_path = 'train_lmdb'
    json_data = json.load(open(json_path, 'r'))
    gen_data_from_json(json_data,db_path)
'''
