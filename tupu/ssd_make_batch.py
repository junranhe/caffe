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
import ssd_util
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

def is_number(uchar):
    return uchar >= u'\u0030' and uchar <=u'\u0039'
def is_char(uchar):
    return (uchar >= u'\u0041' and uchar<=u'\u005a') or (uchar >= u'\u0061' and uchar<=u'\u007a')

def object2string(filepath, obj, label_table=None):
    an_datum = pb.AnnotatedDatum()
    an_datum.type = pb.AnnotatedDatum.BBOX
    h, w = read_file_datum(filepath, an_datum.datum)
    group = {}
    for box in obj:
        #cls = box['name']
        if label_table is not None:
            tm_name = box['label']
            if not (is_number(tm_name) or is_char(tm_name)):
                tm_name = '*'
            if tm_name not in label_table:
                label_table[tm_name] = len(label_table) + 1
            cls = label_table[tm_name]
        else:
            cls = int(box['name'])+1
        if cls not in group:
            group[cls] = []
        group[cls].append(box)
    for k, v in group.items():
        g = an_datum.annotation_group.add()
        g.group_label = int(k)
        for i in range(len(v)):
            instance = g.annotation.add()
            instance.instance_id = i
            bbox = instance.bbox
            bbox.xmin = float(v[i]['xmin'])/w
            bbox.ymin = float(v[i]['ymin'])/h
            bbox.xmax = float(v[i]['xmax'])/w
            bbox.ymax = float(v[i]['ymax'])/h
            bbox.difficult = (int(v[i]['difficult']) <> 0)
            bbox.angle = float(ssd_util.degree2angle(v[i]['degree'], w, h))
    return an_datum.SerializeToString()

def gen_data_from_json(json_data, db_path):
    reload(sys)
    sys.setdefaultencoding('utf-8')
    label_path = json_data.get('fileListJSON')
    f = open(label_path, 'r')
    uriIds = json_data['uriIds']
    objs = {}
    for l in f:
        is_uriId = l[0:-1]
        if is_uriId in uriIds:
            continue
        item = l.split('\t')
        filepath = item[0]
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

        try:
            label = json.loads(item[1])
        except Exception,ex:
            print 'error line:', l
            continue
        is_ok = True
        for box in label:
            xmin = float(box['xmin'])
            ymin = float(box['ymin'])
            xmax = float(box['xmax'])
            ymax = float(box['ymax'])
            degree = 0.
            #print 'old: xmin:%f ymin:%f xmax:%f ymax:%f degree:%f' % (xmin, ymin, xmax, ymax, degree)
            #continue
            if 'degree' in box:
                degree = float(box['degree'])
                xmin, ymin, xmax, ymax =\
                     ssd_util.compute_warp_rec(xmin, ymin, xmax, ymax, degree)
            box['xmin'] = xmin
            box['ymin'] = ymin
            box['xmax'] = xmax
            box['ymax'] = ymax
            box['degree'] = degree

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
    print 'A total of %d images' % len(objs)
    cnt = 0
    arr = objs.items()
    random.shuffle(arr)
    db = lmdb.open(db_path, map_size=int(1e12))
    with db.begin(write=True) as in_txn:
    	for k,v in arr:
            key = '%8d' % cnt
            in_txn.put(key, object2string(k, v))
            cnt += 1
            if cnt % 1000 == 0:
                print 'MeanProcessed %d images' % cnt
    db.close()
    print 'mean_value'

def gen_data_from_json_simple_line(json_data, db_path, dict_path):
    reload(sys)
    sys.setdefaultencoding('utf-8')
    label_path = json_data.get('fileListJSON')
    f = open(label_path, 'r')
    #f = open('/world/data-gpu-57/Tranning-data/frame_Text_rotate/export_full.json', 'r')
    #f1 = open('/world/data-gpu-57/Tranning-data/frame_Text4/export_full.json', 'r')
    lines = f.readlines()
    #lines1 = f1.readlines()
    #lines = lines[0: int(len(lines)/2)]
    #lines1 = lines1[0: int(len(lines1)/2)]
    #lines += lines1
    #data_set = json.load(f)
    #rooturis = json_data['rootUris']
    objs = {}
    #for k, v in data_set.items():
    for line_index, l in enumerate(lines):
        #if len(objs) > 100:
        #    break
        try:
            tm_json = json.loads(l.strip('\n'))
        except:
            print 'error line index:', line_index
            continue
        if len(objs) % 1000 == 0:
            print 'check:', len(objs)
        for k, item in tm_json.items():
            if len(item) == 0:
                continue
            #rootpath = rooturis[k]['rootUri']
            #filepath = rootpath + '/' + item['file_name']
            filepath = k
            #print 'image:', l
            if not os.path.exists(filepath):
                print 'image not exits:', filepath
                continue
            try:
                tm = cv2.imread(filepath)
                if tm is None:
                    print 'image None:', filepath
                    continue
                del tm
            except Exception,ex:
                print 'image read error:', filepath, ' Exception: ', ex
                continue

            label = item
            is_ok = True

            for box in label:
                xmin = float(box['xmin'])
                ymin = float(box['ymin'])
                xmax = float(box['xmax'])
                ymax = float(box['ymax'])
                degree = 0.
                #print 'old: xmin:%f ymin:%f xmax:%f ymax:%f degree:%f' % (xmin, ymin, xmax, ymax, degree)
                #continue
                if 'degree' in box:
                    degree = float(box['degree'])
                    xmin, ymin, xmax, ymax =\
                         ssd_util.compute_warp_rec(xmin, ymin, xmax, ymax, degree)
                box['xmin'] = xmin
                box['ymin'] = ymin
                box['xmax'] = xmax
                box['ymax'] = ymax
                box['degree'] = degree
                #print 'new: xmin:%f ymin:%f xmax:%f ymax:%f degree:%f' % (xmin, ymin, xmax, ymax, degree)
                #exit(0)

            pl_tm = PIL.Image.open(filepath)
            w,h = pl_tm.size
            pl_tm.close()
            del pl_tm
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
        #label_table = {}
    	for k,v in arr:
            key = '%8d' % cnt
            #in_txn.put(key, object2string(k, v, label_table))

            in_txn.put(key, object2string(k, v, None))
            cnt += 1
            if cnt % 1000 == 0:
                print 'save:',cnt
        #dict_file = open(dict_path, 'w')
        #json.dump(label_table, dict_file)
        #print label_table
    db.close()
    print 'finish:', db_path


def make_batch(config, util):
    batches_dir = config['batches_dir']
    if not os.path.exists(batches_dir):
        os.makedirs(batches_dir)
    gen_data_from_json(config, batches_dir)

def make_batch_from_file(filepath):
    config = json.load(open(filepath, 'r'))
    batches_dir = config['batches_dir']
    if not os.path.exists(batches_dir):
        os.makedirs(batches_dir)
    gen_data_from_json(config, batches_dir)



if __name__ == "__main__":
    #json_path = '/home/kevin/tp_server/tpdetect/train_angle_batch.json'
    #db_path = '/world/data-c5/ssd_test/angle_train_lmdb_x10_2'

    json_path = '/world/data-c6/dl-data/57328ea10c4ac91c23d95f72/57b6840264eb0db6507fdb5d/14715791387660.8223382802680135.json'
    db_path = '~/temp_debug'
    json_data = json.load(open(json_path, 'r'))
    gen_data_from_json(json_data,json_data['batches_dir'])

