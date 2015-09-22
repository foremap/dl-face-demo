import os
import sys
from flask import Flask, request, render_template, session, flash, redirect, \
    url_for, jsonify
import subprocess
from pymongo import MongoClient
import Image
from lsh_index import LSHSearch
from feature_extractor import Feature
import time
import zmq
from collections import defaultdict
import csv

context = zmq.Context()
socket = context.socket(zmq.REQ)
socket.connect("tcp://127.0.0.1:5566")
base_path = os.path.dirname(__file__)

app = Flask(__name__)
# build lsh engine
lsh_engine = LSHSearch(os.path.join(
    base_path, '../', 'files',
    'features', 'FaceScrub_feature_small.txt'),
    150, 30, 8)
lsh_engine.build()
# init feature extractor
model = os.path.join(
    base_path, '../', 'files',
    'caffe', 'Deepface_softmax_val_CASIA.prototxt')

caffemodel = os.path.join(
    base_path, '../', 'files',
    'caffe', 'face_siamese_CASIA_data_iter_260000.caffemodel')

feature_extractor = Feature(model, caffemodel, 50, 'pool5')

app.config['UPLOAD_FOLDER'] = os.path.join(base_path, '../', 'data', 'input')
app.config['ALLOWED_EXTENSIONS'] = set(['jpg', 'jpeg', 'gif', 'png'])


client = MongoClient('10.116.66.16', 27017)

maxwidth = 480.0
maxheight = 480.0

celebrities_map_file = os.path.join(
    base_path, '../', 'files', 'PersonMap.txt')
celebrities_map = defaultdict(int)
with open(celebrities_map_file, 'rb') as f:
    reader = csv.reader(f, delimiter=' ')
    for name, idx in reader:
        celebrities_map[name] = idx


@app.route('/face_rec', methods=['GET', 'POST'])
def face_rec():
    if request.method == 'POST':
        tstart = time.time()
        t_tmp = tstart
        img = request.files['file']
        img_name = to_utf8(img.filename)
        if img and allowed_file(img_name):
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], img_name)
            img = Image.open(img)
            (width, height) = img.size
            ratio = min(maxwidth/width, maxheight/height, 1)
            size = int(width * ratio), int(height * ratio)
            print 'Downsample to size : ', size
            img.thumbnail(size, Image.ANTIALIAS)
            img.save(file_path)

            print 'Resize take : ', time.time() - t_tmp, ' seconds'
            t_tmp = time.time()

            socket.send(img_name)
            print socket.recv()
            print 'Detect take : ', time.time() - t_tmp, ' seconds'
            t_tmp = time.time()

            # dlib always return jpg file
            filename, file_extension = os.path.splitext(img_name)
            img_name = filename + '.jpg'

            feature = extract_feature(img_name)
            print 'Feature take : ', time.time() - t_tmp, ' seconds'
            t_tmp = time.time()
            if 'Error' not in feature:
                res = ann_rec(feature)
                res['status'] = 'Sucesses'
            else:
                res = {}
                res['status'] = 'Error'
                res['error_msg'] = feature
            res['time'] = time.time() - tstart
            print 'Search take : ', time.time() - t_tmp, ' seconds'
            print 'Total take : ', res['time'], ' seconds'
        else:
            res = {}
            res['status'] = 'Error'
            res['error_msg'] = 'File format is not allowed'
        print res
        return jsonify(res)


@app.route('/imdb/<index>', methods=['GET'])
def imdb_data(index):
    res = query_imdb(index)
    return jsonify(res)


@app.route('/tag', methods=['POST'])
def tag():
    img = request.files['file']
    img_name = to_utf8(img.filename)
    person = to_utf8(request.form['tag'])
    print img_name, person
    res = {}
    if img and allowed_file(img_name):
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], img_name)
        img = Image.open(img)
        (width, height) = img.size
        ratio = min(maxwidth/width, maxheight/height, 1)
        size = int(width * ratio), int(height * ratio)
        print 'Downsample to size : ', size
        img.thumbnail(size, Image.ANTIALIAS)
        img.save(file_path)

        socket.send(img_name)
        print socket.recv()

        # dlib always return jpg file
        filename, file_extension = os.path.splitext(img_name)
        img_name = filename + '.jpg'

        feature = extract_feature(img_name)
        lsh_engine.update(person, feature)
        res['status'] = 'Sucesses'
    else:
        res['status'] = 'Error'
        res['error_msg'] = 'File format is not allowed'
    return jsonify(res)


# Function Define #
def query_imdb(index):
    db = client.face_demo
    collection = db.celebrities_new
    res = collection.find_one({"idx": int(index)})
    del res['_id']
    return res


def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1] in app.config['ALLOWED_EXTENSIONS']


def extract_feature(filename):
    full_path = os.path.join(base_path, '../', 'data', 'output')
    if os.path.isfile(os.path.join(full_path, 'single', filename)):
        feature = feature_extractor.get_feature(
            os.path.join(full_path, 'single', filename))
        res = ','.join(str(e) for e in feature)
    elif os.path.isfile(os.path.join(full_path, 'others', filename)):
        res = "(Error) No Face"
    else:
        res = "(Error) Face Detector Failed"
    remove(os.path.join(full_path, 'single', filename))
    return res


def ann_rec(feature):
    result = lsh_engine.query(feature)
    # print result[:5]
    name = '_'.join(result[0][0].split('_')[:-1])
    idx = celebrities_map[name]
    res = query_imdb(idx)
    res['Confidence'] = 1 - float(result[0][1])
    return res


def remove(file_path):
    try:
        os.remove(file_path)
    except OSError:
        pass


def to_utf8(text):
    if text:
        return text.encode('utf8', 'replace')
    else:
        return None


def run(cmd, need_return=False):
    print "run the cmd => " + cmd
    if need_return:
        p = subprocess.Popen(
            cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    else:
        p = subprocess.Popen(cmd, shell=True, stderr=subprocess.PIPE)

    while True:
        out = p.stderr.read(1)
        if out == '' and p.poll() != None:
            break
        if out != '':
            sys.stdout.write(out)
            sys.stdout.flush()
    print '------- run cmd end ------'
    out1, err1 = p.communicate()

    if need_return:
        result = out1.decode('utf8')
    else:
        result = ""

    return result

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=8888)


# Test #
# curl -F "file=@test.jpg;" http://localhost:8888/face_rec
# curl -F "file=@test.jpg" -F "tag=girl_generation" http://localhost:8888/tag