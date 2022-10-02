"""
brief: face alignment with FFHQ method (https://github.com/NVlabs/ffhq-dataset)
author: lzhbrian (https://lzhbrian.me)
date: 2020.1.5
note: code is heavily borrowed from 
    https://github.com/NVlabs/ffhq-dataset
    http://dlib.net/face_landmark_detection.py.html

requirements:
    apt install cmake
    conda install Pillow numpy scipy
    pip install dlib
    # download face landmark model from:
    # http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
"""
from argparse import ArgumentParser
import time
import numpy as np
import PIL
import PIL.Image
import os
import scipy
import scipy.ndimage
import dlib
import multiprocessing as mp
import math

#from configs.paths_config import model_paths
SHAPE_PREDICTOR_PATH = 'shape_predictor_68_face_landmarks.dat'#model_paths["shape_predictor"]


def get_landmark(filepath, predictor):
    """get landmark with dlib
    :return: np.array shape=(68, 2)
    """
    detector = dlib.get_frontal_face_detector()
    if type(filepath) == str:
        img = dlib.load_rgb_image(filepath)
    else:
        img = filepath
    dets = detector(img, 1)
    
    if len(dets) == 0:
        print('Error: no face detected! If you are sure there are faces in your input, you may rerun the code or change the image several times until the face is detected. Sometimes the detector is unstable.')
        return None
    
    shape = None
    for k, d in enumerate(dets):
        shape = predictor(img, d)
    
    t = list(shape.parts())
    a = []
    for tt in t:
        a.append([tt.x, tt.y])
    lm = np.array(a)
    return lm


def align_face(filepath, predictor):
    """
    :param filepath: str
    :return: PIL Image
    """

    lm = get_landmark(filepath, predictor)
    if lm is None:
        return None    
    
    lm_chin = lm[0: 17]  # left-right
    lm_eyebrow_left = lm[17: 22]  # left-right
    lm_eyebrow_right = lm[22: 27]  # left-right
    lm_nose = lm[27: 31]  # top-down
    lm_nostrils = lm[31: 36]  # top-down
    lm_eye_left = lm[36: 42]  # left-clockwise
    lm_eye_right = lm[42: 48]  # left-clockwise
    lm_mouth_outer = lm[48: 60]  # left-clockwise
    lm_mouth_inner = lm[60: 68]  # left-clockwise

    # Calculate auxiliary vectors.
    eye_left = np.mean(lm_eye_left, axis=0)
    eye_right = np.mean(lm_eye_right, axis=0)
    eye_avg = (eye_left + eye_right) * 0.5
    eye_to_eye = eye_right - eye_left
    mouth_left = lm_mouth_outer[0]
    mouth_right = lm_mouth_outer[6]
    mouth_avg = (mouth_left + mouth_right) * 0.5
    eye_to_mouth = mouth_avg - eye_avg

    # Choose oriented crop rectangle.
    x = eye_to_eye - np.flipud(eye_to_mouth) * [-1, 1]
    x /= np.hypot(*x)
    x *= max(np.hypot(*eye_to_eye) * 2.0, np.hypot(*eye_to_mouth) * 1.8)
    y = np.flipud(x) * [-1, 1]
    c = eye_avg + eye_to_mouth * 0.1
    quad = np.stack([c - x - y, c - x + y, c + x + y, c + x - y])
    qsize = np.hypot(*x) * 2

    # read image
    if type(filepath) == str:
        img = PIL.Image.open(filepath)
    else:
        img = PIL.Image.fromarray(filepath)

    output_size = 256
    transform_size = 256
    enable_padding = True

    # Shrink.
    shrink = int(np.floor(qsize / output_size * 0.5))
    if shrink > 1:
        rsize = (int(np.rint(float(img.size[0]) / shrink)), int(np.rint(float(img.size[1]) / shrink)))
        img = img.resize(rsize, PIL.Image.ANTIALIAS)
        quad /= shrink
        qsize /= shrink

    # Crop.
    border = max(int(np.rint(qsize * 0.1)), 3)
    crop = (int(np.floor(min(quad[:, 0]))), int(np.floor(min(quad[:, 1]))), int(np.ceil(max(quad[:, 0]))),
            int(np.ceil(max(quad[:, 1]))))
    crop = (max(crop[0] - border, 0), max(crop[1] - border, 0), min(crop[2] + border, img.size[0]),
            min(crop[3] + border, img.size[1]))
    if crop[2] - crop[0] < img.size[0] or crop[3] - crop[1] < img.size[1]:
        img = img.crop(crop)
        quad -= crop[0:2]

    # Pad.
    pad = (int(np.floor(min(quad[:, 0]))), int(np.floor(min(quad[:, 1]))), int(np.ceil(max(quad[:, 0]))),
           int(np.ceil(max(quad[:, 1]))))
    pad = (max(-pad[0] + border, 0), max(-pad[1] + border, 0), max(pad[2] - img.size[0] + border, 0),
           max(pad[3] - img.size[1] + border, 0))
    if enable_padding and max(pad) > border - 4:
        pad = np.maximum(pad, int(np.rint(qsize * 0.3)))
        img = np.pad(np.float32(img), ((pad[1], pad[3]), (pad[0], pad[2]), (0, 0)), 'reflect')
        h, w, _ = img.shape
        y, x, _ = np.ogrid[:h, :w, :1]
        mask = np.maximum(1.0 - np.minimum(np.float32(x) / pad[0], np.float32(w - 1 - x) / pad[2]),
                          1.0 - np.minimum(np.float32(y) / pad[1], np.float32(h - 1 - y) / pad[3]))
        blur = qsize * 0.02
        img += (scipy.ndimage.gaussian_filter(img, [blur, blur, 0]) - img) * np.clip(mask * 3.0 + 1.0, 0.0, 1.0)
        img += (np.median(img, axis=(0, 1)) - img) * np.clip(mask, 0.0, 1.0)
        img = PIL.Image.fromarray(np.uint8(np.clip(np.rint(img), 0, 255)), 'RGB')
        quad += pad[:2]

    # Transform.
    img = img.transform((transform_size, transform_size), PIL.Image.QUAD, (quad + 0.5).flatten(), PIL.Image.BILINEAR)
    if output_size < transform_size:
        img = img.resize((output_size, output_size), PIL.Image.ANTIALIAS)

    # Save aligned image.
    return img


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def extract_on_paths(file_paths):
    predictor = dlib.shape_predictor(SHAPE_PREDICTOR_PATH)
    pid = mp.current_process().name
    print('\t{} is starting to extract on #{} images'.format(pid, len(file_paths)))
    tot_count = len(file_paths)
    count = 0
    for file_path, res_path in file_paths:
        count += 1
        if count % 100 == 0:
            print('{} done with {}/{}'.format(pid, count, tot_count))
        try:
            res = align_face(file_path, predictor)
            res = res.convert('RGB')
            os.makedirs(os.path.dirname(res_path), exist_ok=True)
            res.save(res_path)
        except Exception:
            continue
    print('\tDone!')


def parse_args():
    parser = ArgumentParser(add_help=False)
    parser.add_argument('--num_threads', type=int, default=1)
    parser.add_argument('--root_path', type=str, default='')
    args = parser.parse_args()
    return args


def run(args):
    root_path = args.root_path
    out_crops_path = root_path + '_crops'
    if not os.path.exists(out_crops_path):
        os.makedirs(out_crops_path, exist_ok=True)

    file_paths = []
    for root, dirs, files in os.walk(root_path):
        for file in files:
            file_path = os.path.join(root, file)
            fname = os.path.join(out_crops_path, os.path.relpath(file_path, root_path))
            res_path = '{}.jpg'.format(os.path.splitext(fname)[0])
            if os.path.splitext(file_path)[1] == '.txt' or os.path.exists(res_path):
                continue
            file_paths.append((file_path, res_path))

    file_chunks = list(chunks(file_paths, int(math.ceil(len(file_paths) / args.num_threads))))
    print(len(file_chunks))
    pool = mp.Pool(args.num_threads)
    print('Running on {} paths\nHere we goooo'.format(len(file_paths)))
    tic = time.time()
    pool.map(extract_on_paths, file_chunks)
    toc = time.time()
    print('Mischief managed in {}s'.format(toc - tic))


if __name__ == '__main__':
    args = parse_args()
    run(args)
