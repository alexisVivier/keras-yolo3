import os, random
import datetime
import numpy as np
import pandas as pd
import cv2

import matplotlib.pyplot as plt
from matplotlib import ticker
import seaborn as sns

from keras.models import Sequential
from keras.layers import Input, Dropout, Flatten, Convolution2D, MaxPooling2D, Dense, Activation
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping

from keras.preprocessing.image import ImageDataGenerator, load_img
from keras.utils import np_utils, get_file
from keras import backend, regularizers

DRIVE_DIR = './'
TRAIN_DIR = DRIVE_DIR + 'Train-Set/'
# TEST_DIR = DRIVE_DIR + 'Test-Set/'
EXTRACTED_FRAMES = 'Extracted-Frames-1280x720/'
DRONE_1 = 'Drone1/'
DRONE_2 = 'Drone2/'
MORNING = 'Morning/'
NOON = 'Noon/'
LABELS = 'Labels/'
# la shape de nos images TODO : DEFINIR LA SHAPE
ROWS = 64
COLS = 64
CHANNELS = 3


def get_drone_videos(path):
    f = []
    for (dirpath, dirnames, filenames) in os.walk(path):
        for file in filenames:
            # if file.endswith('.mp4') or file.endswith('.MP4'):
            f.append(file)
    print(f)
    return f


def get_video_labels(video_name, train=True):
    if train == True:
        return TRAIN_DIR + LABELS + 'SingleActionLabels/3840x2160/' + video_name[0:len(video_name) - 4] + '.txt'
    else:
        return TEST_DIR + LABELS + 'SingleActionLabels/3840x2160/' + video_name[0:len(video_name) - 4] + '.txt'


# first_vid_label = get_video_labels()
# print(first_vid_label)

def get_labels(file_path):
    res = []
    with open(file_path) as file:
        for line in file:
            res.append(line.split())
        return res
    # return np.genfromtxt(file_path,delimiter=" ", filling_values= "", missing_values='none')


def reformat_labels(labels, imgs_path):
    f = open("demofile3.txt", "w")
    for (idx, img_path) in enumerate(imgs_path):

        f.write("preprocessedimgs/DRONE_1/1.1.10.mp4/%s %s,%s,%s,%s,0 \r" % (
            'frame%d.jpg' % idx, labels[idx][1], labels[idx][2], labels[idx][3], labels[idx][4],
            ))
    f.close()


images = get_drone_videos(
    "C:/Users/alexi/Documents/Ynov/MasterIA/M1/Deep_learning/developpement/keras-yolo3/preprocessedimgs/DRONE_1/1.1.10.mp4")
print(images)
labels = get_labels(
    "C:/Users/alexi/Documents/Ynov/MasterIA/M1/Deep_learning/developpement/keras-yolo3/Train-Set/Labels/SingleActionLabels/3840x2160/1.1.10.txt")
print(labels)

reformat_labels(labels, images)
