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


DRIVE_DIR='./'
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

os.chdir(DRIVE_DIR)

train_images = [TRAIN_DIR+i for i in os.listdir(TRAIN_DIR)]
#test_images =  [TEST_DIR+i for i in os.listdir(TEST_DIR)]

def read_image(file_path):
    img = cv2.imread(file_path, cv2.IMREAD_COLOR)
    return cv2.resize(img, (ROWS, COLS), interpolation=cv2.INTER_CUBIC)

train_drone1_vids_path = TRAIN_DIR+'Drone1/'
# train_drone2_vids_path = TRAIN_DIR+'Drone2/'

def get_drone_videos(path):
    f = []
    for (dirpath, dirnames, filenames) in os.walk(path):
        for file in filenames:
            if file.endswith('.mp4') or file.endswith('.MP4'):
                f.append(file)
    print(f)
    return f


train_drone1_videos = get_drone_videos(train_drone1_vids_path)
# train_drone2_videos = get_drone_videos(train_drone2_vids_path)

def get_video_path(drone_dir, video_name):
    if video_name.startswith('1.1') or video_name.startswith('2.1'):
        return TRAIN_DIR+drone_dir+MORNING+video_name
    # else:
    #     return TEST_DIR+drone_dir+NOON+video_name


print(cv2.__version__)
def video_to_images_converter(file_path, directory):
    vidcap = cv2.VideoCapture(file_path)
    success, image = vidcap.read()
    count = 0
    images = []
    while success:
        cv2.imwrite("frame%d.jpg" % count, image)  # save frame as JPEG file
        images.append(directory + "/frame%d.jpg" % count)
        print(images)
        os.chdir(directory)
        success, image = vidcap.read()
        if count%250 == 0: print("Processed %d images" % count)
        count += 1
    print(count)
    return images


os.chdir(DRIVE_DIR)

for video in train_drone1_videos:
    os.chdir(DRIVE_DIR)
    directory = "C:/Users/alexi/Documents/Ynov/MasterIA/M1/Deep_learning/developpement/keras-yolo3/preprocessedimgs/DRONE_1/" + video
    os.mkdir(directory)
    video_path = get_video_path(DRONE_1, video)
    video_to_images_converter(video_path, directory)

# for video in train_drone_videos:
#     directory = "C:/Users/alexi/Documents/Ynov/MasterIA/M1/Deep learning/developpement/keras-yolo3/preprocessedimgs/DRONE_2/" + video
#     os.mkdir(directory)
#     video_path = get_video_path(DRONE_2, video)
#     video_to_images_converter(video_path, directory)


# def get_video_labels(video_name, train= True):
#     if train == True:
#         return TRAIN_DIR + LABELS + 'SingleActionLabels/3840x2160/' + video_name[0:len(video_name)-4]+'.txt'
#     else:
#         return TEST_DIR + LABELS + 'SingleActionLabels/3840x2160/' + video_name[0:len(video_name)-4]+'.txt'
#
# first_vid_label = get_video_labels(train_drone1_videos[0])
# print(first_vid_label)
#
# def get_labels(file_path):
#     res = []
#     with open(file_path) as file:
#         for line in file:
#           res.append(line.split())
#         return res
#     # return np.genfromtxt(file_path,delimiter=" ", filling_values= "", missing_values='none')
#
# labels = get_labels(first_vid_label)
# print(labels)
