# USAGE
# python /home/nmorales/cxgn/DroneImageScripts/CNN/KerasCNNSequentialSoftmaxCategorical.py --input_image_label_file  /folder/myimagesandlabels.csv --output_model_file_path /folder/mymodel.h5 --outfile_path /export/myresults.csv

# import the necessary packages
import sys
import argparse
import csv
import imutils
import cv2
import numpy as np
import math
import pandas as pd
from PIL import Image
from sklearn import preprocessing
from tensorflow.keras import Sequential
from tensorflow.keras import Model
from tensorflow.keras import Input
from tensorflow.keras import backend
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import Lambda
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-l", "--log_file_path", required=False, help="file path to write log to. useful for using from the web interface")
ap.add_argument("-i", "--input_image_label_file", required=True, help="file path for file holding image names and labels to be trained")
ap.add_argument("-m", "--output_model_file_path", required=True, help="file path for saving keras model, so that it can be loaded again in the future. it saves an hdf5 file as the model")
ap.add_argument("-o", "--outfile_path", required=True, help="file path where the output will be saved")
ap.add_argument("-c", "--output_class_map", required=True, help="file path where the output for class map will be saved")
args = vars(ap.parse_args())

log_file_path = args["log_file_path"]
input_file = args["input_image_label_file"]
output_model_file_path = args["output_model_file_path"]
outfile_path = args["outfile_path"]
output_class_map = args["output_class_map"]

if sys.version_info[0] < 3:
    raise Exception("Must use Python3. Use python3 in your command line.")

if log_file_path is not None:
    sys.stderr = open(log_file_path, 'a')

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

def conv2d(x,numfilt,filtsz,strides=1,pad='same',act=True,name=None):
    x = Conv2D(numfilt,filtsz,strides=strides,padding=pad,data_format='channels_last',use_bias=False,name=name+'conv2d')(x)
    x = BatchNormalization(axis=3,scale=False,name=name+'conv2d'+'bn')(x)
    if act:
        x = Activation('relu',name=name+'conv2d'+'act')(x)
    return x

def incresA(x,scale,name=None):
    pad = 'same'
    branch0 = conv2d(x,32,1,1,pad,True,name=name+'b0')
    branch1 = conv2d(x,32,1,1,pad,True,name=name+'b1_1')
    branch1 = conv2d(branch1,32,3,1,pad,True,name=name+'b1_2')
    branch2 = conv2d(x,32,1,1,pad,True,name=name+'b2_1')
    branch2 = conv2d(branch2,48,3,1,pad,True,name=name+'b2_2')
    branch2 = conv2d(branch2,64,3,1,pad,True,name=name+'b2_3')
    branches = [branch0,branch1,branch2]
    mixed = Concatenate(axis=3, name=name + '_concat')(branches)
    filt_exp_1x1 = conv2d(mixed,384,1,1,pad,False,name=name+'filt_exp_1x1')
    final_lay = Lambda(lambda inputs, scale: inputs[0] + inputs[1] * scale,
                      output_shape=backend.int_shape(x)[1:],
                      arguments={'scale': scale},
                      name=name+'act_scaling')([x, filt_exp_1x1])
    return final_lay

def incresB(x,scale,name=None):
    pad = 'same'
    branch0 = conv2d(x,192,1,1,pad,True,name=name+'b0')
    branch1 = conv2d(x,128,1,1,pad,True,name=name+'b1_1')
    branch1 = conv2d(branch1,160,[1,7],1,pad,True,name=name+'b1_2')
    branch1 = conv2d(branch1,192,[7,1],1,pad,True,name=name+'b1_3')
    branches = [branch0,branch1]
    mixed = Concatenate(axis=3, name=name + '_mixed')(branches)
    filt_exp_1x1 = conv2d(mixed,1152,1,1,pad,False,name=name+'filt_exp_1x1')
    final_lay = Lambda(lambda inputs, scale: inputs[0] + inputs[1] * scale,
                      output_shape=backend.int_shape(x)[1:],
                      arguments={'scale': scale},
                      name=name+'act_scaling')([x, filt_exp_1x1])
    return final_lay

def incresC(x,scale,name=None):
    pad = 'same'
    branch0 = conv2d(x,192,1,1,pad,True,name=name+'b0')
    branch1 = conv2d(x,192,1,1,pad,True,name=name+'b1_1')
    branch1 = conv2d(branch1,224,[1,3],1,pad,True,name=name+'b1_2')
    branch1 = conv2d(branch1,256,[3,1],1,pad,True,name=name+'b1_3')
    branches = [branch0,branch1]
    mixed = Concatenate(axis=3, name=name + '_mixed')(branches)
    filt_exp_1x1 = conv2d(mixed,2048,1,1,pad,False,name=name+'fin1x1')
    final_lay = Lambda(lambda inputs, scale: inputs[0] + inputs[1] * scale,
                      output_shape=backend.int_shape(x)[1:],
                      arguments={'scale': scale},
                      name=name+'act_saling')([x, filt_exp_1x1])
    return final_lay

unique_labels = {}
unique_image_types = {}
unique_drone_run_band_names = {}
labels = []
data = []

print("[INFO] reading labels and image data...")
with open(input_file) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    for row in csv_reader:
        stock_id = row[0]
        trait_name = row[3]
        image_type = row[4]
        drone_run_band_name = row[5]
        image = Image.open(row[1])
        image = np.array(image.resize((75,75))) / 255.0

        if (len(image.shape) == 2):
            empty_mat = np.ones(image.shape, dtype=image.dtype) * 0
            image = cv2.merge((image, empty_mat, empty_mat))

        #print(image.shape)
        data.append(image)

        value = float(row[2])
        labels.append(value)

        if value in unique_labels.keys():
            unique_labels[str(value)] += 1
        else:
            unique_labels[str(value)] = 1

        if image_type in unique_image_types.keys():
            unique_image_types[image_type] += 1
        else:
            unique_image_types[image_type] = 1

        if drone_run_band_name in unique_drone_run_band_names.keys():
            unique_drone_run_band_names[drone_run_band_name] += 1
        else:
            unique_drone_run_band_names[drone_run_band_name] = 1

lines = []
class_map_lines = []
if len(unique_labels.keys()) < 2:
    lines = ["Number of labels is less than 2, so nothing to predict!"]
else:
    separator = ","
    labels_string = separator.join([str(x) for x in labels])
    unique_labels_string = separator.join([str(x) for x in unique_labels.keys()])
    if log_file_path is not None:
        eprint("Labels " + str(len(labels)) + ": " + labels_string)
        eprint("Unique Labels " + str(len(unique_labels.keys())) + ": " + unique_labels_string)
    else:
        print("Labels " + str(len(labels)) + ": " + labels_string)
        print("Unique Labels " + str(len(unique_labels.keys())) + ": " + unique_labels_string)

    categorical_object = pd.cut(labels, 25)
    labels_predict_codes = categorical_object.codes
    categories = categorical_object.categories

    labels_predict_map = {}
    labels_predict_unique = {}
    for index in range(len(labels)):
        label = labels[index]
        label_code = labels_predict_codes[index]
        cat_mid = categories[label_code].mid
        labels_predict_map[str(label_code)] = cat_mid
        if str(label_code) in labels_predict_unique.keys():
            labels_predict_unique[str(label_code)] += 1
        else:
            labels_predict_unique[str(label_code)] = 1

    #labels_predict = preprocessing.normalize([labels_predict], norm='l2')
    #labels_predict = labels_predict[0]
    labels_predict = labels_predict_codes.astype(str)
    lb = LabelBinarizer()
    labels_lb = lb.fit_transform(labels_predict)

    separator = ","
    lb_classes_string = separator.join([str(x) for x in lb.classes_])
    if log_file_path is not None:
        eprint("Classes " + str(len(lb.classes_)) + ": " + lb_classes_string)
    else:
        print("Classes " + str(len(lb.classes_)) + ": " + lb_classes_string)

    separator = ", "
    lines.append("Predicted Labels: " + separator.join(lb.classes_))

    print("[INFO] number of labels: %d" % (len(labels_lb)))
    print("[INFO] number of images: %d" % (len(data)))

    print("[INFO] splitting training set...")
    (trainX, testX, trainY, testY) = train_test_split(np.array(data), np.array(labels_lb), test_size=0.2)

    img_input = Input(shape=(75,75,3))

    #STEM
    x = conv2d(img_input,32,3,2,'valid',True,name='conv1')
    x = conv2d(x,32,3,1,'valid',True,name='conv2')
    x = conv2d(x,64,3,1,'valid',True,name='conv3')

    x_11 = MaxPooling2D(3,strides=1,padding='valid',name='stem_br_11'+'_maxpool_1', data_format="channels_last")(x)
    x_12 = conv2d(x,64,3,1,'valid',True,name='stem_br_12')

    x = Concatenate(axis=3, name = 'stem_concat_1')([x_11,x_12])

    x_21 = conv2d(x,64,1,1,'same',True,name='stem_br_211')
    x_21 = conv2d(x_21,64,[1,7],1,'same',True,name='stem_br_212')
    x_21 = conv2d(x_21,64,[7,1],1,'same',True,name='stem_br_213')
    x_21 = conv2d(x_21,96,3,1,'valid',True,name='stem_br_214')

    x_22 = conv2d(x,64,1,1,'same',True,name='stem_br_221')
    x_22 = conv2d(x_22,96,3,1,'valid',True,name='stem_br_222')

    x = Concatenate(axis=3, name = 'stem_concat_2')([x_21,x_22])

    x_31 = conv2d(x,192,3,1,'valid',True,name='stem_br_31')
    x_32 = MaxPooling2D(3,strides=1,padding='valid',name='stem_br_32'+'_maxpool_2', data_format="channels_last")(x)
    x = Concatenate(axis=3, name = 'stem_concat_3')([x_31,x_32])

    #Inception-ResNet-A modules
    x = incresA(x,0.15,name='incresA_1')
    x = incresA(x,0.15,name='incresA_2')
    x = incresA(x,0.15,name='incresA_3')
    x = incresA(x,0.15,name='incresA_4')

    #35 × 35 to 17 × 17 reduction module.
    x_red_11 = MaxPooling2D(3,strides=2,padding='valid',name='red_maxpool_1', data_format="channels_last")(x)

    x_red_12 = conv2d(x,384,3,2,'valid',True,name='x_red1_c1')

    x_red_13 = conv2d(x,256,1,1,'same',True,name='x_red1_c2_1')
    x_red_13 = conv2d(x_red_13,256,3,1,'same',True,name='x_red1_c2_2')
    x_red_13 = conv2d(x_red_13,384,3,2,'valid',True,name='x_red1_c2_3')

    x = Concatenate(axis=3, name='red_concat_1')([x_red_11,x_red_12,x_red_13])

    #Inception-ResNet-B modules
    x = incresB(x,0.1,name='incresB_1')
    x = incresB(x,0.1,name='incresB_2')
    x = incresB(x,0.1,name='incresB_3')
    x = incresB(x,0.1,name='incresB_4')
    x = incresB(x,0.1,name='incresB_5')
    x = incresB(x,0.1,name='incresB_6')
    x = incresB(x,0.1,name='incresB_7')

    #17 × 17 to 8 × 8 reduction module.
    x_red_21 = MaxPooling2D(3,strides=2,padding='valid',name='red_maxpool_2', data_format="channels_last")(x)

    x_red_22 = conv2d(x,256,1,1,'same',True,name='x_red2_c11')
    x_red_22 = conv2d(x_red_22,384,3,2,'valid',True,name='x_red2_c12')

    x_red_23 = conv2d(x,256,1,1,'same',True,name='x_red2_c21')
    x_red_23 = conv2d(x_red_23,256,3,2,'valid',True,name='x_red2_c22')

    x_red_24 = conv2d(x,256,1,1,'same',True,name='x_red2_c31')
    x_red_24 = conv2d(x_red_24,256,3,1,'same',True,name='x_red2_c32')
    x_red_24 = conv2d(x_red_24,256,3,2,'valid',True,name='x_red2_c33')

    x = Concatenate(axis=3, name='red_concat_2')([x_red_21,x_red_22,x_red_23,x_red_24])

    #Inception-ResNet-C modules
    x = incresC(x,0.2,name='incresC_1')
    x = incresC(x,0.2,name='incresC_2')
    x = incresC(x,0.2,name='incresC_3')

    #TOP
    x = GlobalAveragePooling2D(data_format='channels_last')(x)
    x = Dropout(0.6)(x)
    x = Dense(len(lb.classes_), activation='softmax')(x)

    model = Model(img_input,x,name='inception_resnet_v2')

    # init = "he_normal"
    # reg = regularizers.l2(0.01)
    # chanDim = -1
    #
    # model = Sequential()
    # model.add(Conv2D(16, (7, 7), strides=(2, 2), padding="valid", kernel_initializer=init, kernel_regularizer=reg, input_shape=(75, 75, 3)))
    # model.add(Conv2D(32, (3, 3), padding="same", kernel_initializer=init, kernel_regularizer=reg))
    # model.add(Activation("relu"))
    # model.add(BatchNormalization(axis=chanDim))
    # model.add(Conv2D(32, (3, 3), strides=(2, 2), padding="same", kernel_initializer=init, kernel_regularizer=reg))
    # model.add(Activation("relu"))
    # model.add(BatchNormalization(axis=chanDim))
    # model.add(Dropout(0.25))
    #
    # # stack two more CONV layers, keeping the size of each filter
    # # as 3x3 but increasing to 64 total learned filters
    # model.add(Conv2D(64, (3, 3), padding="same", kernel_initializer=init, kernel_regularizer=reg))
    # model.add(Activation("relu"))
    # model.add(BatchNormalization(axis=chanDim))
    # model.add(Conv2D(64, (3, 3), strides=(2, 2), padding="same", kernel_initializer=init, kernel_regularizer=reg))
    # model.add(Activation("relu"))
    # model.add(BatchNormalization(axis=chanDim))
    # model.add(Dropout(0.25))
    #
    # # increase the number of filters again, this time to 128
    # model.add(Conv2D(128, (3, 3), padding="same", kernel_initializer=init, kernel_regularizer=reg))
    # model.add(Activation("relu"))
    # model.add(BatchNormalization(axis=chanDim))
    # model.add(Conv2D(128, (3, 3), strides=(2, 2), padding="same", kernel_initializer=init, kernel_regularizer=reg))
    # model.add(Activation("relu"))
    # model.add(BatchNormalization(axis=chanDim))
    # model.add(Dropout(0.25))
    #
    # # fully-connected layer
    # model.add(Flatten())
    # model.add(Dense(512, kernel_initializer=init))
    # model.add(Activation("relu"))
    # model.add(BatchNormalization())
    # model.add(Dropout(0.5))
    #
    # # softmax classifier
    # model.add(Dense(len(lb.classes_)))
    # model.add(Activation("softmax"))

    for layer in model.layers:
        print(layer.output_shape)

    print("[INFO] training network...")
    opt = Adam(lr=1e-3, decay=1e-3 / 50)
    model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

    checkpoint = ModelCheckpoint(output_model_file_path, monitor='acc', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint]

    H = model.fit(trainX, trainY, validation_data=(testX, testY), epochs=150, batch_size=32, callbacks=callbacks_list)

    # print("[INFO] evaluating network...")
    # predictions = model.predict(testX, batch_size=32)
    # report = classification_report(testY.argmax(axis=1), predictions.argmax(axis=1), target_names=lb.classes_)
    # print(report)
    # 
    # report_lines = report.split('\n')
    # separator = ""
    # for l in report_lines:
    #     lines.append(separator.join(l))

    iterator = 0
    for c in lb.classes_:
        class_map_lines.append([iterator, labels_predict_map[str(c)], labels_predict_unique[str(c)]])
        iterator += 1

#print(lines)
with open(outfile_path, 'w') as writeFile:
    writer = csv.writer(writeFile)
    writer.writerows(lines)
writeFile.close()

#print(class_map_lines)
with open(output_class_map, 'w') as writeFile:
    writer = csv.writer(writeFile)
    writer.writerows(class_map_lines)
writeFile.close()
