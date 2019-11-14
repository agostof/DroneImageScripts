# import the necessary packages
import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split

class CNNProcessData:
    def __init__(self):
        pass

    def get_imagedatagenerator(self):
        datagen = ImageDataGenerator(
            featurewise_center=True,
            featurewise_std_normalization=True,
            #rotation_range=20,
            width_shift_range=0.05,
            height_shift_range=0.05,
            horizontal_flip=True,
            # vertical_flip=True,
            brightness_range=[0.8,1.2]
        )
        return datagen

    def generate_croppings(self, testX, testY, image_size, number):
        if number != 11:
            raise Exception("Only implemented for number = 11 right now")

        augmented_testX_1 = []
        augmented_testX_2 = []
        augmented_testX_3 = []
        augmented_testX_4 = []
        augmented_testX_5 = []
        augmented_testX_6 = []
        augmented_testX_7 = []
        augmented_testX_8 = []
        augmented_testX_9 = []
        augmented_testX_10 = []
        augmented_testX_11 = []
        mid_image_size = int(round(image_size/2))
        for img in testX:
            height = img.shape[0]
            small_height = int(round(height*0.1))
            mid_height = int(round(height/2))
            width = img.shape[1]
            mid_width = int(round(width/2))
            crop_img1 = img[height-image_size:height, 0:image_size]
            crop_img2 = img[height-image_size:height, width-image_size:width]
            crop_img3 = img[0:image_size, width-image_size:width]
            crop_img4 = img[0:image_size, 0:image_size]
            crop_img5 = img[mid_height-mid_image_size:mid_height+mid_image_size, mid_width-mid_image_size:mid_width+mid_image_size]
            crop_img6 = img[mid_height-mid_image_size:mid_height+mid_image_size, 0:image_size]
            crop_img7 = img[mid_height-mid_image_size:mid_height+mid_image_size, width-image_size:width]
            crop_img8 = img[mid_height+small_height-mid_image_size:mid_height+small_height+mid_image_size, 0:image_size]
            crop_img9 = img[mid_height+small_height-mid_image_size:mid_height+small_height+mid_image_size, width-image_size:width]
            crop_img10 = img[mid_height-small_height-mid_image_size:mid_height-small_height+mid_image_size, 0:image_size]
            crop_img11 = img[mid_height-small_height-mid_image_size:mid_height-small_height+mid_image_size, width-image_size:width]
            augmented_testX_1.append(crop_img1)
            augmented_testX_2.append(crop_img2)
            augmented_testX_3.append(crop_img3)
            augmented_testX_4.append(crop_img4)
            augmented_testX_5.append(crop_img5)
            augmented_testX_6.append(crop_img6)
            augmented_testX_7.append(crop_img7)
            augmented_testX_8.append(crop_img8)
            augmented_testX_9.append(crop_img9)
            augmented_testX_10.append(crop_img10)
            augmented_testX_11.append(crop_img11)

        augmented_testX_1 = np.array(augmented_testX_1)
        augmented_testX_2 = np.array(augmented_testX_2)
        augmented_testX_3 = np.array(augmented_testX_3)
        augmented_testX_4 = np.array(augmented_testX_4)
        augmented_testX_5 = np.array(augmented_testX_5)
        augmented_testX_6 = np.array(augmented_testX_6)
        augmented_testX_7 = np.array(augmented_testX_7)
        augmented_testX_8 = np.array(augmented_testX_8)
        augmented_testX_9 = np.array(augmented_testX_9)
        augmented_testX_10 = np.array(augmented_testX_10)
        augmented_testX_11 = np.array(augmented_testX_11)
        testX = np.concatenate((augmented_testX_1, augmented_testX_2, augmented_testX_3, augmented_testX_4, augmented_testX_5, augmented_testX_6, augmented_testX_7, augmented_testX_8, augmented_testX_9, augmented_testX_10, augmented_testX_11))
        # testXflipped = []
        # for img in testX:
        #     horizontal_flip = cv2.flip( img, 0 )
        #     testXflipped.append(horizontal_flip)
        # testXflipped = np.array(testXflipped)
        # testX = np.concatenate((testX, testXflipped))
        testY = np.repeat(testY, number)
        return (testX, testY)

    def process_cnn_data(self, data, labels_lb, num_unique_stock_ids, num_unique_image_types, num_unique_time_days, input_image_size, image_size, number_labels, keras_model_type, data_augmentation, data_augmentation_test):
        trainX = []
        testX = []
        trainY = []
        testY = []

        datagen = self.get_imagedatagenerator()

        datagen.fit(data)

        # LSTM models group images by time, but are still ties to a single label e.g. X, Y = [img_t1, img_t2, img_t3], y1.
        if keras_model_type == 'densenet121_lstm_imagenet':
            print(len(data))
            print(len(labels_lb))
            print(data.shape)
            print(labels_lb.shape)
            data = data.reshape(num_unique_stock_ids * num_unique_image_types, num_unique_time_days, input_image_size, input_image_size, 3)
            labels_lb = labels_lb.reshape(num_unique_stock_ids * num_unique_image_types, num_unique_time_days, number_labels)
            print(len(data))
            print(len(labels_lb))
            print(data.shape)
            print(labels_lb.shape)

            (trainX, testX, trainY, testY) = train_test_split(data, labels_lb, test_size=0.2)
            trainX_length = len(trainX) 
            trainY_length = len(trainY)
            print(trainX_length)
            print(trainY_length)
            print(trainX.shape)
            print(trainY.shape)
            testX_length = len(testX)
            testY_length = len(testY)
            print(testX_length)
            print(testY_length)
            print(testX.shape)
            print(testY.shape)
            trainX = trainX.reshape(trainX_length * num_unique_time_days, input_image_size, input_image_size, 3)
            trainY = trainY.reshape(trainY_length * num_unique_time_days, number_labels)
            testX = testX.reshape(testX_length * num_unique_time_days, input_image_size, input_image_size, 3)
            testY = testY.reshape(testY_length * num_unique_time_days, number_labels)
            trainX_length_flat = len(trainX)
            trainY_length_flat = len(trainY)
            print(len(trainX))
            print(len(trainY))
            print(trainX.shape)
            print(trainY.shape)

            testX = datagen.standardize(testX)

            (testX, testY) = self.generate_croppings(testX, testY, image_size, data_augmentation_test)

            testX = testX.reshape(data_augmentation_test * testX_length, num_unique_time_days, image_size, image_size, 3)
            testY = testY.reshape(data_augmentation_test * testY_length, num_unique_time_days, number_labels)

            labels = []
            for l in testY:
                labels.append(l[0])
            testY = np.array(labels)

            print(data_augmentation)
            augmented = datagen.flow(trainX, trainY, batch_size=trainX_length_flat)
            for i in range(0, data_augmentation-1):
                X, y = augmented.next()
                trainX = np.concatenate((trainX, X))
                trainY = np.concatenate((trainY, y))

            print(len(trainX))
            print(len(trainY))
            trainX_resized = []
            for img in trainX:
                trainX_resized.append(cv2.resize(img, (image_size, image_size)))
            trainX = np.array(trainX_resized)
            print(len(trainX))
            print(len(trainY))

            trainX = trainX.reshape(data_augmentation * trainX_length, num_unique_time_days, image_size, image_size, 3)
            trainY = trainY.reshape(data_augmentation * trainY_length, num_unique_time_days, number_labels)

            labels = []
            for l in trainY:
                labels.append(l[0])
            trainY = np.array(labels)
        else:
            (trainX, testX, trainY, testY) = train_test_split(data, labels_lb, test_size=0.2)
            testY_length = len(testY)
            
            testX = datagen.standardize(testX)
            
            (testX, testY) = self.generate_croppings(testX, testY, image_size, data_augmentation_test)
            
            testY = testY.reshape(data_augmentation_test * testY_length, number_labels)

            augmented = datagen.flow(trainX, trainY, batch_size=len(trainX))
            for i in range(0, data_augmentation-1):
                X, y = augmented.next()
                trainX = np.concatenate((trainX, X))
                trainY = np.concatenate((trainY, y))

            trainX_resized = []
            for img in trainX:
                trainX_resized.append(cv2.resize(img, (image_size, image_size)))
            trainX = np.array(trainX_resized)
        
        return (testX, testY, trainX, trainY)

    def process_cnn_data_predictions(self, data, num_unique_stock_ids, num_unique_image_types, num_unique_time_days, input_image_size, image_size, keras_model_type, training_data, data_augmentation_test):
        trainX = []
        testX = []
        trainY = []
        testY = []

        datagen = self.get_imagedatagenerator()

        datagen.fit(training_data)
        data = datagen.standardize(data)

        augmented_data = []
        ret = self.generate_croppings(data, None, image_size, data_augmentation_test)
        augmented_data = ret[0]

        # LSTM models group images by time, but are still ties to a single label e.g. X, Y = [img_t1, img_t2, img_t3], y1.
        if keras_model_type == 'KerasCNNLSTMDenseNet121ImageNetWeights':
            augmented_data = augmented_data.reshape(data_augmentation_test * num_unique_stock_ids * num_unique_image_types, num_unique_time_days, image_size, image_size, 3)
            
        return augmented_data
