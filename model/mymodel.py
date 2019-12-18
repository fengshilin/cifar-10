import tensorflow as tf
from tensorflow.keras.layers import Dropout, Input, concatenate, GlobalAveragePooling2D, Dense, BatchNormalization, Activation

import sys
sys.path.append('/home/dennis/competition/classify/cifar-10')
from config import PATH, CONFIG


image_shape = CONFIG.image_shape

class MyModel(tf.keras.Model):
    def __init__(self, n_class=2):
        super().__init__()
        self.n_class = n_class
        self.vgg16_model = self.load_vgg16()
        self.res50_model = self.load_res50()
        self.global_pool = GlobalAveragePooling2D()
        self.conv_vgg = Dense(512/4, use_bias=False,
                              kernel_initializer='uniform')
        self.conv_res = Dense(2048/4, use_bias=False,
                              kernel_initializer='uniform')
        self.batch_normalize = BatchNormalization()
        self.batch_normalize_res = BatchNormalization()
        self.relu = Activation("relu")
        self.concat = concatenate
        self.dropout_1 = Dropout(0.3)
        self.conv_1 = Dense(640, use_bias=False, kernel_initializer='uniform')
        self.batch_normalize_1 = BatchNormalization()
        self.relu_1 = Activation("relu")
        self.dropout_2 = Dropout(0.5)
        self.classify = Dense(
            n_class, kernel_initializer='uniform', activation="softmax")

    def call(self, input):
      x_vgg16 = self.vgg16_model(input)
      x_vgg16 = self.global_pool(x_vgg16)
      x_vgg16 = self.conv_vgg(x_vgg16)
      x_vgg16 = self.batch_normalize(x_vgg16)
      x_vgg16 = self.relu(x_vgg16)
      x_res50 = self.res50_model(input)
      x_res50 = self.global_pool(x_res50)
      x_res50 = self.conv_res(x_res50)
      x_res50 = self.batch_normalize_res(x_res50)
      x_res50 = self.relu(x_res50)
      x = self.concat([x_vgg16, x_res50])
      x = self.dropout_1(x)
      x = self.conv_1(x)
      x = self.batch_normalize_1(x)
      x = self.relu_1(x)
      x = self.dropout_2(x)
      x = self.classify(x)

      return x
    
    def load_vgg16(self):
        vgg16 = tf.keras.applications.vgg16.VGG16(weights='imagenet', include_top=False, input_tensor=Input(
            shape=(image_shape[0], image_shape[1], 3)), classes=self.n_class)
        vgg16.trainable = False

        return vgg16
    
    def load_res50(self):
        res50 = tf.keras.applications.resnet50.ResNet50(weights='imagenet', include_top=False, input_tensor=Input(
            shape=(image_shape[0], image_shape[1], 3)), classes=self.n_class)
        res50.trainable = False

        return res50

# model = MyModel()


class Vgg16net(tf.keras.Model):
    def __init__(self, n_class=2):
        super().__init__()
        self.n_class = n_class
        self.vgg16_model = self.load_vgg16()
        self.global_pool = GlobalAveragePooling2D()
        self.conv_vgg = Dense(256, use_bias=False,
                              kernel_initializer='uniform')
        self.batch_normalize = BatchNormalization()
        self.relu = Activation("relu")
        self.dropout_1 = Dropout(0.2)
        self.conv_1 = Dense(64, use_bias=False, kernel_initializer='uniform')
        self.batch_normalize_1 = BatchNormalization()
        self.relu_1 = Activation("relu")
        self.dropout_2 = Dropout(0.2)
        self.classify = Dense(
            n_class, kernel_initializer='uniform', activation="softmax")

    def call(self, input):
      x = self.vgg16_model(input)
      x = self.global_pool(x)
      x = self.conv_vgg(x)
      x = self.batch_normalize(x)
      x = self.relu(x)
      x = self.dropout_1(x)
      x = self.conv_1(x)
      x = self.batch_normalize_1(x)
      x = self.relu_1(x)
      x = self.dropout_2(x)
      x = self.classify(x)

      return x
    
    def load_vgg16(self):
        vgg16 = tf.keras.applications.vgg16.VGG16(weights='imagenet', include_top=False, input_tensor=Input(
            shape=(image_shape[0], image_shape[1], 3)), classes=self.n_class)

        # for layer in vgg16.layers[:15]:
        #     layer.trainable = False

        return vgg16
    
    def load_res50(self):
        res50 = tf.keras.applications.resnet50.ResNet50(weights='imagenet', include_top=False, input_tensor=Input(
            shape=(image_shape[0], image_shape[1], 3)), classes=self.n_class)
        res50.trainable = False
        for layer in res50.layers[:15]:
            layer.trainable = False

        return res50