import tensorflow as tf
import pandas as pd
import os
import cv2
import random

import sys
sys.path.append('/home/dennis/competition/classify/cifar-10')
from config import PATH, CONFIG



def load_csv(csv_path):
    """读取csv文件"""
    csv_data = pd.read_csv(csv_path)
    id_list = list(csv_data['id'])
    id_list = map(lambda x:str(x), id_list)
    label_list = list(csv_data['label'])

    return list(zip(id_list, label_list))


def split_dataset(data_list):
    """将数据集均匀地分为训练集与验证集"""
    random.seed(3)
    train_list = random.sample(data_list, int(
        len(data_list)*CONFIG.train_per))
    val_list = list(set(data_list) - set(train_list))

    return train_list, val_list


def unzip_data(data_list):
    """将[(a1,b1), (a2, b3)]分解为[a1,a2], [b1,b2]"""
    id_list = [data[0] for data in data_list]
    label_list  = [data[1] for data in data_list]

    return tuple((id_list, label_list))

def load_image(image_id, label):
    
    image = cv2.imread(os.path.join(PATH.train_root, str(image_id.numpy(), 'utf-8')+".png"))
    # print(os.path.join(PATH.train_root, str(image_id.numpy(), 'utf-8')+".png"))
    image = image/255

    label_class = CONFIG.label_class
    label = label_class.index(label)

    return tf.convert_to_tensor(image, dtype=tf.float32), tf.convert_to_tensor(label, dtype=tf.int32)

def set_shape(image, label):
    image.set_shape([CONFIG.image_shape[0], CONFIG.image_shape[1], 3])
    label.set_shape([])
    return image, label


if __name__ == "__main__":
    result = load_csv('/home/dennis/competition/classify/cifar-10/dataset/trainLabels.csv')
    train_list, val_list = split_dataset(result)
    train_data = unzip_data(train_list)

    dataset = tf.data.Dataset.from_tensor_slices(train_data)
    dataset = dataset.shuffle(len(result[0])).map(load_image)
    index = 0
    for id, label in dataset:
        if index <=10:
            print(id, label)
        index+=1
