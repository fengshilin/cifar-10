import numpy as np
import csv
import cv2
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint

from model.mymodel import MyModel, Vgg16net
from dataset.dataloader import load_csv, load_image, split_dataset, unzip_data, set_shape
import os
import sys
sys.path.append('/home/dennis/competition/classify/cifar-10')
from config import PATH, CONFIG


model = Vgg16net(CONFIG.n_class)

# 加载之前训练过的模型可以加快收敛速度
model.load_weights(PATH.weight_path)


def load_data(test_root):
    file_list = os.listdir(test_root)
    # 排序
    file_list.sort()
    row_list = []
    with tqdm(total=100) as pbar:
        for i, file_name in enumerate(file_list):
            
            image = cv2.imread(os.path.join(test_root, file_name))
            image = image[np.newaxis,:,:,:]
            image = tf.convert_to_tensor(image/255, dtype=tf.float32)
            pred = model.predict(image)

            # 将预测结果还原为label
            label_class = CONFIG.label_class
            index = np.argmax(pred)
            result = label_class[index]

            row_list.append([file_name.split('.')[0], result])
            if (100*i / 300000) % 1 ==0:
                pbar.update(1)
    print("全部处理完成")
    return row_list


if __name__ == "__main__":
    
    with open(PATH.result_csv,"w") as csvfile: 
        writer = csv.writer(csvfile)
        #先写入columns_name
        writer.writerow(["id","label"])

        row_list = load_data(PATH.test_root)
        row_list.sort(key=lambda x:int(x[0]))
        #写入多行用writerows
        writer.writerows(row_list)
