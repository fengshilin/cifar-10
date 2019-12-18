import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint

from model.mymodel import MyModel, Vgg16net
from dataset.dataloader import load_csv, load_image, split_dataset, unzip_data, set_shape

import sys
sys.path.append('/home/dennis/competition/classify/cifar-10')
from config import PATH, CONFIG


# 读取csv
csv_data = load_csv(PATH.csv_path)
train_list, val_list = split_dataset(csv_data)
train_list = unzip_data(train_list)
train_dataset = tf.data.Dataset.from_tensor_slices(train_list)
tf.random.set_seed(1)
train_dataset = train_dataset.shuffle(len(train_list)).map(lambda x,y: tf.py_function(load_image, [x,y], [tf.float32, tf.int32])).map(set_shape).map(
    lambda image, label: (tf.image.convert_image_dtype(image, tf.float32), label)
).cache().map(
    lambda image, label: (tf.image.random_flip_left_right(image), label)
).map(
    lambda image, label: (tf.image.random_contrast(image, lower=0.0, upper=1.0), label)
).batch(CONFIG.batch_size)

val_list = unzip_data(val_list)
val_dataset = tf.data.Dataset.from_tensor_slices(val_list)
tf.random.set_seed(2)
val_dataset = val_dataset.shuffle(len(val_list)).map(lambda x,y: tf.py_function(load_image, [x,y], [tf.float32, tf.int32])).map(set_shape).batch(CONFIG.batch_size)

model = Vgg16net(CONFIG.n_class)

# 加载之前训练过的模型可以加快收敛速度
model.load_weights(PATH.weight_path)

checkpoint_callback = ModelCheckpoint(
    PATH.weight_path, monitor='val_accuracy', verbose=1,
    save_best_only=False, save_weights_only=True,
    save_frequency=1)

optimizer = tf.keras.optimizers.SGD(
    learning_rate=CONFIG.lr, decay=CONFIG.decay_lr)

model.compile(
    optimizer=optimizer,
    loss=tf.keras.losses.sparse_categorical_crossentropy,
    metrics=[tf.metrics.SparseCategoricalAccuracy()]
)
model.fit(train_dataset, validation_data=val_dataset,
          epochs=CONFIG.num_epochs, callbacks=[checkpoint_callback])

# model.save(weight_path+'dog_breeds', save_format="tf")




