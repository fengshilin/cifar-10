
class PATH:
    csv_path = '/home/dennis/competition/classify/cifar-10/dataset/trainLabels.csv'
    train_root = '/home/dennis/competition/classify/cifar-10/dataset/train'
    weight_path = '/home/dennis/competition/classify/cifar-10/weight_path/cifar_vgg16_not_trained_20191216.ckpt'
    test_root = '/home/dennis/competition/classify/cifar-10/dataset/test'
    result_csv = '/home/dennis/competition/classify/cifar-10/test.csv'

class CONFIG:
    image_shape = (32,32)
    train_per = 0.999  # 训练集的占比
    n_class = 10
    lr = 1e-3
    decay_lr = 1e-4
    num_epochs = 100
    batch_size = 200
    label_class = ["airplane",  "automobile", "bird" ,"cat" ,"deer", "dog" ,"frog" ,"horse" ,"ship", "truck"]
