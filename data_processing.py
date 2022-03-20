import tensorflow as tf
from configuration import *
from utils import ReflectionPadding2D
import tensorflow_datasets as tfds

def transform_data(x, mean, sd, padding, padding_type, random_crop = None):
    x = tf.cast(x, tf.float64)
    x = x.numpy()
    for d in range(x.shape[3]):
        x[:, :, :, d:(d + 1)] = x[:, :, :, d:(d + 1)]/255.0
        x[:,:,:,d:(d+1)] = ((x[:,:,:,d:(d+1)] - mean[d])/sd[d])
    x = tf.convert_to_tensor(x)
    if padding_type == 'zero':
        pad_layer = tf.keras.layers.ZeroPadding2D(padding = padding)
    elif padding_type == 'reflection':
        pad_layer = ReflectionPadding2D(padding = (padding, padding))
    x = pad_layer(x)

    if random_crop != None:
        random_crop_layer = tf.keras.layers.RandomCrop(height=random_crop[0], width=random_crop[1])
        x = random_crop_layer(x)
    return x

def get_data(split, data_set, extract_number = None, one_hot = False, shuffle = False):
    if data_set == 'mnist':
        mnist = tf.keras.datasets.mnist
        (x_train, y_train), (x_test, y_test) = mnist.load_data()

        x_train = tf.expand_dims(x_train, axis=3)
        x_test = tf.expand_dims(x_test, axis=3)

        x_train = transform_data(x_train, mean = mean, sd = sd, padding = padding, padding_type = padding_type, random_crop = random_crop)
        x_test = transform_data(x_test, mean=mean, sd=sd, padding=padding, padding_type=padding_type,
                                 random_crop=random_crop)

        if split == 'train':
            if extract_number == None:
                extract_number = x_train.shape[0]
            if shuffle:
                train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(x_train.shape[0]).batch(extract_number)
            else:
                train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(
                    extract_number)
            dat, = train_ds.take(1)
            example_x = dat[0]
            example_x = tf.reshape(example_x, (extract_number, 784))
        elif split == 'test':
            if extract_number == None:
                extract_number = x_test.shape[0]
            if shuffle:
                test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).shuffle(x_test.shape[0]).batch(
                    extract_number)
            else:
                test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(
                    extract_number)
            dat, = test_ds.take(1)
            example_x = dat[0]
            example_x = tf.reshape(example_x, (extract_number, 784))
    elif data_set == 'cifar10':
        train_ds, info_train = tfds.load('cifar10', split='train', with_info=True)
        test_ds, info_test = tfds.load('cifar10', split='test', with_info=True)

        train_size = len(list(train_ds))
        test_size = len(list(test_ds))

        ds_train = train_ds.batch(train_size)
        example, = ds_train.take(1)
        x_train, y_train = example['image'], example['label']

        ds_test = test_ds.batch(test_size)
        example, = ds_test.take(1)
        x_test, y_test = example['image'], example['label']

        #x_train = tf.expand_dims(x_train, axis=3)
        #x_test = tf.expand_dims(x_test, axis=3)

        x_train = transform_data(x_train, mean = mean, sd = sd, padding = padding, padding_type = padding_type, random_crop = random_crop)
        x_test = transform_data(x_test, mean=mean, sd=sd, padding=padding, padding_type=padding_type,
                                 random_crop=random_crop)

        if split == 'train':
            if extract_number == None:
                extract_number = x_train.shape[0]
            if shuffle:
                train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(x_train.shape[0]).batch(extract_number)
            else:
                train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(
                    extract_number)
            dat, = train_ds.take(1)
            example_x = dat[0]
            example_x = tf.reshape(example_x, (extract_number, 3072))
        elif split == 'test':
            if extract_number == None:
                extract_number = x_test.shape[0]
            if shuffle:
                test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).shuffle(x_test.shape[0]).batch(
                    extract_number)
            else:
                test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(
                    extract_number)
            dat, = test_ds.take(1)
            example_x = dat[0]
            example_x = tf.reshape(example_x, (extract_number, 3072))

    '''
    elif data_set == 'cifar10':
        if split == 'train':
            ds_train = tfds.load(name="cifar10", split="train")
            ds_train = ds_train.batch(batch_size)
            example, = ds_train.take(1)
            example_x = example['image']
            example_x = tf.reshape(example_x, (batch_size, 3072))
        elif split == 'test':
            ds_train = tfds.load(name="cifar10", split="test")
            ds_train = ds_train.batch(batch_size)
            example, = ds_train.take(1)
            example_x = example['image']
            example_x = tf.reshape(example_x, (batch_size, 3072))
    '''
    if one_hot:
        return example_x, tf.one_hot(dat[1], depth = 10)
    else:
        return example_x, dat[1]


def get_data_loader(split, data_set, batch_size = 128, one_hot = False, shuffle = False, number_sample = None):
    if data_set == 'mnist':
        mnist = tf.keras.datasets.mnist
        (x_train, y_train), (x_test, y_test) = mnist.load_data()

        x_train = tf.expand_dims(x_train, axis=3)
        x_test = tf.expand_dims(x_test, axis=3)
        x_train = transform_data(x_train, mean=mean, sd=sd, padding=padding, padding_type=padding_type,
                                 random_crop=random_crop)
        x_test = transform_data(x_test, mean=mean, sd=sd, padding=padding, padding_type=padding_type,
                                random_crop=random_crop)

        if one_hot:
            y_train = tf.one_hot(y_train, depth = 10)
            y_test = tf.one_hot(y_test, depth = 10)

        if split == 'train':
            if number_sample is not None:
                number_sample = min(x_train.shape[0], number_sample)
            if shuffle:
                train_ds = tf.data.Dataset.from_tensor_slices((x_train[0:number_sample], y_train[0:number_sample])).shuffle(x_train.shape[0]).batch(batch_size)
            else:
                train_ds = tf.data.Dataset.from_tensor_slices((x_train[0:number_sample], y_train[0:number_sample])).batch(
                    batch_size)
            return train_ds
        elif split == 'test':
            if number_sample is not None:
                number_sample = min(x_test.shape[0], number_sample)
            if shuffle:
                test_ds = tf.data.Dataset.from_tensor_slices((x_test[0:number_sample], y_test[0:number_sample])).shuffle(x_test.shape[0]).batch(
                    batch_size)
            else:
                test_ds = tf.data.Dataset.from_tensor_slices((x_test[0:number_sample], y_test[0:number_sample])).batch(
                    batch_size)
            return test_ds
    elif data_set == 'cifar10':
        train_ds, info_train = tfds.load('cifar10', split='train', with_info = True)
        test_ds, info_test = tfds.load('cifar10', split='test', with_info=True)

        train_size = len(list(train_ds))
        test_size = len(list(test_ds))

        ds_train = train_ds.batch(train_size)
        example, = ds_train.take(1)
        x_train, y_train = example['image'], example['label']

        ds_test = test_ds.batch(test_size)
        example, = ds_test.take(1)
        x_test, y_test = example['image'], example['label']

        #(x_train, y_train), (x_test, y_test) = cifar10.load_data()
        #x_train, y_train = tuple(zip(*train_ds))
        #x_test, y_test = tuple(zip(*test_ds))
        #if len(x_train.shape) < 4:
        #    x_train = tf.expand_dims(x_train, axis=3)
        #    x_test = tf.expand_dims(x_test, axis=3)
        x_train = transform_data(x_train, mean=mean, sd=sd, padding=padding, padding_type=padding_type,
                                 random_crop=random_crop)
        x_test = transform_data(x_test, mean=mean, sd=sd, padding=padding, padding_type=padding_type,
                                random_crop=random_crop)

        if one_hot:
            y_train = tf.one_hot(y_train, depth = 10)
            y_test = tf.one_hot(y_test, depth = 10)

        if split == 'train':
            if number_sample is not None:
                number_sample = min(x_train.shape[0], number_sample)
            if shuffle:
                train_ds = tf.data.Dataset.from_tensor_slices((x_train[0:number_sample], y_train[0:number_sample])).shuffle(x_train.shape[0]).batch(batch_size)
            else:
                train_ds = tf.data.Dataset.from_tensor_slices((x_train[0:number_sample], y_train[0:number_sample])).batch(
                    batch_size)
            return train_ds
        elif split == 'test':
            if number_sample is not None:
                number_sample = min(x_test.shape[0], number_sample)
            if shuffle:
                test_ds = tf.data.Dataset.from_tensor_slices((x_test[0:number_sample], y_test[0:number_sample])).shuffle(x_test.shape[0]).batch(
                    batch_size)
            else:
                test_ds = tf.data.Dataset.from_tensor_slices((x_test[0:number_sample], y_test[0:number_sample])).batch(
                    batch_size)
            return test_ds


