# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.



from multiprocessing import Pool
import tensorflow_datasets as tfds
import tensorflow as tf


if __name__ == '__main__':
    ds_train = tfds.load(name="cifar10", split="train")
    ds_train = ds_train.batch(1)
    example, = ds_train.take(1)
    example_x = example['image']
    example_x = tf.reshape(example_x, (1, 3072))

    #print(example_x)
    #print(tf.one_hot(example['label'], depth=10))

