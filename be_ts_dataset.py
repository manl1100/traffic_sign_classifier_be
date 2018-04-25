import os

import numpy as np
from skimage import transform
from skimage.color import rgb2gray
import skimage.data
import matplotlib.pyplot as plt

TRAIN_DATA_URL = 'http://btsd.ethz.ch/shareddata/BelgiumTSC/BelgiumTSC_Training.zip'
TEST_DATA_URL = 'http://btsd.ethz.ch/shareddata/BelgiumTSC/BelgiumTSC_Testing.zip'

ROOT_PATH = '/Users/manuelsanchez/Repositories/traffic_sign_classifier_be/'

TRAIN_DATA_PATH = os.path.join(ROOT_PATH, 'traffic_signs/Training')
TEST_DATA_PATH = os.path.join(ROOT_PATH, 'traffic_signs/Testing')


def load_raw_data(data_directory):
    directories = [
        d for d in os.listdir(data_directory)
        if os.path.isdir(os.path.join(data_directory, d))
    ]

    labels = []
    images = []
    for d in directories:
        label_directory = os.path.join(data_directory, d)
        filenames = [
            os.path.join(label_directory, f)
            for f in os.listdir(label_directory)
            if f.endswith('ppm')
        ]

        for f in filenames:
            images.append(skimage.data.imread(f))
            labels.append(int(d))

    return images, labels


def load_data(data_directory):
    images, labels = load_raw_data(data_directory)
    images28 = [transform.resize(image, (28, 28)) for image in images]
    # images28 = rgb2gray(np.array(images28))
    return np.array(images28, dtype=np.float32), np.array(labels)


def inspect_data():
    image_data, label_data = load_raw_data(TRAIN_DATA_PATH)
    images = np.array(image_data)
    labels = np.array(label_data)
    print('Dimension of images:', images.ndim)
    print('Number of images:', images.size)
    print('Sample image:', images[0])
    print('Image flags:', images.flags)
    print('Itemsize:', images.itemsize)
    print('Nbytes:', images.nbytes)

    print('Dimension of labels:', labels.ndim)
    print('Number of labels:', labels.size)
    print('Number of labels:', len(set(labels)))

    # Distribution of sign types
    plt.hist(labels, len(set(labels)))
    plt.show()

    # Random sample of signs
    samples = [200, 2310, 3453, 4000]
    for i in range(len(samples)):
        plt.subplot(1, 4, i+1)
        plt.axis('off')
        plt.imshow(images[samples[i]])
        plt.subplots_adjust(wspace=0.5)
        print("shape: {0}, min: {1}, max: {2}".format(
            images[samples[i]].shape,
            images[samples[i]].min(),
            images[samples[i]].max())
        )
    plt.show()

    # Displays image of sign along with count
    unique_labels = set(labels)
    for index, label in enumerate(unique_labels, 1):
        image = images[label_data.index(label)]
        plt.subplot(8, 8, index)
        plt.axis('off')
        plt.title("Label {0} ({1})".format(label, label_data.count(label)))
        plt.imshow(image)
    plt.show()

    # Rescaled images
    images28 = [transform.resize(image, (28, 28)) for image in images]
    for i in range(len(samples)):
        plt.subplot(1, 4, i+1)
        plt.axis('off')
        plt.imshow(images28[samples[i]])
        plt.subplots_adjust(wspace=0.5)
        print("shape: {0}, min: {1}, max: {2}".format(
            images28[samples[i]].shape,
            images28[samples[i]].min(),
            images28[samples[i]].max())
        )
    images28 = np.array(images28)
    print('Shape of resized images', images28.shape)
    plt.show()

    # Converted to grayscale
    images28 = rgb2gray(images28)
    for i in range(len(samples)):
        plt.subplot(1, 4, i+1)
        plt.axis('off')
        plt.imshow(images28[samples[i]], cmap="gray")
        plt.subplots_adjust(wspace=0.5)

    plt.show()
