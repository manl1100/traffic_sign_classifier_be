import os

import numpy as np
import skimage.data
import matplotlib.pyplot as plt

TRAIN_DATA_SET = 'http://btsd.ethz.ch/shareddata/BelgiumTSC/BelgiumTSC_Training.zip'
TEST_DATA_SET = 'http://btsd.ethz.ch/shareddata/BelgiumTSC/BelgiumTSC_Testing.zip'

ROOT_PATH = '/Users/manuelsanchez/Repositories/traffic_sign_classifier_be/'


def load_data(data_directory):
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


train_data_directory = os.path.join(ROOT_PATH, 'traffic_signs/Training')
test_data_directory = os.path.join(ROOT_PATH, 'traffic_signs/Testing')


def inspect_data():
    image_data, label_data = load_data(train_data_directory)
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

    plt.hist(labels, len(set(labels)))
    plt.show()

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

    unique_labels = set(labels)
    for index, label in enumerate(unique_labels, 1):
        image = images[label_data.index(label)]
        plt.subplot(8, 8, index)
        plt.axis('off')
        plt.title("Label {0} ({1})".format(label, label_data.count(label)))
        plt.imshow(image)
    plt.show()
