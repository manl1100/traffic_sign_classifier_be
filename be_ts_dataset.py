import os
import skimage.data

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

images, labels = load_data(train_data_directory)
