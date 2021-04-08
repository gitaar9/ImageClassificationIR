import datetime
import math

import matplotlib
import tensorflow as tf
from keras_preprocessing.image import ImageDataGenerator
from tensorflow.python.keras import Input
from tensorflow.python.keras.applications.resnet import ResNet152
from tensorflow.python.keras.models import load_model

from ir_image_classification.feature_extraction.keras.keras_dataset import marvel_dataframe
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics


def build_datasets(root_dir, batch_size=32, max_images_per_class=None):
    test_azimuth = []
    with open('/home/gitaar9/AI/TNO/StarMap/tools/test_images_angles.txt', 'r') as f:
        for l in f.readlines():
            name, a, e, t = l.strip().split(',')
            test_azimuth.append(float(a))
    test_azimuth = np.asarray(test_azimuth)
    mask = np.where(test_azimuth != 90)

    datagen = ImageDataGenerator(
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        rotation_range=5,
        preprocessing_function=tf.keras.applications.resnet.preprocess_input
    )
    print(test_azimuth.shape)
    test_azimuth = test_azimuth[mask]
    print(test_azimuth.shape)

    test_df = marvel_dataframe(root_dir, is_train=False)
    print(len(test_df))
    test_df = test_df.loc[mask]
    print(len(test_df))

    test_generator = datagen.flow_from_dataframe(
        dataframe=test_df,
        directory=None,
        x_col="paths",
        y_col="labels",
        batch_size=batch_size,
        seed=42,
        shuffle=False,
        class_mode="categorical",
        target_size=(224, 224),
    )

    return test_generator, test_azimuth


def main():
    # Load the data
    root_dir = '/home/gitaar9/AI/TNO/marveldataset2016/'
    # root_dir = '/data/s2576597/MARVEL/'
    max_images_per_class = None
    bs = 10
    test_generator, test_azimuths = build_datasets(root_dir, batch_size=bs, max_images_per_class=max_images_per_class)

    # CREATE THE MODEL
    # load pretrained model without head
    model = load_model('output/finetuned_20_20_2021-04-01T19:19:51.809959')
    print(model.summary())

    pred_ints = []
    label_ints = []

    for idx, (batch_data, batch_labels) in enumerate(test_generator):
        print(f"{idx}/{len(test_generator)}")
        azimuths = test_azimuths[idx * bs:idx * bs + bs]
        if len(azimuths) == 0:
            break
        batch_data = batch_data[:len(azimuths)]
        batch_labels = batch_labels[:len(azimuths)]
        output = model.predict(batch_data)
        int_labels = np.argmax(batch_labels, axis=1)
        int_preds = np.argmax(output, axis=1)
        pred_ints.extend(int_preds)
        label_ints.extend(int_labels)

    pred_ints = np.asarray(pred_ints)
    label_ints = np.asarray(label_ints)
    print(len(pred_ints))
    print(len(label_ints))
    print(len(test_azimuths))
    validation_accuracy = metrics.accuracy_score(label_ints, pred_ints)
    print("Validation Accuracy:", validation_accuracy)

    # load azimuth calculated using starmap
    def create_slices(limits=None, amount_of_slices=8, verbose=False):
        limits = limits or (-math.pi, math.pi)
        slices = []
        slice_size = abs(limits[0] - limits[1]) / amount_of_slices
        for i in range(amount_of_slices):
            slice = (limits[0] + i * slice_size, limits[0] + (i + 1) * slice_size)
            slices.append(slice)
        if verbose:
            print([(math.degrees(s_start), math.degrees(s_end)) for s_start, s_end in slices])
        return slices

    slices = create_slices(amount_of_slices=40)
    slice_accs = []
    xs = []
    for slice_start, slice_end in slices:
        x = math.degrees((slice_start + slice_end) / 2)
        xs.append(x)
        mask = np.where((slice_start <= test_azimuths) & (test_azimuths < slice_end))
        slice_pred = pred_ints[mask]
        slice_gt = label_ints[mask]
        slice_acc = metrics.accuracy_score(slice_gt, slice_pred) * 100
        slice_accs.append(slice_acc)
        print(int(x), slice_acc)
    plt.bar(["%.0f" % x for x in xs], slice_accs)
    plt.title('Average accuracy: {:.2f}'.format(validation_accuracy))
    plt.xlabel('Average bin azimuth in degrees')
    plt.ylabel('Average bin accuracy (%)')
    plt.xticks(rotation="vertical")
    plt.show()

    exit()


    # final_training_history = model.fit(train_generator, epochs=final_epochs, validation_data=test_generator)

    # Create some final plots and save the model


if __name__ == "__main__":
    main()
