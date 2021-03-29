import datetime

import matplotlib
import tensorflow as tf
from keras_preprocessing.image import ImageDataGenerator
from tensorflow.python.keras import Input
from tensorflow.python.keras.applications.resnet import ResNet152

from ir_image_classification.feature_extraction.keras.keras_dataset import marvel_dataframe

matplotlib.use('Agg')
import matplotlib.pyplot as plt


def create_training_validation_plots(head_history, final_history, plot_name):
    amount_of_epochs = len(head_history['loss'] + final_history['loss'])

    # summarize history for accuracy
    plt.plot(head_history['categorical_accuracy'] + final_history['categorical_accuracy'])
    plt.plot(head_history['val_categorical_accuracy'] + final_history['val_categorical_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.xticks(range(amount_of_epochs))
    plt.savefig(f"{plot_name}_acc.png")
    plt.clf()

    # summarize history for loss
    plt.plot(head_history['loss'] + final_history['loss'])
    plt.plot(head_history['val_loss'] + final_history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.xticks(range(amount_of_epochs))
    plt.savefig(f"{plot_name}_loss.png")
    plt.clf()


def build_datasets(root_dir, batch_size=32, max_images_per_class=None):
    datagen = ImageDataGenerator(
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        rotation_range=5,
        preprocessing_function=tf.keras.applications.resnet.preprocess_input
    )

    train_df = marvel_dataframe(root_dir, is_train=True, max_images_per_class=max_images_per_class)
    train_generator = datagen.flow_from_dataframe(
        dataframe=train_df,
        directory=None,
        x_col="paths",
        y_col="labels",
        batch_size=batch_size,
        seed=42,
        shuffle=True,
        class_mode="categorical",
        target_size=(224, 224),
    )

    test_df = marvel_dataframe(root_dir, is_train=False)
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

    return train_generator, test_generator


def main():
    # Load the data
    # root_dir = '/home/gitaar9/AI/TNO/marveldataset2016/'
    root_dir = '/data/s2576597/MARVEL/'
    max_images_per_class = None
    train_generator, test_generator = build_datasets(root_dir, batch_size=100, max_images_per_class=max_images_per_class)

    # CREATE THE MODEL
    # load pretrained model without head
    base_model = ResNet152(
        include_top=False,
        weights="imagenet",
        pooling=None,
        input_tensor=Input(shape=(224, 224, 3))
    )

    # Freeze the base_model
    base_model.trainable = False

    # Create a new head model
    head_model = base_model.output
    head_model = tf.keras.layers.GlobalAveragePooling2D(name='avg_pool')(head_model)
    outputs = tf.keras.layers.Dense(26, activation='softmax', name='predictions')(head_model)

    # Combine the whole thing
    model = tf.keras.Model(base_model.inputs, outputs)
    model.summary()

    # Compile the model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.CategoricalAccuracy()],
    )

    # TRAINING PART
    # Train the head model
    epochs = 20  # 10
    head_training_history = model.fit(
        train_generator,
        epochs=epochs,
        validation_data=test_generator,
    )

    # Train the model as a whole for bit more
    base_model.trainable = True
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-5),
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.CategoricalAccuracy()],
    )
    final_epochs = 20  # 10
    final_training_history = model.fit(train_generator, epochs=final_epochs, validation_data=test_generator)

    # Create some final plots and save the model
    result_name = f"/data/s2576597/ImageClassificationIR_results/output/finetuned_{epochs}_{final_epochs}_{datetime.datetime.now().isoformat()}"
    if max_images_per_class:
        result_name += f"_mi_{max_images_per_class}"
    create_training_validation_plots(head_training_history.history, final_training_history.history, result_name)
    model.save(result_name)


if __name__ == "__main__":
    main()
