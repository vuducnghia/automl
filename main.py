import os
import zipfile

import tensorflow as tf
from tensorflow import keras
import tensorflow_datasets as tfds
from utils.preprocessing import preprocess_data
from utils.encode_label import LabelEncoder
from model.model import setup_callback, ODHyperModel
from keras_tuner import RandomSearch

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)

BATCH_SIZE = 2
EPOCHS = 10


def create_dataset():
    (train_dataset, val_dataset, test_dataset), dataset_info = tfds.load("coco/2017",
                                                                         split=["train", "validation", "test"],
                                                                         with_info=True,
                                                                         data_dir="data")

    label_encoder = LabelEncoder()
    autotune = tf.data.AUTOTUNE

    train_dataset = train_dataset.map(preprocess_data, num_parallel_calls=autotune)
    train_dataset = train_dataset.shuffle(8 * BATCH_SIZE)
    train_dataset = train_dataset.padded_batch(BATCH_SIZE,
                                               padding_values=(0.0, 1e-8, -1),
                                               drop_remainder=True)
    train_dataset = train_dataset.map(label_encoder.encode_batch, num_parallel_calls=autotune)
    # train_dataset = train_dataset.apply(tf.data.experimental.ignore_errors())
    train_dataset = train_dataset.prefetch(autotune)

    val_dataset = val_dataset.map(preprocess_data, num_parallel_calls=autotune)
    val_dataset = val_dataset.shuffle(8 * BATCH_SIZE)
    val_dataset = val_dataset.padded_batch(BATCH_SIZE,
                                           padding_values=(0.0, 1e-8, -1),
                                           drop_remainder=True)
    val_dataset = val_dataset.map(label_encoder.encode_batch, num_parallel_calls=autotune)
    # val_dataset = val_dataset.apply(tf.data.experimental.ignore_errors())
    val_dataset = val_dataset.prefetch(autotune)

    test_dataset = test_dataset.map(preprocess_data, num_parallel_calls=autotune)
    test_dataset = test_dataset.padded_batch(BATCH_SIZE,
                                             padding_values=(0.0, 1e-8, -1),
                                             drop_remainder=True)
    test_dataset = test_dataset.map(label_encoder.encode_batch, num_parallel_calls=autotune)
    test_dataset = test_dataset.prefetch(autotune)

    return train_dataset, val_dataset, test_dataset, dataset_info

    # train_steps_per_epoch = dataset_info.splits["train"].num_examples // BATCH_SIZE
    # val_steps_per_epoch = dataset_info.splits["validation"].num_examples // BATCH_SIZE


def main():
    train_dataset, val_dataset, test_dataset, dataset_info = create_dataset()
    print(dataset_info.splits["train"].num_examples)
    print(dataset_info.splits["validation"].num_examples)
    print(dataset_info.splits["test"].num_examples)

    hypermodel = ODHyperModel(num_classes=80)
    tuner = RandomSearch(
        hypermodel,
        objective="val_loss",
        max_trials=2,
        executions_per_trial=1,
        overwrite=True,
        directory="my_dir",
        project_name="helloworld",
    )
    tuner.search_space_summary()
    tuner.search(
        val_dataset,
        epochs=EPOCHS,
        validation_data=test_dataset,
        # steps_per_epoch=steps_per_epoch,
        # validation_steps=validation_steps,
        # callbacks=setup_callback
    )
    best_model = tuner.get_best_models(num_models=1)[0]
    best_model.fit(train_dataset, )
    best_model.save("best_model")


if __name__ == "__main__":
    main()
