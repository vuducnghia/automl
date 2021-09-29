import tensorflow as tf
from tensorflow import keras
import tensorflow_datasets as tfds

from model.loss import Loss
from utils.preprocessing import preprocess_data
from utils.encode_label import LabelEncoder
from model.model import setup_callback, ODHyperModel, ObjectDetectionNet
from keras_tuner import RandomSearch
from configs import NUM_CLASSES, BATCH_SIZE, EPOCHS

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)




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
    test_dataset = test_dataset.shuffle(8 * BATCH_SIZE)
    test_dataset = test_dataset.padded_batch(BATCH_SIZE,
                                             padding_values=(0.0, 1e-8, -1),
                                             drop_remainder=True)
    test_dataset = test_dataset.map(label_encoder.encode_batch, num_parallel_calls=autotune)
    # test_dataset = test_dataset.apply(tf.data.experimental.ignore_errors())
    test_dataset = test_dataset.prefetch(autotune)

    return train_dataset, val_dataset, test_dataset, dataset_info

    # train_steps_per_epoch = dataset_info.splits["train"].num_examples // BATCH_SIZE
    # val_steps_per_epoch = dataset_info.splits["validation"].num_examples // BATCH_SIZE


def main():
    train_dataset, val_dataset, test_dataset, dataset_info = create_dataset()
    print(dataset_info.splits["train"].num_examples)
    print(dataset_info.splits["validation"].num_examples)
    print(dataset_info.splits["test"].num_examples)

    # hypermodel = ODHyperModel(num_classes=80)
    # tuner = RandomSearch(
    #     hypermodel,
    #     objective="val_loss",
    #     max_trials=2,
    #     executions_per_trial=1,
    #     overwrite=True,
    #     directory="my_dir",
    #     project_name="helloworld",
    # )
    # tuner.search_space_summary()
    # tuner.search(
    #     train_dataset,
    #     epochs=1,
    #     validation_data=val_dataset,
    #     steps_per_epoch=1,
    #     validation_steps=1,
    #     # callbacks=setup_callback
    # )

    # best_model = tuner.get_best_models(num_models=1)[0]
    # best_model.fit(train_dataset, )
    # best_model.save("best_model")
    model = ObjectDetectionNet(None, NUM_CLASSES)
    # model.build((None,512, 512, 3))


    optimizer = tf.optimizers.SGD(learning_rate=0.01, momentum=0.9)
    loss_fn=Loss(NUM_CLASSES)
    model.compile(loss=loss_fn, optimizer=optimizer)
    model.fit(test_dataset, validation_data=val_dataset, steps_per_epoch=1, validation_steps=1, epochs=EPOCHS)
    model.save("my_model")


if __name__ == "__main__":
    main()
